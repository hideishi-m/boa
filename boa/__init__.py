# -*- coding: utf-8 -*-

import argparse
import gettext
import json
import logging
import os
import re
import time
from collections.abc import Callable
from configparser import ConfigParser
from contextlib import AbstractContextManager, contextmanager
from datetime import timedelta
from functools import wraps
from importlib.resources import as_file, files
from typing import Any, NamedTuple, TypeAlias

__all__ = [
    '__version__',
    'main',
    'Contest',
]
__version__ = '0.0.1'


"""
パッケージ名
"""
package_name = __name__.partition('.')[0]


def get_gettext(domain: str) -> Callable[[str], str]:
    """
    gettext関数 _() を返す

    環境変数LANGから言語を取得し、日本語の場合は'ja_JP'を返す
    環境変数LANGがない、または、それ以外の言語はデフォルトの言語
    'en_US'を返す

    Args:
        domain (str): ドメイン

    Returns:
        Callable[[str, sty]: gettext関数 _()
    """
    locale_traversable = files(package_name).joinpath('locale')

    language = os.environ.get('LANG') or 'en_US'  # デフォルトの言語
    # 日本語
    if 'ja' == language or language.startswith('ja_JP'):
        language = 'ja_JP'

    with as_file(locale_traversable) as locale_path:
        translation = gettext.translation(
            domain, localedir=str(locale_path), languages=(language,),
            fallback=True)
        return translation.gettext


"""
デフォルトのgettext関数
"""
_ = get_gettext(package_name)


"""
argparseのgettext関数をオーバーライド
"""
argparse._ = get_gettext('argparse')


def getLogger(name: str) -> logging.Logger:
    """
    ロガーを返す

    Args:
        name: ロガー名

    Returns:
        logging.Logger
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        '%(asctime)s:%(levelname)s:%(name)s:%(process)d:%(message)s'))
    logger.addHandler(handler)
    return logger


"""
ロガー
"""
logger = getLogger(package_name)


@contextmanager
def benchmark() -> AbstractContextManager[Callable[[], float]]:
    """
    経過時間を測定するコンテキストマネージャ

    以下のように取得する。

        with benchmark() as timer:
            ...
        t = timer()

    Yields:
        floatを返す関数
    """
    start = stop = time.perf_counter()
    yield lambda: stop - start
    stop = time.perf_counter()


def trace(fn: Callable) -> Callable:
    """
    関数のENTER/LEAVEをロギングするデコレータ

    DEBUG: 関数のENTER/LEAVEを表示
    INFO: 関数の経過時間を表示
    """
    @wraps(fn)
    def wrapper(*args, **kwargs) -> Any:
        logger.debug('%s():ENTER' % fn.__qualname__)
        try:
            with benchmark() as timer:
                return fn(*args, **kwargs)
        finally:
            logger.debug('%s():LEAVE' % fn.__qualname__)
            logger.info(
                '%(fn)s():%(elapsed_time)s=%(delta)s'
                % {'fn': fn.__qualname__,
                   'elapsed_time': _('elapsed_time'),
                   'delta': timedelta(seconds=timer())})
    return wrapper


"""
D20のダイス目のイテレータ
"""
D20 = range(1, 21)


def count_min_at_least(value: int, dice: int) -> int:
    """
    D20をdice個振って、出目の最小値がvalue以上になるパターン数を返す

    diceが0の場合は常に1を返す。出目が空の場合にall()が真になるのと
    一致する。

    Args:
        value: 最小値の下限 int
        dice: ダイス数 int

    Returns:
        パターン数 int
    """
    return (21 - min(max(value, 1), 21)) ** dice


def count_min_at_most(value: int, dice: int) -> int:
    """
    D20をdice個振って、出目の最小値がvalue以下になるパターン数を返す

    diceが0の場合は常に0を返す。出目が空の場合にany()が偽になるのと
    一致する。

    Args:
        value: 最小値の上限 int
        dice: ダイス数 int

    Returns:
        パターン数 int
    """
    return 20 ** dice - count_min_at_least(value + 1, dice)


class Options(NamedTuple):
    """
    オプションの名前付きタプル
    """
    target: int | tuple[int] | str
    roll: int | tuple[int] | str
    critical: int | tuple[int] | str
    fumble: int | tuple[int] | str
    opponent_roll: int | tuple[int] | str
    opponent_critical: int | tuple[int] | str
    opponent_fumble: int | tuple[int] | str


"""
オプションのフィールド名のタプル
"""
OPTIONS = Options._fields


"""
オプションのデフォルト値の名前付きタプル
"""
DEFAULTS = Options(
    target=argparse.SUPPRESS,  # デフォルトなし
    roll=1,
    critical=1,
    fumble=20,
    opponent_roll=0,
    opponent_critical=1,
    opponent_fumble=20,
)


"""
オプションの選択肢の名前付きタプル
"""
CHOICES = Options(
    target=(D20),
    roll=(range(1, 6)),
    critical=(D20),
    fumble=(D20),
    opponent_roll=(range(0, 6)),
    opponent_critical=(D20),
    opponent_fumble=(D20),
)


"""
オプションのヘルプの名前付きタプル
"""
HELPS = Options(
    target=_('your %s') % _('target'),
    roll=_('your %s') % _('roll'),
    critical=_('your %s') % _('critical'),
    fumble=_('your %s') % _('fumble'),
    opponent_roll=_("opponent's %s") % _('roll'),
    opponent_critical=_("opponent's %s") % _('critical'),
    opponent_fumble=_("opponent's %s") % _('fumble'),
)


"""
-/+表記の正規表現
"""
MODIFIER_PATTERN = re.compile(r'[-+]\d+')


class InputAction(argparse.Action):
    """
    --inputオプションを処理するargparse.Actionクラス
    """

    def __call__(self, parser, namespace, values, option_string=None):
        """
        --inputオプションに指定したINIファイルのパスを読み込む

        Args:
            parser: argparse.ArgumentParserオブジェクト
            namespace: argparse.Namespaceオブジェクト
            values: INIファイルのファイルオブジェクト
            option_string: オプション文字列 デフォルトNone

        Raises:
            argparse.ArgumentError
        """
        configs = dict()
        try:
            config = ConfigParser(defaults=DEFAULTS._asdict())
            config.read_file(values)
            values.close()

            # デフォルトを読み込み
            defaults = dict()
            try:
                for option, value in config.defaults().items():
                    defaults[option] = int(value)
            except ValueError as error:
                raise ValueError(
                    _('invalid int value in section=%(section)r, '
                      'option=%(option)r: %(value)r')
                    % {
                        'section': config.default_section,
                        'option': option,
                        'value': value,
                    }) from error

            # セクションを読み込み
            for section in config.sections():
                # ファンブルの初期値
                roll_fumble = opponent_roll_fumble = False

                configs[section] = dict()
                for option in OPTIONS:
                    value = config.get(section, option)
                    try:
                        # 値が+/-の形式
                        if MODIFIER_PATTERN.match(value):
                            value = int(value) + defaults[option]
                            if 'target' == option:
                                # 最大20
                                value = min(value, 20)
                            elif 'roll' == option:
                                # 判定0以下はファンブル値15
                                roll_fumble = value < 1
                                # 最低1
                                value = max(value, 1)
                            elif 'critical' == option:
                                value = max(value, 1)   # 最低1
                                value = min(value, 19)  # 最大19
                            elif 'fumble' == option:
                                value = max(value, 2)   # 最低2
                                value = min(value, 20)  # 最大20
                            elif 'opponent_roll' == option:
                                if int(defaults['opponent_roll']):
                                    # 判定0以下はファンブル値15
                                    opponent_roll_fumble = value < 1
                                    # 判定があれば最低1
                                    value = max(value, 1)
                            elif 'opponent_critical' == option:
                                value = max(value, 1)   # 最低1
                                value = min(value, 19)  # 最大19
                            elif 'opponent_fumble' == option:
                                value = max(value, 2)  # 最低2
                                value = min(value, 20)  # 最大20
                            else:
                                raise ValueError(
                                    _('Unknown option %s') % option)
                        # 値が数値の形式
                        else:
                            value = int(value)
                    except ValueError as error:
                        raise ValueError(
                            _('invalid int value in section=%(section)r, '
                              'option=%(option)r: %(value)r')
                            % {
                                'section': section,
                                'option': option,
                                'value': value,
                            }) from error
                    if value not in getattr(CHOICES, option):
                        raise ValueError(
                            _('invalid choice in section=%(section)r, '
                              'option=%(option)r: %(value)r (choose from '
                              '%(choices)s)')
                            % {
                                'section': section,
                                'option': option,
                                'value': value,
                                'choices': ', '.join(
                                    map(str, getattr(CHOICES, option))),
                            })
                    configs[section][option] = value

                # ファンブルの調整
                if roll_fumble:
                    # 判定0以下はファンブル値15
                    configs[section]['fumble'] = min(
                        15, configs[section]['fumble'])
                if opponent_roll_fumble:
                    # 判定0以下はファンブル値15
                    configs[section]['opponent_fumble'] = min(
                        15, configs[section]['opponent_fumble'])

        except Exception as error:
            logger.exception(error)
            raise argparse.ArgumentError(self, error) from error
        setattr(namespace, self.dest, configs)


def main() -> None:
    """
    引数を解析して、Contestを実行する

    --log-level: ロギングレベル str

    以下、排他、かつ、必須
    --target: 目標値 int
    --input: 入力INIファイル str

    --roll: 行為判定のダイス数 int デフォルト 1
    --critical: 行為判定のクリティカル値 int デフォルト 1
    --fumble: 行為判定のファンブル値 int デフォルト 20
    --opponent-roll: 対抗判定のダイス数 int デフォルト 0
    --opponent-critical: 対抗判定のクリティカル値 int デフォルト 1
    --opponent-fumble: 対抗判定のファンブル値 int デフォルト 20
    --title: 行為判定の名前 str デフォルト "argparse"

    --output: 出力JSONファイル str
    """

    parser = argparse.ArgumentParser(
        prog=package_name,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        allow_abbrev=False)
    parser.add_argument(
        '-V',
        '--version',
        action='version',
        version=f'%(prog)s {__version__}')
    parser.add_argument(
        '--log-level',
        default=argparse.SUPPRESS,
        choices=('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'),
        help=_('set logging level'))

    group = parser.add_argument_group(
        title=_('mandatory arguments'))
    required_group = group.add_mutually_exclusive_group(
        required=True)
    required_group.add_argument(
        '--target',
        default=DEFAULTS.target,
        type=int,
        choices=CHOICES.target,
        metavar='{%d-%d}' % (CHOICES.target[0], CHOICES.target[-1]),
        help=HELPS.target)
    required_group.add_argument(
        '--input',
        action=InputAction,
        default=argparse.SUPPRESS,
        type=argparse.FileType('r', encoding='utf-8'),
        help=_('input INI file'))

    optional_group = parser.add_argument_group(
        title=_('optional arguments'))
    for option in OPTIONS[1:]:  # --targetを除く
        optional_group.add_argument(
            '--%s' % option.replace('_', '-'),
            default=getattr(DEFAULTS, option),
            type=int,
            choices=getattr(CHOICES, option),
            metavar='{%d-%d}' % (getattr(CHOICES, option)[0],
                                 getattr(CHOICES, option)[-1]),
            help=getattr(HELPS, option))

    optional_group.add_argument(
        '--output',
        default=argparse.SUPPRESS,
        type=argparse.FileType('w', encoding='utf-8'),
        help=_('output JSON file'))
    args = parser.parse_args()

    if hasattr(args, 'log_level'):
        logger.setLevel(args.log_level)
        logger.info('%s=%r' % (_('log_level'), args.log_level))

    if not hasattr(args, 'input'):
        args.input = {
            'args': {option: getattr(args, option) for option in OPTIONS},
        }
    outcomes = tuple(
        Contest(**config, title=section).execute()
        for section, config in args.input.items())
    if hasattr(args, 'output'):
        json.dump(outcomes, args.output, ensure_ascii=False, indent=2)
        args.output.close()


"""
行為判定の結果

行為判定の結果は6つの要素を持つintのタプルである。
    [0]: クリティカル
    [1]: 成功
    [2]: 失敗
    [3]: ファンブル
    [4]: 対抗判定のクリティカル
    [5]: 対抗判定の成功
"""
Result: TypeAlias = tuple[int, int, int, int, int, int]


class Contest:
    """
    Brade of Arcanaの行為判定を行うクラス

    行為判定の結果はダイスの最小値だけで決まるので、最小値の分布から
    成功、失敗、クリティカル、ファンブル等のパターン数を算術的に求めて、
    確率を算出する。

    対抗判定のダイス数 opponent_roll が 0 の場合は、
    自身の行為判定の結果のみで算出する。

    Attributes:
        target: 目標値 int
        roll: 行為判定のダイス数 int
        critical: 行為判定のクリティカル値 int
        fumble: 行為判定のファンブル値 int
        opponent_roll: 対抗判定のダイス数 int
        opponent_critical: 対抗判定のクリティカル値 int
        opponent_fumble: 対抗判定のファンブル値 int
        title: 行為判定の名前 str
    """
    def __init__(
            self, target: int, roll: int, critical: int, fumble: int,
            opponent_roll: int, opponent_critical: int,
            opponent_fumble: int, *, title: str) -> None:
        """
        初期化

        Args:
            target: 目標値 int
            roll: 行為判定のダイス数 int
            critical: 行為判定のクリティカル値 int
            fumble: 行為判定のファンブル値 int
            opponent_roll: 対抗判定のダイス数 int
            opponent_critical: 対抗判定のクリティカル値 int
            opponent_fumble: 対抗判定のファンブル値 int
            title: 行為判定の名前 str
        """
        self.target = target
        self.roll = roll
        self.critical = critical
        self.fumble = fumble
        self.opponent_roll = opponent_roll
        self.opponent_critical = opponent_critical
        self.opponent_fumble = opponent_fumble
        self.title = title

    @trace
    def count(self) -> Result:
        """
        行為判定の結果ごとのダイスのパターン数を算出する

        自身のダイスの出目の最小値を a 、対抗判定のダイスの出目の最小値を b
        とすると、行為判定の結果は以下の順に判定する。対抗判定のダイス数が
        0 の場合は、対抗判定は常にファンブルとして扱う。

            a >= fumble: ファンブル
            a <= critical:
                b >= opponent_fumble: クリティカル
                b <= opponent_critical: 対抗判定のクリティカル
                それ以外: クリティカル
            a <= target:
                b >= opponent_fumble: 成功
                b <= opponent_critical: 対抗判定のクリティカル
                b <= a: 対抗判定の成功
                それ以外: 成功
            それ以外: 失敗

        Returns:
            行為判定の結果
        """
        your_rolls = 20 ** self.roll
        opponent_rolls = 20 ** self.opponent_roll

        # ファンブルを先に判定するので、クリティカルと成功になる最小値は
        # ファンブル値未満に限られる
        critical_limit = min(self.critical, self.fumble - 1)
        target_limit = min(self.target, self.fumble - 1)
        opponent_critical_limit = min(
            self.opponent_critical, self.opponent_fumble - 1)

        your_fumbles = count_min_at_least(self.fumble, self.roll)
        your_criticals = count_min_at_most(critical_limit, self.roll)
        your_successes = max(
            0, count_min_at_most(target_limit, self.roll) - your_criticals)
        your_failures = (
            your_rolls - your_fumbles - your_criticals - your_successes)

        opponent_criticals = count_min_at_most(
            opponent_critical_limit, self.opponent_roll)

        # 自身の最小値 a ごとに、対抗判定が a 以下で成功するパターン数
        opponent_successes = 0
        for your_min in range(critical_limit + 1, target_limit + 1):
            your_patterns = (
                count_min_at_least(your_min, self.roll)
                - count_min_at_least(your_min + 1, self.roll))
            opponent_patterns = max(0, count_min_at_most(
                min(your_min, self.opponent_fumble - 1),
                self.opponent_roll) - opponent_criticals)
            opponent_successes += your_patterns * opponent_patterns

        return (
            your_criticals * (opponent_rolls - opponent_criticals),
            your_successes * (opponent_rolls - opponent_criticals)
            - opponent_successes,
            your_failures * opponent_rolls,
            your_fumbles * opponent_rolls,
            (your_criticals + your_successes) * opponent_criticals,
            opponent_successes,
        )

    def execute(self) -> dict:
        """
        行為判定を実行する

        Returns:
            行為判定の確率の算出結果 dict
        """
        rolls = 20 ** (self.roll + self.opponent_roll)
        logger.info('%s=%d' % (_('rolls'), rolls))

        with benchmark() as timer:
            criticals, successes, failures, fumbles, \
                opponent_criticals, opponent_successes = self.count()

        delta = timedelta(seconds=timer())
        logger.info('%s=%s' % (_('elapsed_time'), delta))

        assert rolls == (criticals + successes + failures + fumbles
                         + opponent_criticals + opponent_successes)

        outcome = {
            _('title'): self.title,
            _('input'): {
                _('target'): self.target,
                _('roll'): self.roll,
                _('critical'): self.critical,
                _('fumble'): self.fumble,
                _('opponent_roll'): self.opponent_roll,
                _('opponent_critical'): self.opponent_critical,
                _('opponent_fumble'): self.opponent_fumble,
            },
            _('output'): {
                _('p(criticals)'): f'{criticals / rolls:.3%}',
                _('p(successes)'): f'{successes / rolls:.3%}',
                _('p(failures)'): f'{failures / rolls:.3%}',
                _('p(fumbles)'): f'{fumbles / rolls:.3%}',
                _('p(opponent_criticals)'):
                f'{opponent_criticals / rolls:.3%}',
                _('p(opponent_successes)'):
                f'{opponent_successes / rolls:.3%}',
            },
            _('stats'): {
                _('elapsed_time'): str(delta),
                _('rolls'): rolls,
                _('criticals'): criticals,
                _('successes'): successes,
                _('failures'): failures,
                _('fumbles'): fumbles,
                _('opponent_criticals'): opponent_criticals,
                _('opponent_successes'): opponent_successes,
            },
        }
        assert (
            criticals + successes + failures + fumbles
            + opponent_criticals + opponent_successes) \
            == rolls, _('verification')
        print(json.dumps(outcome, ensure_ascii=False, indent=2))
        return outcome
