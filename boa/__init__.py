# -*- coding: utf-8 -*-

import argparse
import configparser
import contextlib
import datetime
import functools
import gettext
import itertools
import json
import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any, NamedTuple, TypeAlias

__all__ = [
    '__version__',
    'main',
    'Contest',
]
__version__ = '0.0.1'


"""
プログラム名
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
    locale_path = (Path(__file__) / '../locale').resolve()

    language = os.environ.get('LANG') or 'en_US'  # デフォルトの言語
    # 日本語
    if 'ja' == language or language.startswith('ja_JP'):
        language = 'ja_JP'

    translation = gettext.translation(
        domain,
        localedir=str(locale_path),
        languages=(language,),
        fallback=True,
    )
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


@contextlib.contextmanager
def benchmark() -> Callable[[], float]:
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
    @functools.wraps(fn)
    def wrapper(*args, **kwargs) -> Any:
        logger.debug('%s():ENTER' % fn.__qualname__)
        try:
            with benchmark() as timer:
                return fn(*args, **kwargs)
        finally:
            logger.debug('%s():LEAVE' % fn.__qualname__)
            logger.info('%(fn)s():%(elapsed_time)s=%(delta)s'
                        % {'fn': fn.__qualname__,
                           'elapsed_time': _('elapsed_time'),
                           'delta': datetime.timedelta(seconds=timer())})
    return wrapper


"""
D20のダイス目のイテレータ
"""
D20 = range(1, 21)


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
            config = configparser.ConfigParser(defaults=DEFAULTS._asdict())
            config.read_file(values)
            values.close()
            for section in config.sections():
                configs[section] = dict()
                for option in OPTIONS:
                    try:
                        value = config.get(section, option)
                        value = int(value)
                    except ValueError as error:
                        raise ValueError(
                            _('invalid int value in section=%(section)r, '
                              'option=%(option)r: %(value)r') % {
                                'section': section,
                                'option': option,
                                'value': value,
                              }) from error
                    if value not in getattr(CHOICES, option):
                        raise ValueError(
                            _('invalid choice in section=%(section)r, '
                              'option=%(option)r: %(value)r (choose from '
                              '%(choices)s)') % {
                                'section': section,
                                'option': option,
                                'value': value,
                                'choices': ', '.join(
                                    map(str, CHOICES[option])),
                              })
                    configs[section][option] = value
        except Exception as error:
            raise argparse.ArgumentError(self, error) from error
        setattr(namespace, self.dest, configs)


def main() -> None:
    """
    引数を解析して、Contestを実行する

    --log-level: ロギングレベル str
    --workers マルチプロセス数 int デフォルト cpu数

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

    cpus = os.cpu_count()

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
    parser.add_argument(
        '--workers',
        default=cpus,
        type=int,
        choices=range(1, cpus + 1),
        metavar='{1-%d}' % cpus,
        help=_('set number of multiprocessing workers'))

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

    for option in OPTIONS[1:]:  # --targetを除く
        parser.add_argument(
            '--%s' % option.replace('_', '-'),
            default=getattr(DEFAULTS, option),
            type=int,
            choices=getattr(CHOICES, option),
            metavar='{%d-%d}' % (getattr(CHOICES, option)[0],
                                 getattr(CHOICES, option)[-1]),
            help=getattr(HELPS, option))

    parser.add_argument(
        '--output',
        default=argparse.SUPPRESS,
        type=argparse.FileType('w', encoding='utf-8'),
        help=_('output JSON file'))
    args = parser.parse_args()

    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        '%(asctime)s:%(levelname)s:%(name)s:%(process)d:%(message)s'))
    if not logger.handlers:
        logger.addHandler(handler)
    if hasattr(args, 'log_level'):
        logger.setLevel(args.log_level)
        logger.info('%s=%r' % (_('log_level'), args.log_level))

    if not hasattr(args, 'input'):
        args.input = {
            'args': {
                option: getattr(args, option) for option in OPTIONS
            },
        }
    outcomes = tuple(
        Contest(**config, title=section).execute(args.workers)
        for section, config in args.input.items())
    if hasattr(args, 'output'):
        json.dump(outcomes, args.output, ensure_ascii=False, indent=2)
        args.output.close()


"""
行為判定のダイスのパターン
"""
Role: TypeAlias = tuple[int, ...]


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

    行為判定のダイス数に応じて全パターンを生成して、
    成功、失敗、クリティカル、ファンブル等の確率を算出する。

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
    def __init__(self, target: int, roll: int, critical: int, fumble: int,
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

    def generate(self) -> Iterator[Role]:
        """
        行為判定のダイスの全パターンを生成する

        Returns:
            行為判定のダイスのパターンのイテレータ
        """
        return itertools.product(D20, repeat=self.roll + self.opponent_roll)

    def generate_iterator(self, workers: int) -> Iterator[Iterator[Role]]:
        """
        行為判定のダイスの全パターンをworker数で分割したイテレータを生成する

        Args:
            workers: worker数 int

        Returns:
            "行為判定のダイスのパターンのイテレータ" のイテレータ
        """
        return (itertools.islice(self.generate(), n, None, workers)
                for n in range(workers))

    def contest(self, roll: Role) -> Result:
        """
        行為判定を行い、行為判定の結果を返す

        Args:
            roll: 行為判定のダイスのパターン

        Returns:
            行為判定の結果
        """
        your_roll = roll[:self.roll]
        opponent_roll = roll[self.roll:]

        if all(r >= self.fumble for r in your_roll):
            # fumble
            return 0, 0, 0, 1, 0, 0
        elif any(r <= self.critical for r in your_roll):
            # critical
            if all(r >= self.opponent_fumble for r in opponent_roll):
                # opponent fumble (critical)
                return 1, 0, 0, 0, 0, 0
            elif any(r <= self.opponent_critical for r in opponent_roll):
                # opponent critical
                return 0, 0, 0, 0, 1, 0
            else:
                # critical
                return 1, 0, 0, 0, 0, 0
        elif any(r <= self.target for r in your_roll):
            # success
            if all(r >= self.opponent_fumble for r in opponent_roll):
                # opponent fumble (success)
                return 0, 1, 0, 0, 0, 0
            elif any(r <= self.opponent_critical for r in opponent_roll):
                # opponent critical
                return 0, 0, 0, 0, 1, 0
            elif any(r <= min(your_roll) for r in opponent_roll):
                # opponent success
                return 0, 0, 0, 0, 0, 1
            else:
                # success
                return 0, 1, 0, 0, 0, 0
        else:
            # failure
            return 0, 0, 1, 0, 0, 0

    def reduce(self, iterable: Iterator[Result]) -> Result:
        """
        行為判定の結果のイテレータを reduce して単一の結果を返す

        Args:
            iterable: 行為判定の結果のイテレータ

        Returns:
            行為判定の結果
        """
        results = [0, 0, 0, 0, 0, 0]
        for result in iterable:
            results[0] += result[0]  # critical
            results[1] += result[1]  # success
            results[2] += result[2]  # failure
            results[3] += result[3]  # fumble
            results[4] += result[4]  # opponent critical
            results[5] += result[5]  # opponent success
        return results

    @trace
    def map_reduce(self, iterable: Iterator[Role]) -> Result:
        """
        行為判定のダイスのパターンのイテレータに対して行為判定を実行する

        行為判定の結果は reduce して単一の結果を返す。

        Args:
            iterable: 行為判定のダイスのパターンのイテレータ

        Returns:
            行為判定の結果
        """
        return self.reduce(map(self.contest, iterable))

    def execute(self, workers: int) -> dict:
        """
        行為判定を実行する

        Args:
            int: マルチプロセス数 int

        Returns:
            行為判定の確率の算出結果 dict
        """
        rolls = 20 ** (self.roll + self.opponent_roll)
        logger.info('%s=%d' % (_('rolls'), rolls))

        with benchmark() as timer:
            if 1 < workers and 400 < rolls:  # 2D20まではシングルの方が速い
                logger.info('%s=%d' % (_('workers'), workers))
                logger.info('%s/%s=%.0f' % (_('rolls'), _('workers'),
                                            rolls / workers))
                with ProcessPoolExecutor(max_workers=workers) as executor:
                    criticals, successes, failures, fumbles, \
                        opponent_criticals, opponent_successes \
                        = self.reduce(executor.map(
                            self.map_reduce, self.generate_iterator(workers)))

            else:
                criticals, successes, failures, fumbles, \
                    opponent_criticals, opponent_successes \
                    = self.map_reduce(self.generate())

        delta = datetime.timedelta(seconds=timer())
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
                _('workers'): workers,
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
        print(json.dumps(outcome, ensure_ascii=False, indent=2))
        return outcome
