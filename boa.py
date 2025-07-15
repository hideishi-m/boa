import argparse
import configparser
import contextlib
import datetime
import functools
import itertools
import json
import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor
from collections.abc import Callable, Iterator
from types import MappingProxyType
from typing import Any


"""
ロガー
"""
logger = logging.getLogger(__name__)


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
        logger.debug(f'{fn.__qualname__}():ENTER')
        try:
            with benchmark() as timer:
                return fn(*args, **kwargs)
        finally:
            logger.debug(f'{fn.__qualname__}():LEAVE')
            delta = datetime.timedelta(seconds=timer())
            logger.info(f'{fn.__qualname__}():elapsed time={delta}')
    return wrapper


"""
行為判定のダイスのパターン
"""
Role = tuple[int, ...]


"""
行為判定の結果
"""
Result = tuple[int, int, int, int, int, int]


"""
D20のダイス目のイテレータ
"""
D20 = range(1, 21)


def main() -> None:
    """
    引数を解析して、Contestを実行する

    --loglevel: ロギングレベル str
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
    defaults = MappingProxyType({
        # 'target': デフォルトなし
        'roll': 1,
        'critical': 1,
        'fumble': 20,
        'opponent_roll': 0,
        'opponent_critical': 1,
        'opponent_fumble': 20,
    })
    options = ('target', *defaults,)
    choices = MappingProxyType({
        'target': (D20),
        'roll': (range(1, 6)),
        'critical': (D20),
        'fumble': (D20),
        'opponent_roll': (range(0, 6)),
        'opponent_critical': (D20),
        'opponent_fumble': (D20),
    })
    metavars = MappingProxyType({
        option: '{{{}-{}}}'.format(choices[option][0], choices[option][-1])
        for option in choices
    })

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
                config = configparser.ConfigParser(defaults=defaults)
                config.read_file(values)
                values.close()
                for section in config.sections():
                    configs[section] = dict()
                    for option in options:
                        try:
                            value = config.get(section, option)
                            value = int(value)
                        except ValueError as error:
                            raise ValueError(
                                f'invalid int value in {section=}, '
                                f'{option=}: {value!r}') from error
                        if value not in choices[option]:
                            raise ValueError(
                                f'invalid choice in {section=}, '
                                f'{option=}: {value} (choose from '
                                '{})'.format(', '.join(
                                    map(str, choices[option]))))
                        configs[section][option] = value
            except Exception as error:
                raise argparse.ArgumentError(self, error) from error
            setattr(namespace, self.dest, configs)

    cpus = os.cpu_count()

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        allow_abbrev=False)
    parser.add_argument(
        '--loglevel',
        choices=('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'),
        help='set logging level')
    parser.add_argument(
        '--workers', default=cpus, type=int, choices=range(1, cpus + 1),
        metavar='{{{}-{}}}'.format(1, cpus),
        help='set number of multiprocessing workers')

    group = parser.add_argument_group(title='mandatory arguments')
    required_group = group.add_mutually_exclusive_group(required=True)
    required_group.add_argument(
        '--target', default=argparse.SUPPRESS,
        type=int, choices=choices['target'], metavar=metavars['target'],
        help='your target')
    required_group.add_argument(
        '--input', action=InputAction, default=argparse.SUPPRESS,
        type=argparse.FileType('r', encoding='utf-8'),
        help='input INI file')

    for option in defaults:
        parser.add_argument(
            '--{}'.format(option.replace('_', '-')),
            default=defaults[option], type=int, choices=choices[option],
            metavar=metavars[option],
            help="opponent's {}".format(option.removeprefix('opponent_'))
                 if option.startswith('opponent_') else f'your {option}')

    parser.add_argument(
        '--output', type=argparse.FileType('w', encoding='utf-8'),
        help='output JSON file')
    args = parser.parse_args()

    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        '%(asctime)s:%(levelname)s:%(process)d:%(message)s'))
    if not logger.handlers:
        logger.addHandler(handler)
    if args.loglevel:
        logger.setLevel(args.loglevel)
        logger.info(f'loglevel={args.loglevel!r}')

    if not args.input:
        args.input = {
            'args': {
                option: getattr(args, option) for option in options
            },
        }
    outcomes = tuple(
        Contest(**config, title=section).execute(args.workers)
        for section, config in args.input.items())
    if args.output:
        json.dump(outcomes, args.output, ensure_ascii=False, indent=2)
        args.output.close()


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

        行為判定の結果は6つの要素を持つintのtupleである。
            result[0]: クリティカル
            result[1]: 成功
            result[2]: 失敗
            result[3]: ファンブル
            result[4]: 対抗判定のクリティカル
            result[5]: 対抗判定の成功

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
        logger.info(f'{rolls=}')

        with benchmark() as timer:
            if 1 < workers and 400 < rolls:  # 2D20まではシングルの方が速い
                logger.info(f'{workers=}')
                logger.info(f'rolls/worker={rolls / workers:.0f}')
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
        logger.info(f'elapsed time={delta}')

        assert rolls == (criticals + successes + failures + fumbles
                         + opponent_criticals + opponent_successes)

        outcome = {
            'title': self.title,
            'input': {
                'target': self.target,
                'roll': self.roll,
                'critical': self.critical,
                'fumble': self.fumble,
                'opponent_roll': self.opponent_roll,
                'opponent_critical': self.opponent_critical,
            },
            'output': {
                'p(criticals)': f'{criticals / rolls:.3%}',
                'p(successes)': f'{successes / rolls:.3%}',
                'p(failures)': f'{failures / rolls:.3%}',
                'p(fumbles)': f'{fumbles / rolls:.3%}',
                'p(opponent_criticals)': f'{opponent_criticals / rolls:.3%}',
                'p(opponent_successes)': f'{opponent_successes / rolls:.3%}',
            },
            'stats': {
                'workers': workers,
                'elapsed_time': str(delta),
                'rolls': rolls,
                'criticals': criticals,
                'successes': successes,
                'failures': failures,
                'fumbles': fumbles,
                'opponent_criticals': opponent_criticals,
                'opponent_successes': opponent_successes,
            },
        }
        print(json.dumps(outcome, ensure_ascii=False, indent=2))
        return outcome


if __name__ == '__main__':
    main()
