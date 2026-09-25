# -*- coding: utf-8 -*-

import json
import random
import sys
from itertools import product

import pytest

from boa import CHOICES, D20, Contest, _, main


def contest(self: Contest, roll: tuple[int, ...]) -> tuple[int, ...]:
    """
    ダイスの1パターンに対して行為判定を行い、行為判定の結果を返す

    Contest.count() の正解の基準として、ダイスの出目ごとに判定する。

    Args:
        self: 行為判定
        roll: 行為判定のダイスのパターン

    Returns:
        該当する結果だけが1の行為判定の結果
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


def brute_force(self: Contest) -> tuple[int, ...]:
    """
    ダイスの全パターンを判定して、行為判定の結果ごとのパターン数を数える

    Args:
        self: 行為判定

    Returns:
        行為判定の結果
    """
    patterns = product(D20, repeat=self.roll + self.opponent_roll)
    return tuple(map(sum, zip(*(contest(self, p) for p in patterns))))


def random_options(rng: random.Random, roll: int,
                   opponent_roll: int) -> dict:
    """
    ダイス数以外のオプションを選択肢から無作為に選ぶ

    Args:
        rng: 乱数生成器
        roll: 行為判定のダイス数 int
        opponent_roll: 対抗判定のダイス数 int

    Returns:
        Contestのキーワード引数 dict
    """
    return dict(
        target=rng.choice(CHOICES.target),
        roll=roll,
        critical=rng.choice(CHOICES.critical),
        fumble=rng.choice(CHOICES.fumble),
        opponent_roll=opponent_roll,
        opponent_critical=rng.choice(CHOICES.opponent_critical),
        opponent_fumble=rng.choice(CHOICES.opponent_fumble),
    )


def test_count_1d20():
    """1D20・目標値12は、1がクリティカル・2〜12が成功・20がファンブル"""
    c = Contest(target=12, roll=1, critical=1, fumble=20,
                opponent_roll=0, opponent_critical=1, opponent_fumble=20,
                title='1D20')
    assert c.count() == (1, 11, 7, 1, 0, 0)


@pytest.mark.parametrize('roll', (1, 2))
def test_count_without_opponent(roll):
    """対抗判定なしは、目標値・クリティカル値・ファンブル値の全組み合わせ"""
    mismatches = []
    for target, critical, fumble in product(
            CHOICES.target, CHOICES.critical, CHOICES.fumble):
        c = Contest(target=target, roll=roll, critical=critical,
                    fumble=fumble, opponent_roll=0, opponent_critical=1,
                    opponent_fumble=20, title='test')
        if c.count() != brute_force(c):
            mismatches.append((target, critical, fumble))
    assert mismatches == []


@pytest.mark.parametrize('options', (
    # クリティカル値がファンブル値以上
    dict(target=20, critical=15, fumble=10,
         opponent_critical=1, opponent_fumble=20),
    # 対抗判定のクリティカル値がファンブル値以上
    dict(target=12, critical=3, fumble=20,
         opponent_critical=15, opponent_fumble=10),
    # 目標値がクリティカル値未満
    dict(target=2, critical=5, fumble=20,
         opponent_critical=3, opponent_fumble=18),
    # ファンブル値1は常にファンブル
    dict(target=12, critical=1, fumble=1,
         opponent_critical=1, opponent_fumble=1),
    # クリティカル値20
    dict(target=12, critical=20, fumble=20,
         opponent_critical=20, opponent_fumble=20),
), ids=('critical>=fumble', 'opponent_critical>=opponent_fumble',
        'target<critical', 'fumble=1', 'critical=20'))
@pytest.mark.parametrize('roll,opponent_roll', ((1, 1), (2, 1), (1, 2)))
def test_count_boundary(options, roll, opponent_roll):
    """判定の優先順位が効く境界値"""
    c = Contest(**options, roll=roll, opponent_roll=opponent_roll,
                title='test')
    assert c.count() == brute_force(c)


@pytest.mark.parametrize('roll,opponent_roll,samples', (
    (1, 1, 500),
    (2, 1, 40),
    (1, 2, 40),
    (2, 2, 4),
))
def test_count_with_opponent(roll, opponent_roll, samples):
    """対抗判定ありは、オプションを無作為に選んで全パターンと照合する"""
    rng = random.Random(f'{roll}-{opponent_roll}')
    mismatches = []
    for _i in range(samples):
        options = random_options(rng, roll, opponent_roll)
        c = Contest(**options, title='test')
        if c.count() != brute_force(c):
            mismatches.append(options)
    assert mismatches == []


@pytest.mark.parametrize('roll', CHOICES.roll)
@pytest.mark.parametrize('opponent_roll', CHOICES.opponent_roll)
def test_count_total(roll, opponent_roll):
    """全パターンを生成できないダイス数でも、合計が全パターン数に一致する"""
    rng = random.Random(f'{roll}-{opponent_roll}')
    for _i in range(100):
        c = Contest(**random_options(rng, roll, opponent_roll), title='test')
        counts = c.count()
        assert all(0 <= n for n in counts)
        assert sum(counts) == 20 ** (roll + opponent_roll)


def test_execute():
    """算出結果の統計は count() のパターン数と一致する"""
    c = Contest(target=12, roll=4, critical=1, fumble=20,
                opponent_roll=2, opponent_critical=1, opponent_fumble=20,
                title='test')
    stats = c.execute()[_('stats')]
    assert stats[_('rolls')] == 20 ** 6
    assert (
        stats[_('criticals')], stats[_('successes')],
        stats[_('failures')], stats[_('fumbles')],
        stats[_('opponent_criticals')], stats[_('opponent_successes')],
    ) == c.count()


def test_main(monkeypatch, tmp_path):
    """--target で行為判定を実行して、--output に出力する"""
    output = tmp_path / 'output.json'
    monkeypatch.setattr(
        sys, 'argv', ['boa', '--target', '12', '--output', str(output)])
    main()
    outcomes = json.loads(output.read_text(encoding='utf-8'))
    assert [outcome[_('title')] for outcome in outcomes] == ['args']


def test_main_workers(monkeypatch):
    """--workers は受け付けない"""
    monkeypatch.setattr(
        sys, 'argv', ['boa', '--target', '12', '--workers', '2'])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
