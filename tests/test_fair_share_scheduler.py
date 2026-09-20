

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from aios.scheduler.fair_share_scheduler import StrideFairShareCore, STRIDE_BASE


def run_until_empty_refilling(core: StrideFairShareCore, agents, total_selections):
    
    for agent_id, weight in agents.items():
        core.enqueue(agent_id, f"{agent_id}-req-0", weight=weight)

    for _ in range(total_selections):
        agent_id, _ = core.select()
        core.enqueue(agent_id, f"{agent_id}-req-more", weight=None)

    return core.service_counts()


class TestBasicSelection:
    def test_single_agent_gets_every_request(self):
        core = StrideFairShareCore()
        core.enqueue("a1", "req1")
        core.enqueue("a1", "req2")
        core.enqueue("a1", "req3")

        got = [core.select()[1] for _ in range(3)]
        assert got == ["req1", "req2", "req3"]
        assert core.select() is None

    def test_equal_weights_alternate_evenly(self):
        core = StrideFairShareCore()
        core.enqueue("a1", "a1-1", weight=1)
        core.enqueue("a2", "a2-1", weight=1)
        core.enqueue("a1", "a1-2", weight=1)
        core.enqueue("a2", "a2-2", weight=1)

        order = [core.select()[0] for _ in range(4)]
        # Equal weight & equal starting pass -> ties broken by arrival,
        # so it should alternate a1, a2, a1, a2.
        assert order == ["a1", "a2", "a1", "a2"]

    def test_select_on_empty_returns_none(self):
        core = StrideFairShareCore()
        assert core.select() is None

    def test_zero_or_negative_weight_rejected(self):
        core = StrideFairShareCore()
        with pytest.raises(ValueError):
            core.enqueue("a1", "req", weight=0)
        with pytest.raises(ValueError):
            core.enqueue("a1", "req", weight=-2)


class TestTieBreaking:
    def test_ties_broken_by_arrival_order_deterministically(self):
        core = StrideFairShareCore()
        # Three agents, same weight, enqueued in a specific order --
        # first selection round should follow arrival order exactly.
        core.enqueue("c", "c-1", weight=2)
        core.enqueue("a", "a-1", weight=2)
        core.enqueue("b", "b-1", weight=2)

        first_pick = core.select()[0]
        assert first_pick == "c"  # earliest arrival among equal pass values


class TestWeightedShare:
    @pytest.mark.parametrize(
        "weights,total_selections",
        [
            ({"heavy": 2, "light": 1}, 3000),
            ({"heavy": 3, "light": 1}, 4000),
            ({"heavy": 4, "light": 1}, 5000),
        ],
    )
    def test_service_ratio_matches_weight_ratio(self, weights, total_selections):
        core = StrideFairShareCore()
        counts = run_until_empty_refilling(core, weights, total_selections)

        heavy_w, light_w = weights["heavy"], weights["light"]
        expected_ratio = heavy_w / light_w
        observed_ratio = counts["heavy"] / counts["light"]

        # Stride scheduling approximates the ratio; over a few thousand
        # selections it should be within 2% of the target.
        assert observed_ratio == pytest.approx(expected_ratio, rel=0.02)

    def test_three_agents_share_proportionally(self):
        core = StrideFairShareCore()
        weights = {"w1": 1, "w2": 2, "w3": 5}
        counts = run_until_empty_refilling(core, weights, total_selections=8000)

        per_weight_unit = {aid: counts[aid] / w for aid, w in weights.items()}
        values = list(per_weight_unit.values())
        # service_count / weight should be roughly equal across all
        # three agents if the split is proportional to weight.
        assert max(values) / min(values) < 1.02

    def test_deterministic_selection_sequence_is_reproducible(self):
        weights = {"a": 3, "b": 1}

        core1 = StrideFairShareCore()
        seq1 = _selection_sequence(core1, weights, 500)

        core2 = StrideFairShareCore()
        seq2 = _selection_sequence(core2, weights, 500)

        assert seq1 == seq2  # no randomness anywhere in stride selection


def _selection_sequence(core, weights, n):
    for agent_id, w in weights.items():
        core.enqueue(agent_id, f"{agent_id}-0", weight=w)
    seq = []
    for _ in range(n):
        agent_id, _ = core.select()
        seq.append(agent_id)
        core.enqueue(agent_id, f"{agent_id}-more")
    return seq


class TestNewAgentSeeding:
    def test_new_agent_not_seeded_at_zero_once_others_have_run(self):
        core = StrideFairShareCore()
        core.enqueue("veteran", "v-1", weight=1)
        core.select()  # veteran's pass_value is now > 0

        core.enqueue("veteran", "v-2", weight=1)
        core.select()  # advance veteran's pass further

        core.enqueue("newcomer", "n-1", weight=1)
        pass_values = core.pass_values()

        assert pass_values["newcomer"] == pytest.approx(min(pass_values.values()))
        assert pass_values["newcomer"] != 0.0  # not naively seeded at 0

    def test_first_agent_ever_seeded_at_zero(self):
        core = StrideFairShareCore()
        core.enqueue("first", "req")
        assert core.pass_values()["first"] == 0.0

    def test_newcomer_is_competitive_not_starved(self):
        core = StrideFairShareCore()
        # veteran runs far ahead
        core.enqueue("veteran", "v-1", weight=1)
        for _ in range(50):
            core.select()
            core.enqueue("veteran", "v-more", weight=1)

        core.enqueue("newcomer", "n-1", weight=1)
        pass_values = core.pass_values()
        # Newcomer should be seeded level with whoever is currently
        # most-deserving (the veteran, who was just selected), not
        # starting 50 strides behind.
        assert pass_values["newcomer"] == pytest.approx(pass_values["veteran"])

        chosen_agent, _ = core.select()  # veteran wins the exact tie (earlier arrival)
        assert chosen_agent == "veteran"
        chosen_agent, _ = core.select()  # newcomer goes next, immediately
        assert chosen_agent == "newcomer"


class TestStrideMath:
    def test_stride_is_inversely_proportional_to_weight(self):
        core = StrideFairShareCore()
        core.set_weight("double", weight=2)
        core.set_weight("single", weight=1)
        weights = core.weights()
        assert weights["double"] == 2
        assert weights["single"] == 1
        # stride_i = STRIDE_BASE / weight_i
        assert core._agents["double"].stride == pytest.approx(STRIDE_BASE / 2)
        assert core._agents["single"].stride == pytest.approx(STRIDE_BASE / 1)

    def test_updating_weight_changes_future_stride(self):
        core = StrideFairShareCore()
        core.enqueue("a1", "req", weight=1)
        core.select()
        original_stride = core._agents["a1"].stride

        core.set_weight("a1", weight=4)
        new_stride = core._agents["a1"].stride

        assert new_stride < original_stride  # higher weight -> smaller stride
        assert new_stride == pytest.approx(STRIDE_BASE / 4)