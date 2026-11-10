import numpy as np
import pytest

import samplers as s

LOGITS = np.log(np.array([0.5, 0.3, 0.15, 0.05]))


def test_temperature_one_is_identity():
    assert s.filtered(LOGITS) == pytest.approx([0.5, 0.3, 0.15, 0.05])


def test_low_temperature_approaches_greedy():
    assert s.filtered(LOGITS, temperature=0.01)[0] == pytest.approx(1.0)


def test_high_temperature_approaches_uniform():
    assert s.filtered(LOGITS, temperature=1000) == pytest.approx([0.25] * 4, abs=0.01)


def test_temperature_two_matches_hand_computation():
    expected = np.sqrt([0.5, 0.3, 0.15, 0.05])
    assert s.filtered(LOGITS, temperature=2) == pytest.approx(expected / expected.sum())


def test_top_k_keeps_exactly_k():
    probs = s.filtered(LOGITS, top_k=2)
    assert (probs > 0).sum() == 2
    assert probs[0] == pytest.approx(0.5 / 0.8)


def test_top_p_keeps_smallest_covering_set():
    assert (s.filtered(LOGITS, top_p=0.75) > 0).tolist() == [True, True, False, False]
    assert (s.filtered(LOGITS, top_p=0.81) > 0).tolist() == [True, True, True, False]


def test_top_p_always_keeps_at_least_one_token():
    assert (s.filtered(LOGITS, top_p=0.01) > 0).sum() == 1


def test_min_p_is_relative_to_the_top_token():
    # threshold = 0.5 * 0.5 = 0.25, so 0.5 and 0.3 survive
    assert (s.filtered(LOGITS, min_p=0.5) > 0).tolist() == [True, True, False, False]


def test_min_p_adapts_to_confidence_where_top_p_does_not():
    flat = np.log(np.full(1000, 1 / 1000))
    assert (s.filtered(flat, top_p=0.9) > 0).sum() >= 900
    peaked = np.log(np.array([0.9] + [0.1 / 999] * 999))
    assert (s.filtered(peaked, min_p=0.1) > 0).sum() == 1


def test_filters_compose_and_renormalise():
    probs = s.filtered(LOGITS, temperature=1.5, top_k=3, min_p=0.1)
    assert probs.sum() == pytest.approx(1.0)


def test_greedy_ignores_sampling():
    assert s.greedy(LOGITS) == 0


@pytest.mark.parametrize(
    "kwargs",
    [{"temperature": 0}, {"top_k": 0}, {"top_p": 0}, {"top_p": 1.5}, {"min_p": -0.1}],
)
def test_invalid_parameters_are_rejected(kwargs):
    with pytest.raises(ValueError):
        s.filtered(LOGITS, **kwargs)
