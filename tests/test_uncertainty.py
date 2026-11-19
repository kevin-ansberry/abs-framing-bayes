import numpy as np

from absbayes.uncertainty import change_summary


def test_known_shift_is_detected():
    rng = np.random.default_rng(0)
    s25 = rng.normal(0.0, 0.02, size=(400, 5))
    s26 = s25 + np.array([0.2, 0.0, -0.2, 0.0, 0.0]) + rng.normal(0, 0.02, size=(400, 5))
    mean, p_pos, lo, hi = change_summary(s25, s26)
    assert p_pos[0] > 0.99 and p_pos[2] < 0.01
    assert 0.05 < p_pos[1] < 0.95 and 0.05 < p_pos[3] < 0.95  # unshifted cells stay uncertain
    assert lo[0] > 0 and hi[2] < 0 and lo[1] < 0 < hi[1]
    assert mean.shape == (5,)


def test_tiny_precise_shift_is_not_meaningful():
    from absbayes.uncertainty import meaningful_change
    rng = np.random.default_rng(1)
    base = rng.normal(0, 0.001, size=(400, 3))
    shift = np.array([0.20, 0.01, -0.20])  # large, tiny-but-precise, large negative
    mask, p_up, p_down = meaningful_change(base, base + shift + rng.normal(0, 0.001, size=(400, 3)))
    assert list(mask) == [True, False, True]
    assert p_up[0] > 0.99 and p_down[2] > 0.99
