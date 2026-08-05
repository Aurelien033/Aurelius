from src.model.tapered_transformer import cosine_taper_dff, tapered_dff_schedule


def test_cosine_taper_preserves_average_for_7b_config():
    baseline = 14336
    n_layers = 40
    schedule = tapered_dff_schedule(baseline, n_layers)

    assert schedule[0] == 21504
    assert schedule[-1] == 7168
    assert abs(sum(schedule) / len(schedule) - baseline) <= 1e-9


def test_cosine_taper_is_monotone_nonincreasing():
    schedule = tapered_dff_schedule(5632, 24)
    assert all(a >= b for a, b in zip(schedule, schedule[1:]))


def test_cosine_taper_single_layer_returns_baseline():
    assert cosine_taper_dff(5632, 0, 1) == 5632
