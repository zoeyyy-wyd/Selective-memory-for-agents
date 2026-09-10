import math
from datetime import datetime, timedelta

from smem.hawkes import HawkesIntensity, hawkes_keys, source_weight, specificity
from smem.schemas import Episode, Fact

T0 = datetime(2023, 1, 1)


def test_intensity_rises_with_mentions_and_decays():
    h = HawkesIntensity(alpha=0.5, beta=0.1)
    h.observe("boston", T0)
    fresh = h.intensity("boston", T0)
    later = h.intensity("boston", T0 + timedelta(days=30))
    assert fresh > later > h.mu_floor
    h.observe("boston", T0 + timedelta(days=30))
    assert h.intensity("boston", T0 + timedelta(days=30)) > fresh * 0.9
    assert h.intensity("never", T0) == h.mu_floor


def test_repeated_entity_beats_one_off():
    h = HawkesIntensity(alpha=0.5, beta=0.03)
    for d in (0, 10, 20, 30, 40):
        h.observe("user/city", T0 + timedelta(days=d))
    h.observe("ticket", T0 + timedelta(days=5))
    now = T0 + timedelta(days=45)
    assert h.intensity("user/city", now) > h.intensity("ticket", now)


def test_fit_returns_grid_values_with_finite_likelihood():
    h = HawkesIntensity()
    for e, days in {"a": [0, 1, 2, 30, 31], "b": [5, 40]}.items():
        for d in days:
            h.observe(e, T0 + timedelta(days=d))
    ll = h.log_likelihood(0.5, 0.1)
    assert math.isfinite(ll)
    a, b = h.fit(alphas=[0.1, 0.5], betas=[0.05, 0.5])
    assert a in (0.1, 0.5) and b in (0.05, 0.5)
    assert h.log_likelihood(a, b) >= ll - 1e-9


def test_hawkes_keys_and_priors():
    f = Fact(id="f", entity="user", attribute="city", value="Seattle", valid_from=T0)
    assert hawkes_keys(f) == ["user/city", "seattle"]
    ep = Episode(id="e", ts=T0, session_id="s", turn_idx=0, speaker="user", text="x", entities=["user", "Milo"])
    assert hawkes_keys(ep) == ["milo"]
    plain = Episode(id="e2", ts=T0, session_id="s", turn_idx=0, speaker="user", text="x", entities=["user"])
    assert hawkes_keys(plain) == ["user"]
    assert specificity("I moved to Boston on 2023-02-04") > specificity("that was nice")
    assert source_weight("user") > source_weight("assistant") > source_weight("user", "inferred")
