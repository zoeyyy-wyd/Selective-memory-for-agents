from datetime import datetime

from smem.config import SystemConfig, parse_override
from smem.schemas import Fact, Session, Turn


def test_config_hash_is_stable_and_sensitive_to_overrides():
    a = SystemConfig()
    b = SystemConfig()
    assert a.config_hash() == b.config_hash()
    c = a.with_overrides({"write.policy": "all", "budget.read_tokens": 4000})
    assert c.write.policy == "all" and c.budget.read_tokens == 4000
    assert c.config_hash() != a.config_hash()


def test_parse_override_yaml_values():
    assert parse_override("write.validity_chain=false") == ("write.validity_chain", False)
    assert parse_override("budget.store_tokens=2500") == ("budget.store_tokens", 2500)
    assert parse_override("evict.policy=lru") == ("evict.policy", "lru")


def test_session_content_hash_ignores_id_and_date():
    a = Session(session_id="a", ts=datetime(2023, 1, 1), turns=[Turn(role="user", content="hi")])
    b = Session(session_id="b", ts=datetime(2024, 1, 1), turns=[Turn(role="user", content="hi")])
    assert a.content_hash() == b.content_hash()


def test_fact_validity():
    f = Fact(id="f_1", entity="user", attribute="city", value="Boston", valid_from=datetime(2023, 2, 4),
             valid_to=datetime(2023, 5, 19))
    assert f.is_valid_at(datetime(2023, 3, 1))
    assert not f.is_valid_at(datetime(2023, 5, 19))
    assert not f.is_valid_at(datetime(2023, 1, 1))


def test_infra_knobs_do_not_change_the_run_identity():
    """A timeout edit or a device move mid-campaign must not orphan a half-finished run."""
    from smem.config import SystemConfig

    base = SystemConfig.load("configs/tokenrouter.yaml")
    same = base.with_overrides({"models.embed_device": "cpu", "models.nli_device": "cpu",
                                "models.request_timeout": 5.0, "models.retry_attempts": 1, "extract.cache_dir": "/x"})
    assert same.config_hash() == base.config_hash()
    diff = base.with_overrides({"evict.policy": "fifo"})
    assert diff.config_hash() != base.config_hash()
    assert SystemConfig.is_infra_key("models.embed_device") and not SystemConfig.is_infra_key("evict.policy")
