from datetime import datetime, timedelta

import pytest

from smem.config import SystemConfig
from smem.schemas import Session, Turn

T0 = datetime(2023, 2, 4, 10, 0)


def session(sid: str, days: int, *contents: str, assistant: str | None = None) -> Session:
    turns = [Turn(role="user", content=c) for c in contents]
    if assistant:
        turns.append(Turn(role="assistant", content=assistant))
    return Session(session_id=sid, ts=T0 + timedelta(days=days), turns=turns)


@pytest.fixture
def worked_example() -> list[Session]:
    """The plan's section 01 example: Boston in February, Seattle in May, filler in between."""
    return [
        session("s3", 0, "I just moved to Boston, renting near Kendall.", "My favorite cuisine is Thai.",
                assistant="Congrats on the move to Boston, Kendall Square is lovely."),
        session("s5", 10, "I love garlic in almost everything I cook.", "Can you give me a pasta recipe?",
                assistant="Sure, start by boiling water and then add salt before the pasta."),
        session("s9", 30, "I am debugging a segfault in my C++ project at work today.",
                assistant="Run it under valgrind to find the invalid access first."),
        session("s12", 45, "My cat Milo is turning three next week and I want to bake him a treat."),
        session("s21", 104, "I moved to Seattle this week because of a company transfer."),
    ]


@pytest.fixture
def offline_cfg() -> SystemConfig:
    return SystemConfig.load("configs/offline.yaml", {"budget.read_tokens": 200})
