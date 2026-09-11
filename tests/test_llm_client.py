from evals.extract_cache import unique_sessions
from evals.longmemeval.data import load_longmemeval
from smem.extract import _render_session
from smem.llm import OpenAICompatLLM
from smem.schemas import Session, Turn
from tests.conftest import T0
from tests.test_eval_harness import write_dataset

SCHEMA = {"type": "object", "properties": {"a": {"type": "string"}}}
THINK_OFF = {"chat_template_kwargs": {"enable_thinking": False}}


def test_vllm_request_uses_response_format_and_keeps_extra_body():
    llm = OpenAICompatLLM("qwen", base_url="http://localhost:8000/v1", extra_body=THINK_OFF)
    kw = llm.request_kwargs("sys", "usr", SCHEMA, 0.0, 100)
    assert kw["response_format"]["json_schema"]["schema"] == SCHEMA
    assert kw["extra_body"] == THINK_OFF
    assert "guided_json" not in kw["extra_body"]


def test_guided_json_mode_and_openai_never_gets_extra_body():
    llm = OpenAICompatLLM("qwen", base_url="http://localhost:8000/v1", extra_body=THINK_OFF, schema_mode="guided_json")
    kw = llm.request_kwargs("sys", "usr", SCHEMA, 0.0, 100)
    assert kw["extra_body"] == {**THINK_OFF, "guided_json": SCHEMA} and "response_format" not in kw
    openai = OpenAICompatLLM("gpt-4.1-mini", base_url="https://api.openai.com/v1", api_key="k", extra_body=THINK_OFF)
    kw = openai.request_kwargs("sys", "usr", SCHEMA, 0.0, 100)
    assert "extra_body" not in kw and kw["response_format"]["type"] == "json_schema"


def test_unconstrained_arm_asks_for_json_in_prompt_only():
    llm = OpenAICompatLLM("qwen", base_url="http://localhost:8000/v1", constrained_decoding=False)
    kw = llm.request_kwargs("sys", "usr", SCHEMA, 0.0, 100)
    assert "response_format" not in kw and "extra_body" not in kw
    assert kw["messages"][0]["content"].endswith("Respond with a single JSON object and nothing else.")


def test_long_turns_are_cut_in_the_middle():
    long = " ".join(f"w{i}" for i in range(3000))
    s = Session(session_id="s", ts=T0, turns=[Turn(role="user", content="I moved to Boston."), Turn(role="assistant", content=long)])
    rendered = _render_session(s, max_turn_tokens=200)
    assert "I moved to Boston." in rendered and "[...]" in rendered
    assert rendered.count("w0 ") == 1 and "w2999" in rendered and len(rendered) < len(long) / 4
    assert _render_session(s, None).count("[...]") == 0


def test_unique_sessions_dedups_by_content(tmp_path):
    data = tmp_path / "mini.json"
    write_dataset(data)
    qs = load_longmemeval(data)
    total = sum(len(q.sessions) for q in qs)
    uniq = unique_sessions(qs)
    assert len(uniq) < total  # filler sessions repeat across questions
    assert len({s.content_hash() for s in uniq}) == len(uniq)
