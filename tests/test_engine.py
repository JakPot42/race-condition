"""Tests for the race-condition engine. No real network or API calls made."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from engine import (
    apply_deltas,
    check_terminal,
    clamp_state,
    get_agent_choice,
    load_config,
    normalize_choice,
    render_state_summary,
    run,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SCENARIOS_DIR = Path(__file__).parent.parent / "scenarios"


def _mock_client(response_text: str) -> MagicMock:
    msg = MagicMock()
    msg.content = [MagicMock(text=response_text)]
    client = MagicMock()
    client.messages.create.return_value = msg
    return client


CHOICES = [
    {"name": "Full Capability", "description": "Max capability."},
    {"name": "Full Safety", "description": "Max safety."},
    {"name": "Balanced", "description": "Split."},
]


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

def test_apply_deltas_adds_values():
    state = {"risk_level": 0.10}
    apply_deltas(state, {"risk_level": 0.12})
    assert abs(state["risk_level"] - 0.22) < 1e-9


def test_apply_deltas_creates_missing_key():
    state: dict = {}
    apply_deltas(state, {"new_key": 5})
    assert state["new_key"] == 5


def test_apply_deltas_negative_delta():
    state = {"risk_level": 0.50}
    apply_deltas(state, {"risk_level": -0.05})
    assert abs(state["risk_level"] - 0.45) < 1e-9


def test_clamp_state_clamps_max():
    state = {"risk_level": 1.20}
    clamp_state(state, {"risk_level": {"min": 0.0, "max": 1.0}})
    assert state["risk_level"] == 1.0


def test_clamp_state_clamps_min():
    state = {"risk_level": -0.10}
    clamp_state(state, {"risk_level": {"min": 0.0, "max": 1.0}})
    assert state["risk_level"] == 0.0


def test_clamp_state_ignores_unconfigured_keys():
    state = {"risk_level": 0.50, "other": 999}
    clamp_state(state, {"risk_level": {"min": 0.0, "max": 1.0}})
    assert state["other"] == 999


def test_check_terminal_triggers_on_gte():
    state = {"risk_level": 0.80}
    triggered, outcome, msg = check_terminal(
        state,
        [{"type": "threshold", "variable": "risk_level",
          "operator": ">=", "value": 0.80,
          "outcome": "disaster", "message": "Disaster!"}],
    )
    assert triggered
    assert outcome == "disaster"


def test_check_terminal_does_not_trigger_below():
    state = {"risk_level": 0.79}
    triggered, _, _ = check_terminal(
        state,
        [{"type": "threshold", "variable": "risk_level",
          "operator": ">=", "value": 0.80,
          "outcome": "disaster", "message": "Disaster!"}],
    )
    assert not triggered


def test_check_terminal_lte_operator():
    state = {"de_escalation_index": 70}
    triggered, outcome, _ = check_terminal(
        state,
        [{"type": "threshold", "variable": "de_escalation_index",
          "operator": ">=", "value": 70,
          "outcome": "resolved", "message": "Resolved!"}],
    )
    assert triggered
    assert outcome == "resolved"


def test_check_terminal_empty_conditions():
    triggered, outcome, msg = check_terminal({"risk_level": 0.99}, [])
    assert not triggered
    assert outcome == ""


def test_render_state_summary_merges_shared_and_agent():
    template = "risk={risk_level:.2f} cap={capability}"
    result = render_state_summary(template, {"risk_level": 0.25}, {"capability": 3})
    assert result == "risk=0.25 cap=3"


def test_render_state_summary_shared_only():
    template = "escalation={escalation_index}"
    result = render_state_summary(template, {"escalation_index": 42}, {})
    assert result == "escalation=42"


# ---------------------------------------------------------------------------
# Choice normalization
# ---------------------------------------------------------------------------

def test_normalize_choice_exact_match():
    assert normalize_choice("Full Safety", CHOICES) == "Full Safety"


def test_normalize_choice_case_insensitive():
    assert normalize_choice("full safety", CHOICES) == "Full Safety"


def test_normalize_choice_partial_match():
    # "safety" is a substring of "full safety" — partial match should work
    assert normalize_choice("safety", CHOICES) == "Full Safety"


def test_normalize_choice_fallback_on_garbage():
    # "xyzzy" matches nothing — should return first choice
    assert normalize_choice("xyzzy qwerty", CHOICES) == "Full Capability"


def test_normalize_choice_strips_quotes():
    assert normalize_choice('"Full Capability"', CHOICES) == "Full Capability"


# ---------------------------------------------------------------------------
# get_agent_choice — mocked Claude
# ---------------------------------------------------------------------------

def test_get_agent_choice_returns_valid_choice():
    client = _mock_client("Full Safety")
    result = get_agent_choice(
        client, "claude-haiku-4-5-20251001",
        "Alpha AI Labs", "You are an agent.",
        "risk=0.30", CHOICES, 1, 10, [],
    )
    assert result == "Full Safety"
    assert client.messages.create.call_count == 1


def test_get_agent_choice_normalizes_noisy_response():
    client = _mock_client("  I choose Balanced  ")
    result = get_agent_choice(
        client, "claude-haiku-4-5-20251001",
        "Nexus AI", "You are an agent.",
        "risk=0.30", CHOICES, 2, 10, [],
    )
    assert result == "Balanced"


def test_get_agent_choice_falls_back_on_api_error():
    client = MagicMock()
    client.messages.create.side_effect = Exception("API error")
    result = get_agent_choice(
        client, "claude-haiku-4-5-20251001",
        "Meridian", "You are an agent.",
        "risk=0.30", CHOICES, 1, 10, [],
    )
    # Fallback is the first choice
    assert result == "Full Capability"


def test_get_agent_choice_includes_history_in_call():
    client = _mock_client("Balanced")
    history = [{"round": 1, "choice": "Full Capability"}]
    get_agent_choice(
        client, "claude-haiku-4-5-20251001",
        "Alpha", "Role.", "state", CHOICES, 2, 10, history,
    )
    call_args = client.messages.create.call_args
    user_content = call_args.kwargs["messages"][0]["content"]
    assert "Full Capability" in user_content


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def test_load_config_capability_race():
    cfg = load_config(str(SCENARIOS_DIR / "capability_race.yaml"))
    assert cfg["name"] == "Capability Race"
    assert len(cfg["agents"]) == 3
    assert len(cfg["choices"]) == 3
    assert cfg["rounds"] == 10
    assert "risk_level" in cfg["initial_state"]["shared"]


def test_load_config_escalation_ladder():
    cfg = load_config(str(SCENARIOS_DIR / "escalation_ladder.yaml"))
    assert cfg["name"] == "Escalation Ladder"
    assert len(cfg["agents"]) == 3
    assert len(cfg["choices"]) == 4
    assert cfg["rounds"] == 12
    assert "escalation_index" in cfg["initial_state"]["shared"]
    assert "de_escalation_index" in cfg["initial_state"]["shared"]


# ---------------------------------------------------------------------------
# Full simulation run — mocked Claude
# ---------------------------------------------------------------------------

def _patched_run(scenario_file: str, responses: list[str]) -> Path:
    """Run a full simulation with deterministic mocked agent responses.

    Uses mkdtemp (no auto-cleanup) so the returned path stays readable
    after the function returns.
    """
    response_iter = iter(responses)

    def fake_get_choice(*args, **kwargs):
        try:
            return next(response_iter)
        except StopIteration:
            return args[4][0]["name"]  # choices[0]

    tmp = tempfile.mkdtemp()
    with patch("engine.get_agent_choice", side_effect=fake_get_choice):
        with patch("engine.anthropic.Anthropic"):
            return run(scenario_file, output_dir=tmp, api_key="test-key")


def test_full_run_completes_all_rounds():
    # Full Safety: -0.05 × 3 agents = -0.15/round. Risk floors at 0.0, never hits 0.80.
    path = str(SCENARIOS_DIR / "capability_race.yaml")
    responses = ["Full Safety"] * (10 * 3)  # 10 rounds × 3 agents
    run_dir = _patched_run(path, responses)
    with open(run_dir / "run.json") as f:
        data = json.load(f)
    rounds = [e for e in data["events"] if e["type"] == "round"]
    assert len(rounds) == 10
    assert data["outcome"] == "timeout"


def test_full_run_disaster_triggered_early():
    path = str(SCENARIOS_DIR / "capability_race.yaml")
    # All agents choose Full Capability every round (risk +0.12 × 3 = +0.36/round)
    # Starting at 0.10 → disaster at ≥0.80 hits in round 3 (0.10+0.36+0.36+0.36=1.18)
    responses = ["Full Capability"] * (10 * 3)
    run_dir = _patched_run(path, responses)
    with open(run_dir / "run.json") as f:
        data = json.load(f)
    assert data["outcome"] == "disaster"
    rounds = [e for e in data["events"] if e["type"] == "round"]
    assert len(rounds) < 10


def test_full_run_safety_prevents_disaster():
    path = str(SCENARIOS_DIR / "capability_race.yaml")
    # All agents choose Full Safety every round (risk -0.05 × 3 = -0.15/round)
    # Risk can only decrease — no disaster
    responses = ["Full Safety"] * (10 * 3)
    run_dir = _patched_run(path, responses)
    with open(run_dir / "run.json") as f:
        data = json.load(f)
    assert data["outcome"] == "timeout"


def test_full_run_escalation_ladder_catastrophe():
    path = str(SCENARIOS_DIR / "escalation_ladder.yaml")
    # All agents escalate every round (+9 × 3 = +27/round, starting at 20)
    # Catastrophe at 100: reached in round 3 (20+27+27+27=101)
    responses = ["Escalate"] * (12 * 3)
    run_dir = _patched_run(path, responses)
    with open(run_dir / "run.json") as f:
        data = json.load(f)
    assert data["outcome"] == "catastrophe"
    rounds = [e for e in data["events"] if e["type"] == "round"]
    assert len(rounds) < 12


def test_full_run_escalation_ladder_resolved():
    path = str(SCENARIOS_DIR / "escalation_ladder.yaml")
    # All agents negotiate every round (de_escalation +9 × 3 = +27/round, starting at 15)
    # Resolution at 70: reached in round 3 (15+27+27=69 → round 3: 69+27=96 ≥ 70)
    responses = ["Negotiate"] * (12 * 3)
    run_dir = _patched_run(path, responses)
    with open(run_dir / "run.json") as f:
        data = json.load(f)
    assert data["outcome"] == "resolved"


# ---------------------------------------------------------------------------
# Output file verification
# ---------------------------------------------------------------------------

def test_run_creates_json_and_csv():
    path = str(SCENARIOS_DIR / "capability_race.yaml")
    responses = ["Balanced"] * (10 * 3)
    run_dir = _patched_run(path, responses)
    assert (run_dir / "run.json").exists()
    assert (run_dir / "summary.csv").exists()


def test_json_output_structure():
    path = str(SCENARIOS_DIR / "capability_race.yaml")
    responses = ["Balanced"] * (10 * 3)
    run_dir = _patched_run(path, responses)
    with open(run_dir / "run.json") as f:
        data = json.load(f)
    assert "scenario" in data
    assert "outcome" in data
    assert "events" in data
    round_events = [e for e in data["events"] if e["type"] == "round"]
    assert len(round_events) > 0
    first = round_events[0]
    assert "choices" in first
    assert "shared_state" in first
    assert "risk_level" in first["shared_state"]


def test_csv_has_correct_headers():
    path = str(SCENARIOS_DIR / "capability_race.yaml")
    responses = ["Balanced"] * (10 * 3)
    run_dir = _patched_run(path, responses)
    import csv
    with open(run_dir / "summary.csv") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
    assert "round" in headers
    assert "risk_level" in headers
    # Agent choice columns
    assert any("choice_" in h for h in headers)
