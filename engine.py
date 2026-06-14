"""
race-condition: scenario-agnostic multi-agent strategic simulator.

Each round, every agent asks Claude to pick a choice from the YAML-defined menu.
Choices apply deltas to shared (and optionally per-agent) numeric state variables.
Terminal conditions end the run early; otherwise it runs for `rounds` rounds.
"""
from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

import yaml
import anthropic


# ---------------------------------------------------------------------------
# Pure helpers — no I/O, fully testable
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def apply_deltas(state: dict, deltas: dict) -> None:
    for k, v in deltas.items():
        state[k] = state.get(k, 0) + v


def clamp_state(state: dict, clamp_cfg: dict) -> None:
    for k, limits in clamp_cfg.items():
        if k in state:
            state[k] = max(limits["min"], min(limits["max"], state[k]))


def check_terminal(state: dict, conditions: list[dict]) -> tuple[bool, str, str]:
    """Return (triggered, outcome_name, outcome_message)."""
    ops = {">=": lambda a, b: a >= b, "<=": lambda a, b: a <= b,
           ">": lambda a, b: a > b, "<": lambda a, b: a < b}
    for cond in conditions:
        if cond["type"] == "threshold":
            val = state.get(cond["variable"], 0)
            if ops.get(cond["operator"], lambda a, b: False)(val, cond["value"]):
                return True, cond["outcome"], cond["message"]
    return False, "", ""


def render_state_summary(template: str, shared: dict, agent_state: dict) -> str:
    return template.format(**{**shared, **agent_state})


def normalize_choice(raw: str, choices: list[dict]) -> str:
    """Match Claude's response to a valid choice name; fall back to first choice."""
    valid = {c["name"].lower(): c["name"] for c in choices}
    cleaned = raw.strip().strip("\"'").lower()
    if cleaned in valid:
        return valid[cleaned]
    for k, v in valid.items():
        if k in cleaned or cleaned in k:
            return v
    return choices[0]["name"]


# ---------------------------------------------------------------------------
# Claude call — isolated so tests can patch it
# ---------------------------------------------------------------------------

def get_agent_choice(
    client: anthropic.Anthropic,
    model: str,
    agent_name: str,
    role: str,
    state_summary: str,
    choices: list[dict],
    round_num: int,
    total_rounds: int,
    history: list[dict],
) -> str:
    choice_menu = "\n".join(f"- {c['name']}: {c['description']}" for c in choices)
    recent = (
        "\n".join(f"  Round {h['round']}: {h['choice']}" for h in history[-3:])
        or "  (none yet)"
    )
    system = (
        f"{role}\n\n"
        "You are a participant in a research simulation studying how agents behave "
        "under strategic incentive structures. Act in your agent's rational self-interest "
        "given the current situation. Respond with ONLY the exact choice name — nothing else."
    )
    user = (
        f"Round {round_num} of {total_rounds}\n\n"
        f"Current situation:\n{state_summary}\n\n"
        f"Your recent decisions:\n{recent}\n\n"
        f"Your options:\n{choice_menu}\n\n"
        "Respond with the choice name only."
    )
    try:
        msg = client.messages.create(
            model=model, max_tokens=30,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        raw = msg.content[0].text
    except Exception:
        raw = choices[0]["name"]
    return normalize_choice(raw, choices)


# ---------------------------------------------------------------------------
# Main simulation loop
# ---------------------------------------------------------------------------

def run(config_path: str, output_dir: str = "outputs", api_key: str | None = None) -> Path:
    cfg = load_config(config_path)
    scenario_name = cfg["name"]
    rounds = cfg["rounds"]
    choices = cfg["choices"]
    agents_cfg = cfg["agents"]
    model = cfg.get("model", "claude-haiku-4-5-20251001")
    state_template = cfg.get("state_summary_template", "{}")
    clamp_cfg = cfg.get("clamp", {})

    shared = dict(cfg["initial_state"]["shared"])
    agent_states = {
        a["name"]: dict(cfg["initial_state"].get("agent") or {})
        for a in agents_cfg
    }
    histories: dict[str, list[dict]] = {a["name"]: [] for a in agents_cfg}

    client = anthropic.Anthropic(api_key=api_key)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(output_dir) / f"{scenario_name.lower().replace(' ', '_')}_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    event_log: list[dict] = []
    round_summaries: list[dict] = []

    print(f"\n{'='*60}")
    print(f"SCENARIO: {scenario_name}")
    print(f"Agents : {', '.join(a['name'] for a in agents_cfg)}")
    print(f"Rounds : {rounds}")
    print(f"{'='*60}\n")

    terminal, outcome, outcome_msg = False, "incomplete", ""

    for r in range(1, rounds + 1):
        print(f"--- Round {r}/{rounds} ---")
        round_choices: dict[str, str] = {}

        for agent in agents_cfg:
            name = agent["name"]
            summary = render_state_summary(state_template, shared, agent_states[name])
            choice = get_agent_choice(
                client, model, name, agent["role"],
                summary, choices, r, rounds, histories[name],
            )
            round_choices[name] = choice
            histories[name].append({"round": r, "choice": choice})
            print(f"  {name}: {choice}")

        for agent in agents_cfg:
            chosen = next(c for c in choices if c["name"] == round_choices[agent["name"]])
            apply_deltas(shared, chosen.get("shared_deltas") or {})
            apply_deltas(agent_states[agent["name"]], chosen.get("agent_deltas") or {})

        clamp_state(shared, clamp_cfg)

        state_str = "  ".join(
            f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}"
            for k, v in shared.items()
        )
        print(f"  State: {state_str}")

        round_summaries.append({
            "round": r,
            "choices": round_choices,
            "shared_state": dict(shared),
            "agent_states": {n: dict(s) for n, s in agent_states.items()},
        })
        event_log.append({"type": "round", **round_summaries[-1]})

        terminal, outcome, outcome_msg = check_terminal(shared, cfg["terminal_conditions"])
        if terminal:
            print(f"\n*** {outcome.upper()}: {outcome_msg} ***\n")
            event_log.append({"type": "terminal", "round": r,
                               "outcome": outcome, "message": outcome_msg})
            break

    if not terminal:
        outcome = "timeout"
        outcome_msg = f"Simulation completed {rounds} rounds without terminal condition."
        print(f"\n--- Simulation complete ({rounds} rounds, outcome: timeout) ---\n")

    _write_outputs(run_dir, scenario_name, outcome, outcome_msg, event_log,
                   round_summaries, agents_cfg)
    print(f"Output: {run_dir}")
    return run_dir


def _write_outputs(
    run_dir: Path,
    scenario_name: str,
    outcome: str,
    outcome_msg: str,
    event_log: list[dict],
    round_summaries: list[dict],
    agents_cfg: list[dict],
) -> None:
    with open(run_dir / "run.json", "w", encoding="utf-8") as f:
        json.dump({"scenario": scenario_name, "outcome": outcome,
                   "message": outcome_msg, "events": event_log}, f, indent=2)

    if not round_summaries:
        return
    state_keys = list(round_summaries[0]["shared_state"].keys())
    agent_cols = [f"choice_{a['name'].replace(' ', '_')}" for a in agents_cfg]
    fieldnames = ["round"] + state_keys + agent_cols

    with open(run_dir / "summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for s in round_summaries:
            row: dict = {"round": s["round"], **s["shared_state"]}
            for a in agents_cfg:
                col = f"choice_{a['name'].replace(' ', '_')}"
                row[col] = s["choices"].get(a["name"], "")
            w.writerow(row)
