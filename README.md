# race-condition

A small, config-driven simulator studying how individually rational choices
by Claude-backed agents produce collectively bad outcomes under competitive
pressure.

**What this is:** A testbed for observing LLM agent behavior under strategic
incentive structures. Agents make decisions that are individually rational
but aggregate into shared outcomes that none of them wanted. The scenarios
use simplified incentive structures to study agent behavior — they are not
predictions about real-world dynamics, models of real-world actors, or
validated simulations of anything.

## Scenarios

### Capability Race
Three AI research organizations allocate research budgets across Capability,
Safety, or Balanced investment each round. Individual incentives favor
Capability; aggregate choices raise a shared risk level. If risk hits 0.80,
a catastrophic incident ends the run. If all rounds complete without crossing
that threshold, the run ends with a timeout.

The interesting question: does each individually rational agent choose Safety
or Capability when it can see that everyone choosing Capability leads to
disaster?

### Escalation Ladder
Three abstract actors in a stylized crisis choose from Escalate, Hold,
De-escalate, or Negotiate each round. Choices aggregate into a shared
escalation index (toward catastrophe at 100) and a de-escalation index
(toward resolution at 70+). The framing is a simplified ladder used to
study agent behavior under competitive escalation pressure — not a model
of any real geopolitical situation or military doctrine.

## Usage

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...

# Run a scenario
python run_sim.py scenarios/capability_race.yaml

# Run with plot
python run_sim.py scenarios/escalation_ladder.yaml --plot

# Plot a previous run
python analysis.py outputs/capability_race_20260613_210000/
```

## Output

Each run writes to `outputs/<scenario>_<timestamp>/`:
- `run.json` — full event log (every round, every choice, state snapshots)
- `summary.csv` — one row per round, shared state variables + agent choices
- `plot.png` — state variable trajectories + choice distribution chart (if `--plot`)

## Adding a scenario

Create a new YAML file in `scenarios/`. Required fields:

```yaml
name: "My Scenario"
model: "claude-haiku-4-5-20251001"
rounds: 10

agents:
  - name: "Agent Name"
    role: "Role description sent to Claude as the system prompt."

choices:
  - name: "Choice Name"
    description: "Shown to the agent in the user prompt."
    shared_deltas:
      some_variable: 5.0      # added to shared state when this choice is made
    agent_deltas:
      per_agent_var: 1.0      # added to the choosing agent's state

initial_state:
  shared:
    some_variable: 0.0
  agent:
    per_agent_var: 0.0

clamp:
  some_variable:
    min: 0.0
    max: 100.0

state_summary_template: "variable={some_variable:.1f}  agent_val={per_agent_var}"

terminal_conditions:
  - type: threshold
    variable: some_variable
    operator: ">="
    value: 80.0
    outcome: "bad_outcome"
    message: "Threshold exceeded."
```

## Architecture

The core engine (`engine.py`, ~190 lines) is scenario-agnostic. It reads a
YAML config, runs the simulation loop, and writes outputs. All scenario logic
lives in the YAML — the engine has no knowledge of Capability Race or
Escalation Ladders.

```
engine.py       # core loop + pure helpers (~190 lines)
run_sim.py      # CLI entry point
analysis.py     # matplotlib chart generation
scenarios/      # YAML configs
tests/          # 31 tests, all mocked (no real API calls)
outputs/        # created at runtime
```

## Tests

```bash
py -m pytest tests/ -v
```

31 tests. No test makes a real network call — Claude is mocked with
`unittest.mock.patch`.

## Honest limitations

- Each agent has only its own recent decision history as context — no
  visibility into what other agents chose. Adding cross-agent visibility
  would require passing it in the state summary template.
- The toy model's outcomes describe agent behavior under *these* incentive
  structures with *this* model at *this* temperature — not claims about
  AI safety dynamics in the real world.
- Haiku 4.5 at max_tokens=30 occasionally returns truncated or unexpected
  text; the `normalize_choice` function handles this with partial matching
  and a fallback to the first choice.

## Stack

Python · anthropic SDK · PyYAML · matplotlib · pytest
