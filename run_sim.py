"""CLI entry point: python run_sim.py <scenario.yaml> [--plot] [--output DIR]"""
import argparse
import os
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        description="race-condition: multi-agent strategic simulator"
    )
    parser.add_argument("scenario", help="Path to scenario YAML (e.g. scenarios/capability_race.yaml)")
    parser.add_argument("--output", default="outputs", help="Output directory (default: outputs/)")
    parser.add_argument("--api-key", help="Anthropic API key (or set ANTHROPIC_API_KEY env var)")
    parser.add_argument("--plot", action="store_true", help="Generate matplotlib plot after run")
    args = parser.parse_args()

    api_key = args.api_key or os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: set ANTHROPIC_API_KEY or pass --api-key", file=sys.stderr)
        sys.exit(1)

    from engine import run
    run_dir = run(args.scenario, args.output, api_key)

    if args.plot:
        from analysis import plot_run
        plot_run(str(run_dir))


if __name__ == "__main__":
    main()
