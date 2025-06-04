import argparse
import sys
from core.orchestrator import TradingOrchestrator

def main():
    parser = argparse.ArgumentParser(
        description="Command-line interface for OnlyFunds Orchestrator"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command")

    # Run orchestrator
    run_parser = subparsers.add_parser("run", help="Run orchestration cycle once")
    run_parser.add_argument("--profile", choices=["conservative", "normal", "aggressive"], default=None, help="Risk profile to use")
    run_parser.add_argument("--dry-run", action="store_true", help="Force dry run mode")
    run_parser.add_argument("--live", action="store_true", help="Force live mode")
    run_parser.add_argument("--no-retrain", action="store_true", help="Skip ML retraining")
    run_parser.add_argument("--schedule", action="store_true", help="Loop forever, orchestrating every hour")

    # Show status (basic: last log lines)
    status_parser = subparsers.add_parser("status", help="Show orchestrator status (last 20 log lines)")

    # Show last orchestrator metrics
    metrics_parser = subparsers.add_parser("metrics", help="Show metrics of last orchestration run")

    args = parser.parse_args()
    orch = TradingOrchestrator()

    if args.command == "run":
        if args.profile:
            profile = args.profile
        else:
            profile = None
        # Force mode in config if requested
        if args.dry_run:
            orch.config["mode"] = "dry_run"
        elif args.live:
            orch.config["mode"] = "live"
        retrain = not args.no_retrain
        schedule = args.schedule

        print(f"Running orchestrator (profile={profile}, mode={orch.config.get('mode','dry_run')}, retrain={retrain}, schedule={schedule})...")
        orch.orchestrate(profile=profile, retrain=retrain, schedule=schedule)

    elif args.command == "status":
        print("Last 20 lines from orchestrator.log:")
        try:
            with open("orchestrator.log", "r") as f:
                lines = f.readlines()
                print("".join(lines[-20:]))
        except FileNotFoundError:
            print("No orchestrator.log file found.")
    elif args.command == "metrics":
        from pprint import pprint
        print("Last run orchestrator metrics (if any):")
        pprint(orch.metrics)
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()