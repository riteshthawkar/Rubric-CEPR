from __future__ import annotations

import argparse

from qwen_edit_project.self_evolve.loop import SelfEvolveRunner
from qwen_edit_project.utils.config import load_yaml_config, merge_override, parse_override


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the self-evolving editing loop.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--set", action="append", default=[], help="Override config using dotted.key=value")
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    for raw in args.set:
        key, value = parse_override(raw)
        config = merge_override(config, key, value)

    summary = SelfEvolveRunner(config=config, dry_run=args.dry_run, limit=args.limit).run()
    print(f"Self-evolve run complete. Summary written to {summary['output_root']}")


if __name__ == "__main__":
    main()
