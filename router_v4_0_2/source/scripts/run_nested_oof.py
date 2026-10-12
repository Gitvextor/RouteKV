from __future__ import annotations

import argparse
import json

from ..src.pipeline import run_nested_oof


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    print(
        json.dumps(run_nested_oof(args.config, device=args.device), indent=2)
    )


if __name__ == "__main__":
    main()

