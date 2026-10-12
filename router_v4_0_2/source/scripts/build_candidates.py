from __future__ import annotations

import argparse
import json

from ..src.pipeline import build_candidates


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    print(json.dumps(build_candidates(args.config), indent=2))


if __name__ == "__main__":
    main()

