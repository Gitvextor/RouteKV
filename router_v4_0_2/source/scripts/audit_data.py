from __future__ import annotations

import argparse
import json

from ..src.pipeline import audit_data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    print(json.dumps(audit_data(args.config), indent=2))


if __name__ == "__main__":
    main()

