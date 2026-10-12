from __future__ import annotations

import argparse

from ..src.pipeline import render


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    print(render(args.config))


if __name__ == "__main__":
    main()

