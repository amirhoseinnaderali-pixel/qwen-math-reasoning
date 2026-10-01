import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", help="JSON result files")
    args = parser.parse_args()

    for path in args.inputs:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        print(
            json.dumps(
                {
                    "mode": data["mode"],
                    "examples": data["examples"],
                    "exact_match_accuracy": data["exact_match_accuracy"],
                    "parse_rate": data["parse_rate"],
                    "strict_hash4_rate": data["strict_hash4_rate"],
                    "mean_generation_seconds": data["mean_generation_seconds"],
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()