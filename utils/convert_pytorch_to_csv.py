import argparse
import csv
import json
import os

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert lfm_semids.pt to CSV with item_id and semantic_ids columns."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="outputs/ml1m_item_rqvae_semids.pt",
        help="Path to the .pt file containing semantic IDs",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/ml1m_item_rqvae_semids_new.csv",
        help="Path to write the CSV output",
    )
    return parser.parse_args()


def to_list(value):
    if isinstance(value, torch.Tensor):
        return value.tolist()
    return list(value)


def main() -> None:
    args = parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input file not found: {args.input}")

    data = torch.load(args.input, map_location="cpu", weights_only=False)
    semids = data["semantic_ids"] if isinstance(data, dict) else data
    rows = to_list(semids)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    with open(args.output, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["item_id", "semantic_ids"])
        for idx, row in enumerate(rows, start=1):
            writer.writerow([idx, json.dumps(row)])

    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
