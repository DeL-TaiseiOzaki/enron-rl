#!/usr/bin/env python3
"""
Create a filtered dataset from corbt/enron_emails_sample_questions.

Usage:
    python scripts/create_filtered_dataset.py --inbox vince.kaminski@enron.com --output data/art_e_vince_kaminski
"""

import argparse
from pathlib import Path

from datasets import load_dataset


def create_filtered_dataset(inbox_address: str, output_path: str) -> None:
    """Filter the Enron email QA dataset by inbox address and save locally."""

    print(f"Loading dataset from corbt/enron_emails_sample_questions...")
    train_ds = load_dataset("corbt/enron_emails_sample_questions", split="train")
    test_ds = load_dataset("corbt/enron_emails_sample_questions", split="test")

    print(f"Original train size: {len(train_ds):,}")
    print(f"Original test size: {len(test_ds):,}")

    # Filter by inbox_address
    print(f"Filtering by inbox_address = '{inbox_address}'...")
    train_filtered = train_ds.filter(lambda x: x["inbox_address"] == inbox_address)
    test_filtered = test_ds.filter(lambda x: x["inbox_address"] == inbox_address)

    print(f"Filtered train size: {len(train_filtered):,}")
    print(f"Filtered test size: {len(test_filtered):,}")

    # Save to disk
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    from datasets import DatasetDict
    dataset_dict = DatasetDict({
        "train": train_filtered,
        "test": test_filtered,
    })

    print(f"Saving to {output_path}...")
    dataset_dict.save_to_disk(output_path)
    print("Done!")


def main():
    parser = argparse.ArgumentParser(description="Create filtered Enron email QA dataset")
    parser.add_argument(
        "--inbox",
        type=str,
        default="vince.kaminski@enron.com",
        help="Email address to filter by (default: vince.kaminski@enron.com)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/art_e_vince_kaminski",
        help="Output directory path (default: data/art_e_vince_kaminski)",
    )
    args = parser.parse_args()

    create_filtered_dataset(args.inbox, args.output)


if __name__ == "__main__":
    main()
