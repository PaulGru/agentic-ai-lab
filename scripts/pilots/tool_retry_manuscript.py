"""Draft a V9 retry pilot manuscript offline from the original journal."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from research.pilot_manuscript import draft_pilot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--journal", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--review-id", required=True)
    parser.add_argument("--project", required=True)
    args = parser.parse_args()
    result = draft_pilot(args.corpus, args.journal, args.out_dir, args.review_id, args.project)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
