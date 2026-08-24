"""Join line-delimited Reddit submissions and comments using explicit paths."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable


def iter_jsonl(paths: Iterable[str]):
    for raw_path in paths:
        path = Path(raw_path)
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"Malformed JSON in {path}:{line_number}") from error


def join_records(submission_paths: list[str], comment_paths: list[str]) -> tuple[dict[str, dict], int]:
    """Return submissions with nested comments and the orphan-comment count."""

    submissions: dict[str, dict] = {}
    for submission in iter_jsonl(submission_paths):
        submission_id = submission.get("id")
        if not submission_id:
            continue
        normalized = dict(submission)
        normalized["comments"] = []
        submissions.setdefault(str(submission_id), normalized)

    orphan_count = 0
    for comment in iter_jsonl(comment_paths):
        link_id = str(comment.get("link_id") or "").removeprefix("t3_")
        if link_id in submissions:
            submissions[link_id]["comments"].append(comment)
        else:
            orphan_count += 1
    return submissions, orphan_count


def write_jsonl(output_path: str, submissions: dict[str, dict]) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for submission_id in sorted(submissions):
            handle.write(json.dumps(submissions[submission_id], ensure_ascii=False) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Join Reddit submission and comment JSONL files")
    parser.add_argument("--submissions", action="append", required=True, help="Submission JSONL path; repeatable")
    parser.add_argument("--comments", action="append", required=True, help="Comment JSONL path; repeatable")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    submissions, orphan_count = join_records(args.submissions, args.comments)
    write_jsonl(args.output, submissions)
    comment_count = sum(len(item["comments"]) for item in submissions.values())
    print(
        f"Wrote {len(submissions)} submissions and {comment_count} linked comments "
        f"to {args.output}; {orphan_count} comments had no matching submission."
    )


if __name__ == "__main__":
    main()
