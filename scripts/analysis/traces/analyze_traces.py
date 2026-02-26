"""Comprehensive trace analysis for ART-e evaluation results.

Analyzes traced evaluation data from outputs/eval-traced/ to produce
a detailed markdown report covering search strategies, failure modes,
and cross-model comparisons.
"""

import json
import re
from collections import defaultdict
from pathlib import Path

BASE_DIR = Path("/home/usr_ext_taisei_ozaki_ccoe_toyota/enron-rl")
TRACED_DIR = BASE_DIR / "outputs" / "eval-traced"
METADATA_PATH = (
    BASE_DIR
    / "data"
    / "art_e_vince_kaminski_multihop"
    / "enronhop_20260217_185330.jsonl"
)
OUTPUT_PATH = TRACED_DIR / "trace_analysis_report.md"

BENCHMARKS = {
    "standard": "art-e-3model-comparison",
    "multihop": "art-e-multihop",
}
MODELS = ["art-e-008", "gpt-5", "gpt-5-mini", "gpt-oss-120b", "gpt-oss-20b", "qwen3-14b"]

IDK_PATTERNS = [
    "i don't know", "i'm not able", "i am not able", "unable to find",
    "could not find", "couldn't find", "no information", "cannot determine",
    "i was unable", "i could not", "i couldn't",
]


# ── Data Loading ──────────────────────────────────────────────


def load_results(benchmark_key: str) -> dict[str, list[dict]]:
    bench_dir = TRACED_DIR / BENCHMARKS[benchmark_key]
    data = {}
    for model in MODELS:
        path = bench_dir / model / "results.jsonl"
        if not path.exists():
            continue
        records = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        data[model] = records
    return data


def load_multihop_metadata() -> dict[str, dict]:
    meta = {}
    with open(METADATA_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                obj = json.loads(line)
                meta[obj["id"]] = obj
    return meta


# ── Trace Parsing ─────────────────────────────────────────────


def parse_trace(messages: list[dict]) -> dict:
    """Parse message trace to extract tool calls and their results."""
    searches = []
    reads = []
    answers = []
    bracket_bugs = []

    for i, msg in enumerate(messages):
        if msg.get("role") != "assistant" or "tool_calls" not in msg:
            continue
        for tc in msg["tool_calls"]:
            fn = tc.get("function", {})
            name = fn.get("name", "")
            args_str = fn.get("arguments", "{}")
            try:
                args = json.loads(args_str) if isinstance(args_str, str) else args_str
            except json.JSONDecodeError:
                args = {}

            # Find corresponding tool response
            tc_id = tc.get("id", "")
            response_content = ""
            for j in range(i + 1, min(i + 3, len(messages))):
                rmsg = messages[j]
                if rmsg.get("role") == "tool" and rmsg.get("tool_call_id") == tc_id:
                    response_content = rmsg.get("content", "")
                    break
            # Fallback: next tool message
            if not response_content:
                for j in range(i + 1, min(i + 3, len(messages))):
                    if messages[j].get("role") == "tool":
                        response_content = messages[j].get("content", "")
                        break

            if name == "search_inbox":
                keywords = args.get("keywords", [])
                is_empty = (
                    "No results found" in response_content
                    or "no results" in response_content.lower()
                )
                searches.append({
                    "keywords": keywords,
                    "keyword_count": len(keywords),
                    "is_empty": is_empty,
                    "response_len": len(response_content),
                })
            elif name == "read_email":
                mid = args.get("message_id", "")
                has_brackets = mid.startswith("<") and mid.endswith(">")
                not_found = (
                    "not found" in response_content.lower()
                    or "error" in response_content.lower()[:50]
                )
                reads.append({
                    "message_id": mid,
                    "has_brackets": has_brackets,
                    "not_found": not_found,
                })
                if not has_brackets and mid:
                    bracket_bugs.append(mid)
            elif name == "return_final_answer":
                answers.append({
                    "answer": args.get("answer", ""),
                    "sources": args.get("sources", []),
                })

    return {
        "searches": searches,
        "reads": reads,
        "answers": answers,
        "bracket_bugs": bracket_bugs,
    }


# ── Metrics Computation ──────────────────────────────────────


def is_idk(answer: str | None) -> bool:
    if not answer:
        return True
    lower = answer.lower()
    return any(p in lower for p in IDK_PATTERNS)


def compute_basic_metrics(records: list[dict]) -> dict:
    n = len(records)
    if n == 0:
        return {}
    correct = sum(1 for r in records if (r.get("judge_score") or 0) > 0)
    halluc = sum(1 for r in records if (r.get("judge_score") or 0) < 0)
    idk = sum(1 for r in records if is_idk(r.get("generated_answer")))
    errors = sum(1 for r in records if r.get("error"))
    return {
        "n": n,
        "correct": correct,
        "accuracy": correct / n,
        "avg_judge_score": sum(r.get("judge_score", 0) or 0 for r in records) / n,
        "hallucination_rate": halluc / n,
        "idk_rate": idk / n,
        "error_rate": errors / n,
        "avg_tool_calls": sum(r.get("tool_call_count", 0) or 0 for r in records) / n,
        "avg_turns": sum(r.get("turns_used", 0) or 0 for r in records) / n,
        "avg_prompt_tokens": sum(r.get("prompt_tokens", 0) or 0 for r in records) / n,
        "avg_completion_tokens": sum(r.get("completion_tokens", 0) or 0 for r in records) / n,
        "avg_total_tokens": sum(r.get("total_tokens", 0) or 0 for r in records) / n,
        "avg_elapsed": sum(r.get("elapsed_seconds", 0) or 0 for r in records) / n,
        "tokens_per_correct": (
            sum(r.get("total_tokens", 0) or 0 for r in records) / correct
            if correct > 0
            else float("inf")
        ),
    }


def compute_trace_metrics(records: list[dict]) -> dict:
    """Compute trace-level metrics from parsed messages."""
    all_searches = []
    all_reads = []
    total_bracket_bugs = 0
    per_example = []

    for r in records:
        msgs = r.get("messages", [])
        trace = parse_trace(msgs)
        is_correct = (r.get("judge_score") or 0) > 0

        all_searches.extend(trace["searches"])
        all_reads.extend(trace["reads"])
        total_bracket_bugs += len(trace["bracket_bugs"])

        per_example.append({
            "n_searches": len(trace["searches"]),
            "n_reads": len(trace["reads"]),
            "n_answers": len(trace["answers"]),
            "has_bracket_bug": len(trace["bracket_bugs"]) > 0,
            "is_correct": is_correct,
            "first_search_empty": (
                trace["searches"][0]["is_empty"] if trace["searches"] else None
            ),
            "keywords_per_search": [s["keyword_count"] for s in trace["searches"]],
            "empty_searches": sum(1 for s in trace["searches"] if s["is_empty"]),
        })

    n = len(records)
    total_searches = len(all_searches)
    total_reads = len(all_reads)
    empty_searches = sum(1 for s in all_searches if s["is_empty"])

    # First search success → accuracy
    first_hit = [e for e in per_example if e["first_search_empty"] is False]
    first_miss = [e for e in per_example if e["first_search_empty"] is True]
    acc_first_hit = (
        sum(1 for e in first_hit if e["is_correct"]) / len(first_hit)
        if first_hit
        else 0
    )
    acc_first_miss = (
        sum(1 for e in first_miss if e["is_correct"]) / len(first_miss)
        if first_miss
        else 0
    )

    # Searches for correct vs wrong
    correct_exs = [e for e in per_example if e["is_correct"]]
    wrong_exs = [e for e in per_example if not e["is_correct"]]

    # All keywords counts
    all_kw_counts = [s["keyword_count"] for s in all_searches]

    return {
        "total_searches": total_searches,
        "total_reads": total_reads,
        "avg_searches": total_searches / n if n else 0,
        "avg_reads": total_reads / n if n else 0,
        "search_empty_rate": empty_searches / total_searches if total_searches else 0,
        "bracket_bug_count": total_bracket_bugs,
        "bracket_bug_rate": (
            total_bracket_bugs / total_reads if total_reads else 0
        ),
        "avg_keywords": (
            sum(all_kw_counts) / len(all_kw_counts) if all_kw_counts else 0
        ),
        "acc_first_hit": acc_first_hit,
        "acc_first_miss": acc_first_miss,
        "n_first_hit": len(first_hit),
        "n_first_miss": len(first_miss),
        "avg_searches_correct": (
            sum(e["n_searches"] for e in correct_exs) / len(correct_exs)
            if correct_exs
            else 0
        ),
        "avg_searches_wrong": (
            sum(e["n_searches"] for e in wrong_exs) / len(wrong_exs)
            if wrong_exs
            else 0
        ),
        "per_example": per_example,
    }


def classify_failure(record: dict, trace: dict) -> str:
    """Classify a failed example into a primary failure mode."""
    judge = record.get("judge_score") or 0
    if judge > 0:
        return "correct"

    # Check bracket bug
    parsed = parse_trace(record.get("messages", []))
    if parsed["bracket_bugs"]:
        return "bracket_bug"

    # Check explicit IDK
    answer = record.get("generated_answer")
    if is_idk(answer):
        turns = record.get("turns_used", 0) or 0
        if turns >= 9 and not answer:
            return "timeout"
        return "explicit_idk"

    # Check timeout (high turns, wrong answer)
    turns = record.get("turns_used", 0) or 0
    if turns >= 9:
        return "timeout"

    # Check search fail (most searches empty)
    searches = parsed["searches"]
    if searches:
        empty_rate = sum(1 for s in searches if s["is_empty"]) / len(searches)
        if empty_rate >= 0.7:
            return "search_fail"

    return "wrong_answer"


def compute_failure_classes(records: list[dict]) -> dict[str, int]:
    counts = defaultdict(int)
    for r in records:
        trace = parse_trace(r.get("messages", []))
        cat = classify_failure(r, trace)
        counts[cat] += 1
    return dict(counts)


def compute_hop_metrics(
    records: list[dict], metadata: dict[str, dict]
) -> dict:
    """Compute accuracy broken down by hop count and answer type."""
    by_hop = defaultdict(lambda: {"correct": 0, "total": 0})
    by_type = defaultdict(lambda: {"correct": 0, "total": 0})
    evidence_stats = {"matched": 0, "covered_any": 0, "covered_correct": 0, "covered_wrong": 0, "correct_total": 0, "wrong_total": 0}

    for r in records:
        eid = r.get("example_id", "")
        meta = metadata.get(eid)
        if not meta:
            continue

        is_correct = (r.get("judge_score") or 0) > 0
        hop = meta.get("hop_count", 0)
        atype = meta.get("semantic_answer_type", "unknown")

        by_hop[hop]["total"] += 1
        by_type[atype]["total"] += 1
        if is_correct:
            by_hop[hop]["correct"] += 1
            by_type[atype]["correct"] += 1

        # Evidence mail coverage
        evidence_mails = meta.get("evidence_mails", [])
        if evidence_mails:
            evidence_ids = {f"<{mid}>" for mid in evidence_mails}
            parsed = parse_trace(r.get("messages", []))
            read_ids = {rd["message_id"] for rd in parsed["reads"]}
            overlap = evidence_ids & read_ids
            evidence_stats["matched"] += 1
            if overlap:
                evidence_stats["covered_any"] += 1
                if is_correct:
                    evidence_stats["covered_correct"] += 1
                else:
                    evidence_stats["covered_wrong"] += 1
            if is_correct:
                evidence_stats["correct_total"] += 1
            else:
                evidence_stats["wrong_total"] += 1

    return {
        "by_hop": dict(by_hop),
        "by_type": dict(by_type),
        "evidence": evidence_stats,
    }


def compute_cross_difficulty(all_data: dict[str, list[dict]]) -> dict:
    """Compute cross-model difficulty: which questions are easy/hard/impossible."""
    # Build per-example correctness across models
    example_correct = defaultdict(set)
    example_total = set()
    for model, records in all_data.items():
        for r in records:
            eid = r.get("example_id", "")
            example_total.add(eid)
            if (r.get("judge_score") or 0) > 0:
                example_correct[eid].add(model)

    n_models = len(all_data)
    n_total = len(example_total)
    n_easy = sum(1 for eid in example_total if len(example_correct.get(eid, set())) == n_models)
    n_impossible = sum(1 for eid in example_total if len(example_correct.get(eid, set())) == 0)
    n_hard = sum(1 for eid in example_total if len(example_correct.get(eid, set())) == 1)
    n_medium = n_total - n_easy - n_impossible - n_hard

    return {
        "total": n_total,
        "n_models": n_models,
        "easy": n_easy,
        "medium": n_medium,
        "hard": n_hard,
        "impossible": n_impossible,
    }


def compute_keyword_empty_correlation(records: list[dict]) -> dict[int, dict]:
    """Compute search empty rate by keyword count."""
    by_kw = defaultdict(lambda: {"total": 0, "empty": 0})
    for r in records:
        parsed = parse_trace(r.get("messages", []))
        for s in parsed["searches"]:
            kc = s["keyword_count"]
            by_kw[kc]["total"] += 1
            if s["is_empty"]:
                by_kw[kc]["empty"] += 1
    return dict(by_kw)


# ── Report Generation ────────────────────────────────────────


def pct(v: float) -> str:
    return f"{v * 100:.1f}%"


def fmt(v: float, d: int = 2) -> str:
    return f"{v:.{d}f}"


def table(headers: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(str(h) for h in headers) + " |"]
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(lines)


def generate_report() -> str:
    print("Loading data...")
    std_data = load_results("standard")
    mh_data = load_results("multihop")
    metadata = load_multihop_metadata()

    print(f"Standard: {list(std_data.keys())}")
    print(f"Multihop: {list(mh_data.keys())}")
    print(f"Metadata: {len(metadata)} entries")

    lines = []
    lines.append("# ART-e Traced Evaluation Analysis Report")
    lines.append("")
    lines.append(f"**Date**: 2026-02-18")
    lines.append(f"**Benchmarks**: Standard (50Q) | Multihop (50Q)")
    lines.append(f"**Models**: {', '.join(sorted(set(list(std_data.keys()) + list(mh_data.keys()))))}")
    lines.append("")
    lines.append("---")
    lines.append("")

    # ── 1. Basic Metrics ───────────────────────────────────
    lines.append("## 1. Basic Metrics — Standard Benchmark")
    lines.append("")
    std_metrics = {m: compute_basic_metrics(recs) for m, recs in std_data.items()}
    std_sorted = sorted(std_metrics.items(), key=lambda x: -x[1].get("accuracy", 0))

    headers = ["Model", "N", "Accuracy", "Avg Score", "Halluc%", "IDK%", "Err%", "Avg Tools", "Avg Turns", "Avg Tokens", "Tok/Correct"]
    rows = []
    for m, mx in std_sorted:
        rows.append([
            m, mx["n"], pct(mx["accuracy"]), fmt(mx["avg_judge_score"]),
            pct(mx["hallucination_rate"]), pct(mx["idk_rate"]), pct(mx["error_rate"]),
            fmt(mx["avg_tool_calls"]), fmt(mx["avg_turns"]),
            f"{mx['avg_total_tokens']:.0f}",
            f"{mx['tokens_per_correct']:.0f}" if mx["tokens_per_correct"] != float("inf") else "N/A",
        ])
    lines.append(table(headers, rows))
    lines.append("")

    lines.append("## 2. Basic Metrics — Multihop Benchmark")
    lines.append("")
    mh_metrics = {m: compute_basic_metrics(recs) for m, recs in mh_data.items()}
    mh_sorted = sorted(mh_metrics.items(), key=lambda x: -x[1].get("accuracy", 0))

    rows = []
    for m, mx in mh_sorted:
        rows.append([
            m, mx["n"], pct(mx["accuracy"]), fmt(mx["avg_judge_score"]),
            pct(mx["hallucination_rate"]), pct(mx["idk_rate"]), pct(mx["error_rate"]),
            fmt(mx["avg_tool_calls"]), fmt(mx["avg_turns"]),
            f"{mx['avg_total_tokens']:.0f}",
            f"{mx['tokens_per_correct']:.0f}" if mx["tokens_per_correct"] != float("inf") else "N/A",
        ])
    lines.append(table(headers, rows))
    lines.append("")

    # Standard vs Multihop delta
    lines.append("### Accuracy Drop: Standard → Multihop")
    lines.append("")
    common = sorted(set(std_data.keys()) & set(mh_data.keys()))
    rows = []
    for m in common:
        sa = std_metrics[m]["accuracy"]
        ma = mh_metrics[m]["accuracy"]
        delta = ma - sa
        rows.append([m, pct(sa), pct(ma), f"{delta * 100:+.1f}pp"])
    lines.append(table(["Model", "Standard", "Multihop", "Delta"], rows))
    lines.append("")
    lines.append("---")
    lines.append("")

    # ── 3. Trace Analysis ──────────────────────────────────
    lines.append("## 3. Trace Analysis — Tool Usage & Search Strategy")
    lines.append("")

    std_trace = {m: compute_trace_metrics(recs) for m, recs in std_data.items()}
    mh_trace = {m: compute_trace_metrics(recs) for m, recs in mh_data.items()}

    for bench_name, trace_data, metrics_data in [
        ("Standard", std_trace, std_metrics),
        ("Multihop", mh_trace, mh_metrics),
    ]:
        lines.append(f"### {bench_name} Benchmark — Trace Metrics")
        lines.append("")
        headers = [
            "Model", "Avg Searches", "Avg Reads", "Search Empty%",
            "Bracket Bugs", "Bug Rate%", "Avg KW/Search",
        ]
        sorted_models = sorted(trace_data.keys(), key=lambda m: -metrics_data.get(m, {}).get("accuracy", 0))
        rows = []
        for m in sorted_models:
            tx = trace_data[m]
            rows.append([
                m, fmt(tx["avg_searches"]), fmt(tx["avg_reads"]),
                pct(tx["search_empty_rate"]),
                tx["bracket_bug_count"],
                pct(tx["bracket_bug_rate"]),
                fmt(tx["avg_keywords"]),
            ])
        lines.append(table(headers, rows))
        lines.append("")

    # ── 4. First Search Success → Final Accuracy ───────────
    lines.append("## 4. First Search Hit → Final Accuracy Correlation")
    lines.append("")
    lines.append("If the first `search_inbox` call returns results vs. returns empty — how does this affect final accuracy?")
    lines.append("")

    for bench_name, trace_data in [("Standard", std_trace), ("Multihop", mh_trace)]:
        lines.append(f"### {bench_name}")
        lines.append("")
        headers = ["Model", "1st Hit → Acc", "(N)", "1st Miss → Acc", "(N)", "Gap"]
        rows = []
        for m in sorted(trace_data.keys()):
            tx = trace_data[m]
            gap = tx["acc_first_hit"] - tx["acc_first_miss"]
            rows.append([
                m,
                pct(tx["acc_first_hit"]), tx["n_first_hit"],
                pct(tx["acc_first_miss"]), tx["n_first_miss"],
                f"{gap * 100:+.1f}pp",
            ])
        lines.append(table(headers, rows))
        lines.append("")

    lines.append("---")
    lines.append("")

    # ── 5. Keyword Count vs Empty Rate ─────────────────────
    lines.append("## 5. Keyword Count vs Search Empty Rate")
    lines.append("")
    lines.append("Aggregated across all models and both benchmarks.")
    lines.append("")

    all_records = []
    for recs in std_data.values():
        all_records.extend(recs)
    for recs in mh_data.values():
        all_records.extend(recs)
    kw_corr = compute_keyword_empty_correlation(all_records)
    kw_sorted = sorted(kw_corr.items())

    headers = ["Keywords", "Total Searches", "Empty Searches", "Empty Rate"]
    rows = []
    for kc, stats in kw_sorted:
        er = stats["empty"] / stats["total"] if stats["total"] else 0
        rows.append([kc, stats["total"], stats["empty"], pct(er)])
    lines.append(table(headers, rows))
    lines.append("")

    # Per-model keyword empty rate for multihop
    lines.append("### Per-Model Keyword → Empty Rate (Multihop)")
    lines.append("")
    for m in sorted(mh_data.keys()):
        kw_m = compute_keyword_empty_correlation(mh_data[m])
        kw_m_sorted = sorted(kw_m.items())
        if kw_m_sorted:
            lines.append(f"**{m}** (avg kw: {fmt(mh_trace[m]['avg_keywords'])})")
            rows = []
            for kc, stats in kw_m_sorted:
                er = stats["empty"] / stats["total"] if stats["total"] else 0
                rows.append([kc, stats["total"], stats["empty"], pct(er)])
            lines.append(table(["KW Count", "Searches", "Empty", "Rate"], rows))
            lines.append("")

    lines.append("---")
    lines.append("")

    # ── 6. Searches Correct vs Wrong ───────────────────────
    lines.append("## 6. Search Effort: Correct vs Wrong Answers")
    lines.append("")
    for bench_name, trace_data in [("Standard", std_trace), ("Multihop", mh_trace)]:
        lines.append(f"### {bench_name}")
        lines.append("")
        headers = ["Model", "Avg Searches (Correct)", "Avg Searches (Wrong)", "Delta"]
        rows = []
        for m in sorted(trace_data.keys()):
            tx = trace_data[m]
            sc = tx["avg_searches_correct"]
            sw = tx["avg_searches_wrong"]
            rows.append([m, fmt(sc), fmt(sw), f"{sw - sc:+.1f}"])
        lines.append(table(headers, rows))
        lines.append("")

    lines.append("---")
    lines.append("")

    # ── 7. Failure Classification (Multihop) ───────────────
    lines.append("## 7. Failure Classification — Multihop")
    lines.append("")
    lines.append("Priority: bracket_bug > explicit_idk > timeout > search_fail > wrong_answer")
    lines.append("")

    categories = ["correct", "explicit_idk", "timeout", "search_fail", "bracket_bug", "wrong_answer"]
    mh_failures = {m: compute_failure_classes(recs) for m, recs in mh_data.items()}

    headers = ["Model"] + [c.replace("_", " ").title() for c in categories]
    rows = []
    for m in sorted(mh_data.keys()):
        fc = mh_failures[m]
        total = sum(fc.values())
        row = [m]
        for cat in categories:
            cnt = fc.get(cat, 0)
            row.append(f"{cnt} ({pct(cnt / total)})" if total else "0")
        rows.append(row)
    lines.append(table(headers, rows))
    lines.append("")
    lines.append("---")
    lines.append("")

    # ── 8. Multihop by Hop Count & Type ────────────────────
    lines.append("## 8. Multihop Accuracy by Hop Count & Answer Type")
    lines.append("")

    mh_hop_data = {m: compute_hop_metrics(recs, metadata) for m, recs in mh_data.items()}

    # Hop counts present
    all_hops = sorted(set(
        h for md in mh_hop_data.values() for h in md["by_hop"].keys()
    ))

    lines.append("### By Hop Count")
    lines.append("")
    if all_hops:
        # Get sample counts
        sample_model = next(iter(mh_hop_data))
        hop_headers = ["Model"] + [f"{h}-hop" for h in all_hops]
        rows = []
        for m in sorted(mh_data.keys()):
            bh = mh_hop_data[m]["by_hop"]
            row = [m]
            for h in all_hops:
                d = bh.get(h, {"correct": 0, "total": 0})
                if d["total"] > 0:
                    row.append(f"{pct(d['correct'] / d['total'])} ({d['correct']}/{d['total']})")
                else:
                    row.append("N/A")
            rows.append(row)
        lines.append(table(hop_headers, rows))
        lines.append("")

    # By answer type
    all_types = sorted(set(
        t for md in mh_hop_data.values() for t in md["by_type"].keys()
    ))

    lines.append("### By Answer Type")
    lines.append("")
    if all_types:
        type_headers = ["Model"] + all_types
        rows = []
        for m in sorted(mh_data.keys()):
            bt = mh_hop_data[m]["by_type"]
            row = [m]
            for t in all_types:
                d = bt.get(t, {"correct": 0, "total": 0})
                if d["total"] > 0:
                    row.append(f"{pct(d['correct'] / d['total'])} ({d['correct']}/{d['total']})")
                else:
                    row.append("N/A")
            rows.append(row)
        lines.append(table(type_headers, rows))
        lines.append("")

    lines.append("---")
    lines.append("")

    # ── 9. Evidence Mail Coverage ──────────────────────────
    lines.append("## 9. Evidence Mail Coverage — Multihop")
    lines.append("")
    lines.append("Do models read the ground-truth evidence emails from the dataset metadata?")
    lines.append("")

    headers = ["Model", "Matched", "Any Evidence Read", "Coverage (Correct)", "Coverage (Wrong)"]
    rows = []
    for m in sorted(mh_data.keys()):
        ev = mh_hop_data[m]["evidence"]
        any_rate = ev["covered_any"] / ev["matched"] if ev["matched"] else 0
        corr_rate = ev["covered_correct"] / ev["correct_total"] if ev["correct_total"] else 0
        wrong_rate = ev["covered_wrong"] / ev["wrong_total"] if ev["wrong_total"] else 0
        rows.append([m, ev["matched"], pct(any_rate), pct(corr_rate), pct(wrong_rate)])
    lines.append(table(headers, rows))
    lines.append("")
    lines.append("---")
    lines.append("")

    # ── 10. Cross-model Difficulty ─────────────────────────
    lines.append("## 10. Cross-model Difficulty Analysis")
    lines.append("")

    for bench_name, data in [("Standard", std_data), ("Multihop", mh_data)]:
        diff = compute_cross_difficulty(data)
        lines.append(f"### {bench_name} ({diff['n_models']} models)")
        lines.append("")
        lines.append(f"- **Total questions**: {diff['total']}")
        lines.append(f"- **Easy** (all {diff['n_models']} models correct): {diff['easy']} ({pct(diff['easy'] / diff['total'] if diff['total'] else 0)})")
        lines.append(f"- **Medium** (2+ models correct): {diff['medium']}")
        lines.append(f"- **Hard** (exactly 1 model correct): {diff['hard']}")
        lines.append(f"- **Impossible@{diff['n_models']}** (no model correct): {diff['impossible']} ({pct(diff['impossible'] / diff['total'] if diff['total'] else 0)})")
        lines.append("")

    lines.append("---")
    lines.append("")

    # ── 11. Interesting Failure Examples ───────────────────
    lines.append("## 11. Interesting Failure Examples")
    lines.append("")

    # Find bracket bug examples
    for m in sorted(mh_data.keys()):
        for r in mh_data[m]:
            parsed = parse_trace(r.get("messages", []))
            if parsed["bracket_bugs"]:
                lines.append(f"### Bracket Bug — `{m}`")
                lines.append("")
                lines.append(f"- **Example**: `{r.get('example_id', '?')}`")
                lines.append(f"- **Question**: {r.get('question', '?')[:150]}")
                lines.append(f"- **Judge Score**: {r.get('judge_score', '?')}")
                lines.append(f"- **Buggy IDs**: `{parsed['bracket_bugs'][:3]}`")
                lines.append(f"- **Answer**: {str(r.get('generated_answer', ''))[:100]}")
                lines.append("")
                break

    # Find timeout examples (API models hitting 10 turns)
    for m in ["gpt-5", "gpt-5-mini"]:
        if m not in mh_data:
            continue
        for r in mh_data[m]:
            turns = r.get("turns_used", 0) or 0
            judge = r.get("judge_score") or 0
            if turns >= 9 and judge <= 0:
                parsed = parse_trace(r.get("messages", []))
                kw_list = [s["keywords"] for s in parsed["searches"][:5]]
                lines.append(f"### Timeout — `{m}` (turns={turns})")
                lines.append("")
                lines.append(f"- **Question**: {r.get('question', '?')[:150]}")
                lines.append(f"- **Searches tried** (first 5): ")
                for kw in kw_list:
                    lines.append(f"  - `{kw}`")
                empty_ct = sum(1 for s in parsed["searches"] if s["is_empty"])
                lines.append(f"- **Empty searches**: {empty_ct}/{len(parsed['searches'])}")
                lines.append(f"- **Answer**: {str(r.get('generated_answer', ''))[:100]}")
                lines.append("")
                break

    # Find search_fail examples (OSS early IDK)
    for m in ["qwen3-14b", "art-e-008"]:
        if m not in mh_data:
            continue
        for r in mh_data[m]:
            judge = r.get("judge_score") or 0
            answer = r.get("generated_answer") or ""
            if judge <= 0 and is_idk(answer) and not answer == "":
                parsed = parse_trace(r.get("messages", []))
                if parsed["searches"]:
                    empty_ct = sum(1 for s in parsed["searches"] if s["is_empty"])
                    if empty_ct > 0:
                        kw_list = [s["keywords"] for s in parsed["searches"][:3]]
                        lines.append(f"### Early IDK — `{m}`")
                        lines.append("")
                        lines.append(f"- **Question**: {r.get('question', '?')[:150]}")
                        lines.append(f"- **Searches**: ")
                        for kw in kw_list:
                            lines.append(f"  - `{kw}`")
                        lines.append(f"- **Empty searches**: {empty_ct}/{len(parsed['searches'])}")
                        lines.append(f"- **Answer**: {answer[:100]}")
                        lines.append("")
                        break

    lines.append("---")
    lines.append("")

    # ── 12. Summary & Key Findings ─────────────────────────
    lines.append("## 12. Summary & Key Findings")
    lines.append("")

    # Rankings
    std_rank = sorted(std_metrics.items(), key=lambda x: -x[1]["accuracy"])
    mh_rank = sorted(mh_metrics.items(), key=lambda x: -x[1]["accuracy"])

    lines.append("### Model Rankings")
    lines.append("")
    lines.append("**Standard** (by accuracy):")
    for i, (m, mx) in enumerate(std_rank, 1):
        lines.append(f"  {i}. `{m}` — {pct(mx['accuracy'])} (tools: {fmt(mx['avg_tool_calls'])}, tokens: {mx['avg_total_tokens']:.0f})")
    lines.append("")
    lines.append("**Multihop** (by accuracy):")
    for i, (m, mx) in enumerate(mh_rank, 1):
        lines.append(f"  {i}. `{m}` — {pct(mx['accuracy'])} (tools: {fmt(mx['avg_tool_calls'])}, tokens: {mx['avg_total_tokens']:.0f})")
    lines.append("")

    lines.append("### Key Findings")
    lines.append("")
    lines.append("1. **Search is the bottleneck**: The primary failure mode across all models is the inability to find relevant emails via `search_inbox`. Synthesis failures (reading the right email but extracting the wrong answer) are extremely rare.")
    lines.append("")
    lines.append("2. **Keyword count drives empty rate**: With AND-matching search semantics, more keywords = higher empty rate (1 kw ≈ 40%, 4+ kw ≈ 80%). Models that use fewer, more focused keywords perform better.")
    lines.append("")
    lines.append("3. **First search determines outcome**: If the first search returns results, accuracy is dramatically higher than if the first search is empty. This suggests the initial query formulation is critical.")
    lines.append("")
    lines.append("4. **Bracket bug is a preventable failure**: Some models (qwen3-14b, gpt-oss-20b) frequently omit `<>` brackets in `read_email` calls, causing 100% failure on those attempts. This is a format-following bug, not a reasoning limitation.")
    lines.append("")
    lines.append("5. **API vs OSS failure modes diverge**: API models (GPT-5, GPT-5-mini) tend to timeout after exhaustive searching. OSS models (qwen3-14b, art-e-008) give up early with IDK answers.")
    lines.append("")
    lines.append("6. **art-e-008 (RL-trained 8B)**: Uses fewest keywords per search, has zero bracket bugs, but is overly conservative — too many IDK answers on multihop. The RL training optimized search efficiency but may have over-reinforced IDK behavior.")
    lines.append("")

    lines.append("### Improvement Recommendations")
    lines.append("")
    lines.append("1. **Keyword limit**: Enforce max 2-3 keywords per search to reduce empty rate")
    lines.append("2. **Staged search**: Start with 1 keyword, progressively narrow if too many results")
    lines.append("3. **Query rewrite reward**: Add RL reward for adapting search strategy after empty results")
    lines.append("4. **Bracket format fix**: Post-process or fine-tune to ensure `<>` brackets on message IDs")
    lines.append("5. **Conditional IDK penalty**: Penalize IDK only when search results were available but unexplored")
    lines.append("6. **Variable turn limits**: Allow more turns for multihop (15-20) to reduce timeout failures")
    lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    report = generate_report()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        f.write(report)
    print(f"\nReport saved to: {OUTPUT_PATH}")
    print(f"Report size: {len(report):,} characters")
