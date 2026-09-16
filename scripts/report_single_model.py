# -*- coding: utf-8 -*-
"""单模型 Markdown 展示报告。

指标口径：
- 通过率(有效) = true / (true + false)   —— 主口径，排除 null 与 error
- 命中率(总)   = true / 总用例数          —— 辅口径
- null / error 单独统计
- 降级计数：json_fallback_raw / json_fallback_whole / empty / missing_model_output
"""

import argparse
import json
import os
from datetime import datetime

from result_store import load_jsonl


def aggregate(records):
    """按 rubric_id 聚合统计。"""
    by_rubric = {}
    for rec in records:
        rid = rec.get("rubric_id") or ""
        bucket = by_rubric.setdefault(rid, {
            "rubric_id": rid, "criterion_head": rec.get("rubric_criterion", "")[:60],
            "true": 0, "false": 0, "null": 0, "error": 0,
            "degrade_raw": 0, "degrade_whole": 0, "empty": 0, "missing": 0,
            "false_examples": [], "error_examples": [],
        })
        cm = rec.get("criteria_met")
        if cm is True:
            bucket["true"] += 1
        elif cm is False:
            bucket["false"] += 1
            if len(bucket["false_examples"]) < 8:
                bucket["false_examples"].append(rec)
        elif cm is None:
            if rec.get("error"):
                bucket["error"] += 1
                if len(bucket["error_examples"]) < 5:
                    bucket["error_examples"].append(rec)
            else:
                bucket["null"] += 1
        est = rec.get("extract_status") or ""
        if "json_fallback_raw" in est:
            bucket["degrade_raw"] += 1
        elif "json_fallback_whole" in est:
            bucket["degrade_whole"] += 1
        elif est == "empty":
            bucket["empty"] += 1
        elif est == "missing_model_output":
            bucket["missing"] += 1

    return by_rubric


def summarize(records, total_cases):
    s = {"true": 0, "false": 0, "null": 0, "error": 0,
         "degrade_raw": 0, "degrade_whole": 0, "empty": 0, "missing": 0}
    for rec in records:
        cm = rec.get("criteria_met")
        if cm is True:
            s["true"] += 1
        elif cm is False:
            s["false"] += 1
        elif rec.get("error"):
            s["error"] += 1
        else:
            s["null"] += 1
        est = rec.get("extract_status") or ""
        if "json_fallback_raw" in est:
            s["degrade_raw"] += 1
        elif "json_fallback_whole" in est:
            s["degrade_whole"] += 1
        elif est == "empty":
            s["empty"] += 1
        elif est == "missing_model_output":
            s["missing"] += 1
    s["effective"] = s["true"] + s["false"]
    s["total"] = len(records)
    s["pass_effective"] = s["true"] / s["effective"] if s["effective"] else None
    s["pass_total"] = s["true"] / len(records) if records else None
    return s


def aggregate_by_case(records):
    """按用例（case_id）聚合统计：true/false/null/error 计数 + 用例通过率 + 是否全对。

    全对 = 该回复所有 rubric 评分无 false 且无 error（全部 true 或 null）。
    """
    by_case = {}
    for rec in records:
        cid = rec.get("case_id") or "unknown"
        b = by_case.setdefault(cid, {
            "case_id": cid, "true": 0, "false": 0, "null": 0, "error": 0,
            "false_rubrics": [],
        })
        cm = rec.get("criteria_met")
        if cm is True:
            b["true"] += 1
        elif cm is False:
            b["false"] += 1
            b["false_rubrics"].append(rec.get("rubric_id", ""))
        elif rec.get("error"):
            b["error"] += 1
        else:
            b["null"] += 1
    for b in by_case.values():
        eff = b["true"] + b["false"]
        b["pass"] = b["true"] / eff if eff else None
        b["all_pass"] = (b["false"] == 0 and b["error"] == 0)
    return by_case


def render_md(model, records, total_cases, rubrics_map, examples=5):
    s = summarize(records, total_cases)
    by_rubric = aggregate(records)
    by_case = aggregate_by_case(records)

    all_pass_cases = sum(1 for b in by_case.values() if b["all_pass"])
    all_pass_rate = all_pass_cases / len(by_case) if by_case else None
    total_pass_rate = s["pass_effective"]

    lines = []
    lines.append(f"# 模型测评报告：{model}\n")
    lines.append(f"- 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- 判分用例：{total_cases} 条 × {len(by_rubric)} 条 rubric（共 {len(records)} 条记录）")
    lines.append("")

    lines.append("## 总览\n")
    lines.append("| 指标 | 值 |")
    lines.append("|------|----|")
    lines.append(f"| **总通过率%**（true / 有效判分） | **{total_pass_rate * 100:.1f}%** |" if total_pass_rate is not None else "| 总通过率% | - |")
    lines.append(f"| **全通过率%**（回复无 false 且无 error） | **{all_pass_rate * 100:.1f}%**（{all_pass_cases}/{len(by_case)} 条） |" if all_pass_rate is not None else "| 全通过率% | - |")
    lines.append(f"| 有效判分 (true+false) | {s['effective']} |")
    lines.append(f"| true (无问题) | {s['true']} |")
    lines.append(f"| false (有问题) | {s['false']} |")
    lines.append(f"| null (不适用) | {s['null']} |")
    lines.append(f"| error (判分失败) | {s['error']} |")
    lines.append(f"| 片段降级 (整体) | {s['degrade_whole']} |")
    lines.append(f"| 片段降级 (原文) | {s['degrade_raw']} |")
    lines.append(f"| 片段为空 | {s['empty']} |")
    lines.append(f"| 模型输出缺失 | {s['missing']} |")
    lines.append("")

    # 主视角：按 20 条回复展示
    lines.append("## 按回复（20 条用例）明细\n")
    lines.append("| # | case_id | true | false | null | error | 用例通过率 | 全对 | 未通过 rubric |")
    lines.append("|---|---------|------|-------|------|-------|------------|------|---------------|")
    for idx, cid in enumerate(sorted(by_case.keys()), 1):
        b = by_case[cid]
        pass_v = f"{b['pass'] * 100:.0f}%" if b["pass"] is not None else "-"
        all_pass = "✓" if b["all_pass"] else "✗"
        bad = ",".join(sorted(set(b["false_rubrics"]))) or ""
        lines.append(f"| {idx} | {cid[:8]} | {b['true']} | {b['false']} | {b['null']} | {b['error']} | {pass_v} | {all_pass} | {bad} |")
    lines.append("")

    lines.append("## 分 rubric 明细（附录）\n")
    lines.append("| rubric_id | 判分标准 | true | false | null | error | 通过率(有效) |")
    lines.append("|-----------|----------|------|-------|------|-------|--------------|")
    for rid in sorted(by_rubric.keys()):
        b = by_rubric[rid]
        eff = b["true"] + b["false"]
        pass_eff = f"{b['true'] / eff * 100:.0f}%" if eff else "-"
        head = (b["criterion_head"] or rid).replace("|", "\\|")
        lines.append(f"| {rid} | {head} | {b['true']} | {b['false']} | {b['null']} | {b['error']} | {pass_eff} |")
    lines.append("")

    # false 样例
    has_false = any(b["false_examples"] for b in by_rubric.values())
    if has_false and examples:
        lines.append("## 问题点样例（false 典型）\n")
        for rid in sorted(by_rubric.keys()):
            b = by_rubric[rid]
            if not b["false_examples"]:
                continue
            lines.append(f"### {rid}\n")
            lines.append("| case_id | 提取片段(截断) | judge 说明 | 证据 |")
            lines.append("|---------|----------------|------------|------|")
            for rec in b["false_examples"][:examples]:
                excerpt = rec.get("excerpt_preview", "")
                expl = rec.get("explanation", "").replace("\n", " ").replace("|", "\\|")
                ev = rec.get("evidence", "").replace("\n", " ").replace("|", "\\|")[:120]
                lines.append(f"| {rec.get('case_id', '')} | {excerpt[:80]} | {expl[:80]} | {ev} |")
            lines.append("")

    # 错误样例
    has_err = any(b["error_examples"] for b in by_rubric.values())
    if has_err:
        lines.append("## 判分失败样例\n")
        for rid in sorted(by_rubric.keys()):
            b = by_rubric[rid]
            if not b["error_examples"]:
                continue
            lines.append(f"### {rid}\n")
            lines.append("| case_id | error |")
            lines.append("|---------|-------|")
            for rec in b["error_examples"]:
                err = (rec.get("error") or "").replace("|", "\\|")[:120]
                lines.append(f"| {rec.get('case_id', '')} | {err} |")
            lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="单模型 Markdown 报告")
    parser.add_argument("--config", default=r"D:\PycharmProject\git\Health_Management_Diabetes\local_eval\config.json")
    parser.add_argument("--model", required=True)
    parser.add_argument("--examples", type=int, default=0)
    parser.add_argument("--run-tag", default="")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)
    paths = config["paths"]

    # 找最新 run_tag
    raw_root = os.path.join(paths["out_dir"], "raw_judge")
    if args.run_tag:
        run_tag = args.run_tag
    else:
        tags = sorted(os.listdir(raw_root)) if os.path.isdir(raw_root) else []
        if not tags:
            print("[ERROR] 没有 raw_judge 结果目录，请先运行 judge")
            return
        run_tag = tags[-1]

    result_file = os.path.join(raw_root, run_tag, f"{args.model}.jsonl")
    if not os.path.exists(result_file):
        print(f"[ERROR] 结果不存在: {result_file}")
        return
    records = load_jsonl(result_file)

    # 用例数
    test_rows = load_jsonl(paths["test_set"])
    total_cases = len(test_rows)

    # 每个 rubric 的 criterion 头（聚合显示用，兼容 rubric_sets dict 和旧 rubrics list）
    rubrics_map = {}
    rubric_files = paths.get("rubric_sets") or {}
    if not rubric_files and paths.get("rubrics"):
        rubric_files = {"default": paths["rubrics"]}
    for rp in rubric_files.values():
        if isinstance(rp, list):
            paths_list = rp
        else:
            paths_list = [rp]
        for p in paths_list:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            for item in data["items"]:
                rubrics_map[item.get("tags", [""])[0] if item.get("tags") else ""] = item.get("criterion", "")

    examples = args.examples or config.get("report", {}).get("single_examples", 5)
    md = render_md(args.model, records, total_cases, rubrics_map, examples=examples)

    out_dir = os.path.join(paths["out_dir"], "report_single")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{args.model}.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"报告已生成: {out_path}")
    print(md[:1500])


if __name__ == "__main__":
    main()
