#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对 8 个测评结果文件运行 check_format_llm_aikang_四高一重_0724.py 的完整检查逻辑,
将每个文件的【完整详细报告】(逐行 + 全部警告明细 + 总体统计)保存到
result/check_result_{备注}.txt, 并打印汇总表。

模型输出提取: sessions[0].conversation_history 中 role=assistant 消息的 content 字段。
"""

import importlib.util
import io
import json
import os
import sys
from contextlib import redirect_stdout

BASE = r"<PROJECT_ROOT>"
RESULT_DIR = os.path.join(BASE, "result")
CHECK_SCRIPT = os.path.join(
    BASE, "bojie", "ikang_trans", "check_format_llm_aikang_四高一重_0724.py"
)

# 加载原检查脚本的检查逻辑(不执行其 main)
spec = importlib.util.spec_from_file_location("check_format", CHECK_SCRIPT)
check_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check_mod)

FILES = [
    ("results_aikang_line11_OPD-Qwen3.6 35B A3B p1p2_wRAG 0905.jsonl", "aikang_line11_Qwen36_35B_p1p2_wRAG_0905"),
    ("results_DSv4flash_noRAG.jsonl", "DSv4flash_noRAG"),
    ("results_aikang POC_v4_wRAG.jsonl", "aikang_POC_v4_wRAG"),
    ("results_DSv4pro_0813_noRAG-分级准确率.jsonl", "DSv4pro_0813_noRAG"),
    ("results_Qwen3.6_35B_noRAG.jsonl", "Qwen3.6_35B_noRAG"),
    ("results_Qwen3.8_27B_noRAG.jsonl", "Qwen3.8_27B_noRAG"),
    ("results_aikang_line59_OPD-Qwen3.8 27B p1p2_wRAG 0905_wRAG_sichen.jsonl", "aikang_line59_Qwen38_27B_p1p2_wRAG_0905_sichen"),
    ("results_aikang_line57_OPD-Qwen3.8 27B p1p2_noRAG 0905_sichen.jsonl", "aikang_line57_Qwen38_27B_p1p2_noRAG_0905_sichen"),
]


def extract_model_output(d: dict) -> str:
    """从测评结果记录中提取模型输出字符串。"""
    sessions = d.get("sessions") or []
    if not sessions:
        return ""
    conv = sessions[0].get("conversation_history") or []
    for msg in conv:
        if msg.get("role") == "assistant" and msg.get("content"):
            return str(msg["content"])
    return ""


def run_full_check(name: str, outputs: list) -> tuple:
    """逐行运行与原脚本一致的检查,捕获 print_line_report 的完整输出。"""
    total = 0
    parsed_success = 0
    parsed_failed = 0
    passed = 0
    failed = 0

    buf = io.StringIO()
    with redirect_stdout(buf):
        for line_no, raw_line in enumerate(outputs, start=1):
            total += 1
            obj, parse_warnings = check_mod.parse_json_lenient(raw_line)
            if obj is None:
                parsed_failed += 1
                failed += 1
                stats = check_mod.build_line_check_stats(
                    obj=None, missing_fields=[], extra_fields=[],
                    type_warnings=[], semantic_warnings=[],
                )
                check_mod.print_line_report(
                    line_no=line_no, stats=stats, parse_warnings=parse_warnings,
                    missing_fields=[], extra_fields=[], type_warnings=[],
                    semantic_warnings=[],
                )
                continue
            parsed_success += 1

            missing_fields, extra_fields, type_warnings = check_mod.compare_schema(
                obj=obj, schema=check_mod.STANDARD_SCHEMA
            )
            semantic_warnings = check_mod.check_semantic_rules(obj)
            stats = check_mod.build_line_check_stats(
                obj=obj, missing_fields=missing_fields, extra_fields=extra_fields,
                type_warnings=type_warnings, semantic_warnings=semantic_warnings,
            )
            if not missing_fields and not extra_fields and not type_warnings and not semantic_warnings:
                passed += 1
            else:
                failed += 1
            check_mod.print_line_report(
                line_no=line_no, stats=stats, parse_warnings=parse_warnings,
                missing_fields=missing_fields, extra_fields=extra_fields,
                type_warnings=type_warnings, semantic_warnings=semantic_warnings,
            )

        # 总体统计(与原脚本 main 一致)
        buf.write("=" * 100 + "\n")
        buf.write("总体统计\n")
        checked_total = passed + failed
        pass_rate = (passed / checked_total * 100) if checked_total else 0
        buf.write(f"- 总行数：{total}\n")
        buf.write(f"- JSON 解析成功：{parsed_success}\n")
        buf.write(f"- JSON 解析失败：{parsed_failed}\n")
        buf.write(f"- 通过条目数/总检查条目数：{passed}/{checked_total}（通过率：{pass_rate:.2f}%）\n")
        buf.write(f"- 字段检查通过：{passed}\n")
        buf.write(f"- 字段检查未通过：{failed}\n")

    report = buf.getvalue()
    out_path = os.path.join(RESULT_DIR, f"check_result_{name}.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"[已保存] {out_path}  ({len(report)} 字符)")
    return total, parsed_success, parsed_failed, passed, failed


def main():
    summary = []
    for fname, name in FILES:
        fpath = os.path.join(RESULT_DIR, fname)
        outputs = []
        with open(fpath, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                outputs.append(extract_model_output(d))
        total, ps, pf, passed, failed = run_full_check(name, outputs)
        checked = passed + failed
        json_rate = (ps / total * 100) if total else 0
        format_rate = (passed / checked * 100) if checked else 0
        summary.append((name, total, ps, pf, passed, failed, json_rate, format_rate))

    summary_path = os.path.join(RESULT_DIR, "check_result_8_models_summary.txt")
    lines = ["文件\t总行数\tJSON解析成功\tJSON解析失败\tJSON通过率\t格式通过\t格式失败\t格式完整性通过率"]
    for name, total, ps, pf, passed, failed, json_rate, format_rate in summary:
        lines.append(
            f"{name}\t{total}\t{ps}\t{pf}\t{json_rate:.2f}%\t{passed}\t{failed}\t{format_rate:.2f}%"
        )
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[已保存] {summary_path}")
    print("\n========== 汇总 ==========")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
