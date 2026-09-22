# -*- coding: utf-8 -*-
"""对 results_aikang-line17-OPD-0911_qwen3-8-27B-dspv4_sft_noRAG.jsonl 中的模型输出运行四高一重格式检查。

模型输出位置：每条记录 sessions[0].conversation_history[-1].content
复用 check_format_llm_aikang_四高一重_0724.py 的检查逻辑（JSON解析 + 字段结构 + 语义/枚举/完整性）。
"""
import importlib.util
import json
import sys
from collections import Counter

BASE = r"<PROJECT_ROOT>"
RESULT_FILE = BASE + r"\result\results_aikang-line17-OPD-0911_qwen3-8-27B-dspv4_sft_noRAG.jsonl"
CHECK_SCRIPT = BASE + r"\bojie\ikang_trans\check_format_llm_aikang_四高一重_0724.py"
OUTPUT_FILE = BASE + r"\result\check_result_qwen3-8-27B-dspv4_sft_noRAG.txt"

# 加载检查脚本模块（不触发 main）
spec = importlib.util.spec_from_file_location("check_format_0724", CHECK_SCRIPT)
chk = importlib.util.module_from_spec(spec)
sys.modules["check_format_0724"] = chk
spec.loader.exec_module(chk)


def extract_model_outputs(result_file: str):
    """从评测结果JSONL中提取目标模型的原始输出。"""
    outputs = []
    with open(result_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            s0 = d["sessions"][0]
            ch = s0.get("conversation_history") or []
            if not ch:
                outputs.append(None)
                continue
            last = ch[-1]
            content = last.get("content") if isinstance(last, dict) else None
            outputs.append(content)
    return outputs


def main():
    outputs = extract_model_outputs(RESULT_FILE)
    total = len(outputs)

    # 汇总指标
    parse_ok = 0
    pass_all = 0
    fail_count = 0
    missing_counter = Counter()
    extra_counter = Counter()
    type_counter = Counter()
    semantic_counter = Counter()
    parse_fail_records = []
    per_record_detail = []

    for i, content in enumerate(outputs):
        line_no = i + 1
        if content is None:
            parse_fail_records.append(line_no)
            per_record_detail.append((line_no, None, "NO_CONTENT"))
            fail_count += 1
            continue

        obj, parse_warnings = chk.parse_json_lenient(content)

        if obj is None:
            parse_fail_records.append(line_no)
            per_record_detail.append((line_no, content[:120], "PARSE_FAIL"))
            fail_count += 1
            continue

        parse_ok += 1
        missing_fields, extra_fields, type_warnings = chk.compare_schema(obj, chk.STANDARD_SCHEMA)
        semantic_warnings = chk.check_semantic_rules(obj)

        for f in missing_fields:
            missing_counter[f] += 1
        for f in extra_fields:
            extra_counter[f] += 1
        for w in type_warnings:
            type_counter[w] += 1
        for w in semantic_warnings:
            semantic_counter[w] += 1

        if not missing_fields and not extra_fields and not type_warnings and not semantic_warnings:
            pass_all += 1
        else:
            fail_count += 1
        per_record_detail.append((line_no, obj, (missing_fields, extra_fields, type_warnings, semantic_warnings)))

    # ===================== 输出报告 =====================
    lines_out = []
    def log(msg=""):
        lines_out.append(str(msg))

    log("=" * 90)
    log("格式检查报告：results_aikang-line17-OPD-0911_qwen3-8-27B-dspv4_sft_noRAG")
    log("模型输出来源：sessions[0].conversation_history[-1].content")
    log(f"总记录数：{total}")
    log(f"JSON 解析成功：{parse_ok}/{total}（通过率 {parse_ok / total * 100:.2f}%）")
    log(f"JSON 解析失败：{total - parse_ok}/{total}")
    log(f"字段+内容全通过：{pass_all}/{total}（通过率 {pass_all / total * 100:.2f}%）")
    log(f"存在问题的记录：{fail_count}/{total}")
    if parse_fail_records:
        log(f"解析失败的行：{parse_fail_records}")

    log("")
    log("-" * 90)
    log(f"缺失字段 TOP（路径 | 出现次数）")
    for path, cnt in missing_counter.most_common():
        log(f"  {path} | {cnt}")

    log("")
    log("-" * 90)
    log(f"多余字段 TOP（路径 | 出现次数）")
    for path, cnt in extra_counter.most_common():
        log(f"  {path} | {cnt}")

    log("")
    log("-" * 90)
    log(f"类型警告 TOP（路径/消息 | 出现次数）")
    for msg, cnt in type_counter.most_common():
        log(f"  {msg} | {cnt}")

    log("")
    log("-" * 90)
    log(f"内容/枚举问题 TOP（消息 | 出现次数）")
    for msg, cnt in semantic_counter.most_common():
        log(f"  {msg} | {cnt}")

    log("")
    log("-" * 90)
    log("逐条记录汇总")
    for line_no, obj, detail in per_record_detail:
        if obj is None:
            log(f"  第 {line_no} 条：无输出/解析失败")
            continue
        missing_fields, extra_fields, type_warnings, semantic_warnings = detail
        n_miss = len(missing_fields)
        n_extra = len(extra_fields)
        n_type = len(type_warnings)
        n_sem = len(semantic_warnings)
        status = "通过" if (n_miss + n_extra + n_type + n_sem) == 0 else f"未通过"
        log(f"  第 {line_no} 条：{status}（缺字段 {n_miss}、多余字段 {n_extra}、类型 {n_type}、内容/枚举 {n_sem}）")

    text = "\n".join(lines_out)
    print(text)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print(f"\n[已保存] {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
