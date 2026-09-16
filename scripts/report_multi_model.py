# -*- coding: utf-8 -*-
"""多模型分数汇总报告（Markdown + Excel）。

矩阵：行 = rubric，列 = 模型，值 = 通过率(有效) = true/(true+false)。
附平均行 + null/error 汇总行。
"""

import argparse
import json
import os
from datetime import datetime

from result_store import load_jsonl, rubric_id_of


def build_matrix(results_by_model, total_cases):
    """返回 {rubric_id: {"head": str, "models": {model: {"true","false","null","error","pass"}}}}"""
    rubrics = {}
    for model, records in results_by_model.items():
        for rec in records:
            rid = rec.get("rubric_id") or "unknown"
            b = rubrics.setdefault(rid, {"head": rec.get("rubric_criterion", "")[:60], "models": {}})
            m = b["models"].setdefault(model, {"true": 0, "false": 0, "null": 0, "error": 0})
            cm = rec.get("criteria_met")
            if cm is True:
                m["true"] += 1
            elif cm is False:
                m["false"] += 1
            elif rec.get("error"):
                m["error"] += 1
            else:
                m["null"] += 1
    for rid, b in rubrics.items():
        for model, m in b["models"].items():
            eff = m["true"] + m["false"]
            m["pass"] = m["true"] / eff if eff else None
            m["total"] = total_cases
    return rubrics


def _case_stats(records):
    """按用例聚合：返回 (总true, 总false, 总null, 总error, 全对用例数)。"""
    by_case = {}
    t = f = n = e = 0
    for rec in records:
        cm = rec.get("criteria_met")
        cid = rec.get("case_id") or "unknown"
        b = by_case.setdefault(cid, {"false": 0, "error": 0})
        if cm is True:
            t += 1
        elif cm is False:
            f += 1
            b["false"] += 1
        elif rec.get("error"):
            e += 1
            b["error"] += 1
        else:
            n += 1
    all_pass = sum(1 for b in by_case.values() if b["false"] == 0 and b["error"] == 0)
    return t, f, n, e, all_pass


def render_markdown(matrix, model_list, results_by_model):
    lines = []
    lines.append("# 多模型分数汇总\n")
    lines.append(f"- 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- 报告口径：总通过率% = true/(true+false)；全通过率% = 回复无 false 且无 error 的用例占比")
    lines.append(f"- 模型数：{len(model_list)} 个\n")

    # 模型对比（主视角：按回复统计）
    lines.append("## 模型对比（按 20 条回复）\n")
    lines.append("| 模型 | 总通过率% | 全通过率% | true | false | null | error |")
    lines.append("|------|-----------|-----------|------|-------|------|-------|")
    for model in model_list:
        records = results_by_model.get(model, [])
        t, f, n, e, all_pass = _case_stats(records)
        eff = t + f
        total_pass = t / eff if eff else None
        all_pass_rate = all_pass / len({r.get("case_id") for r in records}) if records else None
        lines.append(
            f"| {model} | {total_pass * 100:.1f}% | {all_pass_rate * 100:.1f}% | {t} | {f} | {n} | {e} |"
            if total_pass is not None and all_pass_rate is not None
            else f"| {model} | - | - | {t} | {f} | {n} | {e} |"
        )

    # 通过率矩阵（附录）
    lines.append("\n## 分 rubric 通过率矩阵（附录）\n")
    lines.append("| rubric_id | " + " | ".join(model_list) + " |")
    lines.append("|-----------|" + "|".join(["------------"] * len(model_list)) + "|")
    for rid in sorted(matrix.keys()):
        b = matrix[rid]
        row = [rid]
        for model in model_list:
            m = b["models"].get(model, {})
            pass_v = m.get("pass")
            row.append(f"{pass_v * 100:.0f}%" if pass_v is not None else "-")
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)


def render_excel(matrix, model_list, out_path):
    try:
        import openpyxl
        from openpyxl.styles import PatternFill
    except ImportError:
        print("[WARN] openpyxl 不可用，跳过 Excel")
        return False

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "通过率矩阵"

    header = ["rubric_id"] + model_list
    ws.append(header)
    for rid in sorted(matrix.keys()):
        b = matrix[rid]
        row = [rid]
        for model in model_list:
            m = b["models"].get(model, {})
            pass_v = m.get("pass")
            row.append(pass_v if pass_v is not None else "")
        ws.append(row)

    # 汇总行
    ws.append([])
    sum_row = ["模型汇总"]
    for model in model_list:
        passes = []
        for rid, b in matrix.items():
            m = b["models"].get(model, {})
            if m.get("pass") is not None:
                passes.append(m["pass"])
        sum_row.append(sum(passes) / len(passes) if passes else "")
    ws.append(sum_row)

    # 条件着色：绿(高) 红(低)
    green = PatternFill(start_color="C6EFCE", end_color="C6EFCE", fill_type="solid")
    red = PatternFill(start_color="FFC7CE", end_color="FFC7CE", fill_type="solid")
    for row in ws.iter_rows(min_row=2, max_row=ws.max_row - 1, min_col=2, max_col=1 + len(model_list)):
        for cell in row:
            if isinstance(cell.value, (int, float)):
                if cell.value >= 0.8:
                    cell.fill = green
                elif cell.value < 0.6:
                    cell.fill = red

    ws.freeze_panes = "B2"
    for col in range(1, ws.max_column + 1):
        ws.column_dimensions[openpyxl.utils.get_column_letter(col)].width = 22

    wb.save(out_path)
    return True


def main():
    parser = argparse.ArgumentParser(description="多模型分数汇总报告")
    parser.add_argument("--config", default=r"D:\PycharmProject\git\Health_Management_Diabetes\local_eval\config.json")
    parser.add_argument("--models", default="", help="逗号分隔模型名；空=全部")
    parser.add_argument("--run-tag", default="")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)
    paths = config["paths"]

    raw_root = os.path.join(paths["out_dir"], "raw_judge")
    if args.run_tag:
        run_tag = args.run_tag
    else:
        tags = sorted(os.listdir(raw_root)) if os.path.isdir(raw_root) else []
        if not tags:
            print("[ERROR] 没有 raw_judge 结果目录，请先运行 judge")
            return
        run_tag = tags[-1]

    result_dir = os.path.join(raw_root, run_tag)
    if not os.path.isdir(result_dir):
        print(f"[ERROR] 结果目录不存在: {result_dir}")
        return

    test_rows = load_jsonl(paths["test_set"])
    total_cases = len(test_rows)

    if args.models:
        model_list = [m.strip() for m in args.models.split(",") if m.strip()]
    else:
        model_list = sorted(fn[:-6] for fn in os.listdir(result_dir) if fn.endswith(".jsonl"))
    if not model_list:
        print("[ERROR] 没有可用的模型结果")
        return

    results_by_model = {}
    for model in model_list:
        result_file = os.path.join(result_dir, f"{model}.jsonl")
        if os.path.exists(result_file):
            results_by_model[model] = load_jsonl(result_file)
        else:
            print(f"[WARN] 缺少 {model} 的结果文件，跳过")

    matrix = build_matrix(results_by_model, total_cases)

    md = render_markdown(matrix, list(results_by_model.keys()), results_by_model)
    out_dir = os.path.join(paths["out_dir"], "report_multi", run_tag)
    os.makedirs(out_dir, exist_ok=True)
    md_path = os.path.join(out_dir, "all_models.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"Markdown 已生成: {md_path}")

    xlsx_path = os.path.join(out_dir, "all_models.xlsx")
    if render_excel(matrix, list(results_by_model.keys()), xlsx_path):
        print(f"Excel 已生成: {xlsx_path}")

    print(md[:1200])


if __name__ == "__main__":
    main()
