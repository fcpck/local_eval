# -*- coding: utf-8 -*-
"""预处理：产出爬取用数据集 {id, system_prompt, user_prompt}。

system_prompt 来源：
  1. 若 system_prompt_v8.8 模板含 {{管理总周期}} 占位符 → 用测试集行的 management_total_period 替换
  2. 否则直接取测试集行内已有的 system_prompt 字段（当前 20条测试集即此情况）

user_prompt 直接取自测试集每行。
输出供 batch_infer.py / crawl_answers.py 消费。
"""

import argparse
import json
import os

from result_store import load_jsonl, append_jsonl


def build_crawl_dataset(test_set_path, sysprompt_path, out_path):
    rows = load_jsonl(test_set_path)
    sys_lines = load_jsonl(sysprompt_path)
    sys_template = sys_lines[0]["content"] if sys_lines else ""
    has_placeholder = "{{管理总周期}}" in sys_template

    out_rows = []
    for row in rows:
        period = row.get("management_total_period", "")
        if has_placeholder:
            system_prompt = sys_template.replace("{{管理总周期}}", period)
        else:
            system_prompt = row.get("system_prompt", "")
        out_rows.append({
            "id": row.get("id", ""),
            "system_prompt": system_prompt,
            "user_prompt": row.get("user_prompt", ""),
        })

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return out_rows


def main():
    parser = argparse.ArgumentParser(description="预处理产出爬取用数据集")
    parser.add_argument("--config", default=r"D:\PycharmProject\git\Health_Management_Diabetes\local_eval\config.json")
    parser.add_argument("--out", default="", help="输出路径，默认 data/crawl_dataset.jsonl")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)
    paths = config["paths"]

    out_path = args.out or os.path.join(paths["data_dir"], "crawl_dataset.jsonl")
    rows = build_crawl_dataset(paths["test_set"], paths["system_prompt"], out_path)
    n_placeholder = sum(1 for r in rows if "{{管理总周期}}" in r["system_prompt"])
    print(f"已生成 {len(rows)} 条爬取用数据集: {out_path}")
    print(f"含未替换占位符的行数: {n_placeholder}")


if __name__ == "__main__":
    main()
