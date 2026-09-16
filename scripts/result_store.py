# -*- coding: utf-8 -*-
"""JSONL 读写与断点续跑工具（纯函数）。"""

import json
import os


def load_jsonl(path):
    """读取 JSONL，跳过空行与解析失败行。"""
    data = []
    if not os.path.exists(path):
        return data
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return data


def append_jsonl(path, record):
    """追加写入一条 JSONL 记录。"""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def format_key(model, case_id, rubric_id):
    """断点续跑唯一键。"""
    return f"{model}|{case_id}|{rubric_id}"


def existing_keys(path):
    """返回已判分的唯一键集合。"""
    keys = set()
    for rec in load_jsonl(path):
        key = rec.get("key") or format_key(rec.get("model", ""), rec.get("case_id", ""), rec.get("rubric_id", ""))
        if key and "|" in key:
            keys.add(key)
    return keys


def rubric_id_of(rubric, idx):
    """从 criterion 首行括号编号取稳定 rubric_id；无则用 tags 首段；再无则 idx。"""
    criterion = (rubric.get("criterion") or "").strip()
    m = __import__("re").search(r"^【\s*([A-Za-z0-9_\-]+)", criterion)
    if m:
        return m.group(1).strip()
    if isinstance(rubric.get("tags"), list) and rubric["tags"]:
        return str(rubric["tags"][0])
    return f"rubric_{idx}"
