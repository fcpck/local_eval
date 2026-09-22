#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型测评框架 - Rubric Judge
对每条模型回复 × 每个 rubric 调用一次 judge LLM。

用法:
  python judge_runner.py
  python judge_runner.py --limit-rows 3 --limit-rubrics 5    # 快速测试
  python judge_runner.py --excel out.xlsx          # 默认输出 Excel
"""

import json
import re
import os
import sys
import time
import asyncio
import argparse
from pathlib import Path
from datetime import datetime

import aiohttp

try:
    import openpyxl
except ImportError:
    openpyxl = None


# ==================== 配置 ====================

INPUT_JSONL = r"<REPLACE_WITH_INPUT_JSONL>"
RUBRICS_FILE = r"<REPLACE_WITH_RUBRICS_FILE>"
JUDGE_PROMPT_FILE = r"<REPLACE_WITH_JUDGE_PROMPT_FILE>"
OUTPUT_DIR = r"<REPLACE_WITH_OUTPUT_DIR>"

JUDGE_URL = "https://<MAAS_ENDPOINT>/compatible-mode/v1/chat/completions"
JUDGE_MODEL = "glm-5.2"
JUDGE_TOKEN = "sk-***QAUg"  # ← 运行前替换为真实 token

MAX_TOKENS = 2048
MAX_CONCURRENCY = 15
REQUEST_TIMEOUT = 180
MAX_RETRIES = 3
RETRY_BACKOFF = 5

# 每个字段提取后最大字符数（防止 prompt 过长）
MAX_FIELD_CHARS = 2000
MAX_PLAN_CHARS = 7000

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")


# ==================== 数据加载 ====================

def load_judge_prompt():
    with open(JUDGE_PROMPT_FILE, "r", encoding="utf-8") as f:
        return f.read().strip()


def load_rubrics():
    with open(RUBRICS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["items"]


def load_records(limit_rows=0):
    records = []
    with open(INPUT_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
            if limit_rows and len(records) >= limit_rows:
                break
    return records


# ==================== 字段提取 ====================

def extract_nested(plan, path):
    """按 'a.b.c' / 'a[0].b' 形式的 path 从 plan 中提取值。"""
    if not path:
        return None
    # 支持 a.b.c 和 a[0].b 两种形式
    tokens = re.findall(r"[^.\[\]]+", path)
    cur = plan
    for tok in tokens:
        if cur is None:
            return None
        if isinstance(cur, list):
            if tok.isdigit():
                idx = int(tok)
                if 0 <= idx < len(cur):
                    cur = cur[idx]
                else:
                    return None
            else:
                return None
        elif isinstance(cur, dict):
            cur = cur.get(tok, None)
        else:
            return None
    return cur


def extract_fields(plan, extract_key_path):
    """按 extract_key_path（逗号分隔的字段列表）从 plan 提取内容。"""
    paths = [p.strip() for p in extract_key_path.split(",") if p.strip()]
    if not paths:
        return "(no extract_key_path)"

    parts = []
    for path in paths:
        v = extract_nested(plan, path)
        if v is None:
            parts.append(f"[{path}] (字段不存在)")
            continue
        if isinstance(v, list):
            # 列表：限制条目数，每条序列化
            items = v[:5]
            for i, item in enumerate(items):
                s = json.dumps(item, ensure_ascii=False) if isinstance(item, (dict, list)) else str(item)
                if len(s) > MAX_FIELD_CHARS:
                    s = s[:MAX_FIELD_CHARS] + "...(截断)"
                parts.append(f"[{path}][{i}] {s}")
        elif isinstance(v, dict):
            s = json.dumps(v, ensure_ascii=False)
            if len(s) > MAX_FIELD_CHARS:
                s = s[:MAX_FIELD_CHARS] + "...(截断)"
            parts.append(f"[{path}] {s}")
        else:
            s = str(v)
            if len(s) > MAX_FIELD_CHARS:
                s = s[:MAX_FIELD_CHARS] + "...(截断)"
            parts.append(f"[{path}] {s}")

    joined = "\n".join(parts)
    if len(joined) > MAX_PLAN_CHARS:
        joined = joined[:MAX_PLAN_CHARS] + "\n...(整体截断)"
    return joined


# ==================== Prompt 构建 ====================

def build_conversation(record):
    """构建 conversation：用户消息 + 助手回复中按 rubric 提取的字段。"""
    sessions = record.get("sessions", [])
    if not sessions:
        return "", "no sessions"
    conv_history = sessions[0].get("conversation_history", [])
    if len(conv_history) < 3:
        return "", "conversation_history too short"

    user_msg = conv_history[1].get("content", "")
    assistant_content = conv_history[2].get("content", "")

    # 助手回复是管理方案 JSON 字符串
    try:
        plan = json.loads(assistant_content)
    except (json.JSONDecodeError, TypeError):
        return "", "assistant content is not valid JSON"

    return user_msg, plan


def build_judge_prompt(prompt_template, user_msg, plan, rubric):
    """填充 prompt 模板：{{conversation}} 和 {{rubric_item}}。"""
    extract_paths = rubric.get("extract_key_path", "")
    plan_extracted = extract_fields(plan, extract_paths)
    conversation = f"# 用户输入（含健康数据）\n{user_msg}\n\n# 助手回复（已按 rubric 提取相关字段）\n{plan_extracted}"
    rubric_item = rubric.get("criterion", "")
    return prompt_template.replace("{{conversation}}", conversation).replace("{{rubric_item}}", rubric_item)


# ==================== Judge LLM 调用 ====================

async def call_judge(session, prompt, sem):
    async with sem:
        start = time.time()
        payload = {
            "model": JUDGE_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": MAX_TOKENS,
            "temperature": 0.0,
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {JUDGE_TOKEN}",
        }
        last_err = None
        for attempt in range(MAX_RETRIES):
            try:
                timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)
                async with session.post(JUDGE_URL, json=payload, headers=headers, timeout=timeout) as resp:
                    if resp.status != 200:
                        text = await resp.text()
                        last_err = f"HTTP {resp.status}: {text[:200]}"
                        await asyncio.sleep(RETRY_BACKOFF)
                        continue
                    data = await resp.json()
                    choices = data.get("choices", [])
                    if not choices:
                        last_err = "No choices in response"
                        await asyncio.sleep(RETRY_BACKOFF)
                        continue
                    content = choices[0].get("message", {}).get("content", "")
                    latency = time.time() - start
                    return content, None, latency
            except asyncio.TimeoutError:
                last_err = f"Timeout after {REQUEST_TIMEOUT}s"
            except Exception as e:
                last_err = str(e)[:200]
            await asyncio.sleep(RETRY_BACKOFF)
        return None, last_err, time.time() - start


def parse_judge_response(content):
    """从 judge 回复中解析 {explanation, criteria_met}。"""
    if not content:
        return None, None, "empty content"
    text = content.strip()
    # 尝试直接解析
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj.get("explanation"), obj.get("criteria_met"), None
    except Exception:
        pass
    # 尝试从 markdown 代码块中提取
    m = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(1))
            if isinstance(obj, dict):
                return obj.get("explanation"), obj.get("criteria_met"), None
        except Exception:
            pass
    # 尝试从 { ... } 中提取
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            obj = json.loads(text[start : end + 1])
            if isinstance(obj, dict):
                return obj.get("explanation"), obj.get("criteria_met"), None
        except Exception:
            pass
    return None, None, f"Could not parse JSON. Preview: {text[:300]}"


# ==================== 主流程 ====================

async def run(args):
    print(f"=== Rubric Judge 测评框架启动 ===")
    print(f"时间: {TIMESTAMP}")
    print(f"输入: {INPUT_JSONL}")
    print(f"Rubrics: {RUBRICS_FILE}")
    print(f"Judge: {JUDGE_MODEL} @ {JUDGE_URL}")

    prompt_template = load_judge_prompt()
    rubrics = load_rubrics()
    records = load_records(args.limit_rows)

    if args.limit_rubrics:
        rubrics = rubrics[: args.limit_rubrics]

    total_calls = len(records) * len(rubrics)
    print(f"模型回复: {len(records)} 条, Rubrics: {len(rubrics)} 个")
    print(f"总调用次数: {total_calls}")
    print(f"并发数: {MAX_CONCURRENCY}")

    out_jsonl = Path(OUTPUT_DIR) / f"judge_results_{TIMESTAMP}.jsonl"
    out_log = Path(OUTPUT_DIR) / f"judge_log_{TIMESTAMP}.jsonl"
    out_jsonl.parent.mkdir(parents=True)
    out_log.parent.mkdir(parents=True)

    sem = asyncio.Semaphore(MAX_CONCURRENCY)
    completed = 0
    failed = 0
    start_time = time.time()

    connector = aiohttp.TcpConnector(limit=MAX_CONCURRENCY * 2, limit_per_host=MAX_CONCURRENCY * 2)
    async with aiohttp.ClientSession(connector=connector) as session:
        async def evaluate_one(sample_idx, record, rubric_idx, rubric):
            nonlocal completed, failed
            data_item_id = record.get("data_item_id", "")
            user_msg, plan = build_conversation(record)
            if not user_msg:
                err = plan if isinstance(plan, str) else plan
                result = {
                    "sample_idx": sample_idx, "data_item_id": data_item_id,
                    "rubric_idx": rubric_idx, "rubric_id": rubric.get("tags", [""])[0] if rubric.get("tags") else "",
                    "rubric_tags": ",".join(rubric.get("tags", [])),
                    "criteria_met": None, "explanation": "",
                    "raw_response": "", "error": err, "latency_s": 0,
                }
                completed += 1
                return result

            prompt = build_judge_prompt(prompt_template, user_msg, plan, rubric)
            content, error, latency = await call_judge(session, prompt, sem)
            if error:
                failed += 1
                result = {
                    "sample_idx": sample_idx, "data_item_id": data_item_id,
                    "rubric_idx": rubric_idx, "rubric_id": rubric.get("tags", [""])[0] if rubric.get("tags") else "",
                    "rubric_tags": ",".join(rubric.get("tags", [])),
                    "criteria_met": None, "explanation": "",
                    "raw_response": content or "", "error": error, "latency_s": round(latency, 2),
                }
            else:
                explanation, criteria_met, parse_err = parse_judge_response(content)
                result = {
                    "sample_idx": sample_idx, "data_item_id": data_item_id,
                    "rubric_idx": rubric_idx, "rubric_id": rubric.get("tags", [""])[0] if rubric.get("tags") else "",
                    "rubric_tags": ",".join(rubric.get("tags", [])),
                    "criteria_met": criteria_met, "explanation": explanation or "",
                    "raw_response": content[:2000], "error": parse_err or "", "latency_s": round(latency, 2),
                }
            completed += 1

            # 写入 JSONL
            with open(out_jsonl, "a", encoding="utf-8") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")

            # 进度播报
            if completed % 10 == 0 or completed == total_calls:
                elapsed = time.time() - start_time
                rate = completed / max(elapsed, 1)
                eta = (total_calls - completed) / max(rate, 0.01)
                print(f"  进度: {completed}/{total_calls} (失败 {failed}), 耗时 {elapsed:.0f}s, ETA {eta:.0f}s, 速率 {rate:.2f}/s")

            return result

        # 构建所有任务
        tasks = []
        for s_idx, record in enumerate(records):
            for r_idx, rubric in enumerate(rubrics):
                tasks.append(evaluate_one(s_idx, record, r_idx, rubric))

        print(f"\n开始执行 {len(tasks)} 个评测任务...")
        results = await asyncio.gather(*tasks, return_exceptions=True)

    total_elapsed = time.time() - start_time
    print(f"\n=== 完成 ===")
    print(f"总耗时: {total_elapsed:.1f}s")
    print(f"成功: {completed - failed}, 失败: {failed}")
    print(f"结果: {out_jsonl}")
    print(f"日志: {out_log}")

    # 统计
    valid_results = [r for r in results if isinstance(r, dict)]
    true_count = sum(1 for r in valid_results if r.get("criteria_met") is True)
    false_count = sum(1 for r in valid_results if r.get("criteria_met") is False)
    null_count = sum(1 for r in valid_results if r.get("criteria_met") is None)
    err_count = sum(1 for r in valid_results if r.get("error"))
    print(f"\n统计:")
    print(f"  criteria_met=true (无问题): {true_count}")
    print(f"  criteria_met=false (有问题): {false_count}")
    print(f"  criteria_met=null  (不涉及): {null_count}")
    print(f"  解析错误: {err_count}")

    # Excel 输出
    if not args.no_excel and openpyxl:
        excel_path = Path(OUTPUT_DIR) / f"judge_results_{TIMESTAMP}.xlsx"
        write_excel(valid_results, rubrics, excel_path)
        print(f"  Excel: {excel_path}")


def write_excel(results, rubrics, excel_path):
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "评测结果"

    headers = ["sample_idx", "data_item_id", "rubric_idx", "rubric_id", "rubric_tags",
               "criteria_met", "explanation", "error", "latency_s"]
    for col_idx, h in enumerate(headers):
        ws.cell(row=1, column=col_idx + 1).value = h

    for r in results:
        ws.append([r.get(h, "") for h in headers])

    # 列宽
    for col_idx in range(1, len(headers) + 1):
        ws.column_dimensions[openpyxl.utils.get_column_letter(col_idx + 1)].width = 20

    # 按 rubric 汇总 sheet
    ws2 = wb.create_sheet("按Rubric汇总")
    ws2.append(["rubric_idx", "rubric_id", "rubric_tags", "total", "true", "false", "null", "error"])
    for r_idx, rubric in enumerate(rubrics):
        subset = [r for r in results if r.get("rubric_idx") == r_idx]
        t = sum(1 for r in subset if r.get("criteria_met") is True)
        f = sum(1 for r in subset if r.get("criteria_met") is False)
        n = sum(1 for r in subset if r.get("criteria_met") is None)
        e = sum(1 for r in subset if r.get("error"))
        ws2.append([r_idx, rubric.get("tags", [""])[0] if rubric.get("tags") else "",
                    ",".join(rubric.get("tags", [])), len(subset), t, f, n, e])

    # 按样本汇总 sheet
    ws3 = wb.create_sheet("按样本汇总")
    ws3.append(["sample_idx", "data_item_id", "total", "true", "false", "null", "error"])
    sample_ids = sorted(set(r.get("sample_idx", -1) for r in results))
    for s_idx in sample_ids:
        subset = [r for r in results if r.get("sample_idx") == s_idx]
        did = subset[0].get("data_item_id", "") if subset else ""
        t = sum(1 for r in subset if r.get("criteria_met") is True)
        f = sum(1 for r in subset if r.get("criteria_met") is False)
        n = sum(1 for r in subset if r.get("criteria_met") is None)
        e = sum(1 for r in subset if r.get("error"))
        ws3.append([s_idx, did, len(subset), t, f, n, e])

    wb.save(excel_path)
    print(f"Excel 已保存: {excel_path}")


# ==================== 入口 ====================

def main():
    parser = argparse.ArgumentParser(description="Rubric Judge 测评框架")
    parser.add_argument("--limit-rows", type=int, default=0, help="只处理前 N 条模型回复")
    parser.add_argument("--limit-rubrics", type=int, default=0, help="只使用前 N 个 rubric")
    parser.add_argument("--no-excel", action="store_true", help="不输出 Excel")
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
