# -*- coding: utf-8 -*-
"""Rubric Judge 核心趟。

对每个 (模型 × 用例 × rubric) 构造 conversation → 调用 judge API（CSB 网关）→ 解析判分 → 落盘。

特性：
- 并发控制（judge.concurrency，建议 ≤10）
- 断点续跑：唯一键 model|case_id|rubric_id，已存在跳过
- 失败不中断：单条重试耗尽写 judge_failures/{model}.jsonl
- 提取降级：model_output 非合法 JSON 时降级为原文（extract_status 记录）
"""

import argparse
import asyncio
import json
import os
import time
from datetime import datetime

import aiohttp

from api_client import build_url, call_api_stream
from prompt_builder import extract_excerpt, build_conversation, render_judge_prompt, parse_judge_response
from result_store import load_jsonl, append_jsonl, existing_keys, format_key, rubric_id_of


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_run_tag(config):
    tag = config.get("run_tag", "auto")
    if tag == "auto":
        return datetime.now().strftime("%Y%m%d_%H%M%S")
    return tag


def load_rubrics(rubric_sets, rubric_source="all", limit_rubrics=0):
    """按 rubric_source 选择 rubric 集合。

    rubric_source ∈ {"all", "grading", "doctor_pref"}（或 rubric_sets 中的任意 key）。
    """
    if rubric_source == "all":
        paths = list(rubric_sets.values())
    elif rubric_source in rubric_sets:
        paths = [rubric_sets[rubric_source]]
    else:
        raise ValueError(f"未知 rubric_source: {rubric_source}（可选 all/{'/'.join(rubric_sets.keys())}）")
    items = []
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        items.extend(data["items"])
    if limit_rubrics:
        items = items[:limit_rubrics]
    return items


def load_judge_template(path):
    lines = load_jsonl(path)
    if not lines:
        raise ValueError(f"judge template empty: {path}")
    return lines[0]["content"]


def load_model_outputs(row, model):
    """从用例行取指定模型的 model_output 原始字符串；模型缺失返回 (None, err)。"""
    fmo = row.get("four_models_outputs") or {}
    entry = fmo.get(model)
    if entry is None:
        return None, f"model '{model}' not in four_models_outputs"
    mo = entry.get("model_output") if isinstance(entry, dict) else None
    if mo is None:
        return None, f"model_output missing for '{model}'"
    return mo, None


async def judge_one(session, sem, cfg, prompt, mock=None):
    """调用一次 judge API，返回 (content, err)。mock 不为 None 时返回假响应。

    cfg 支持两种端点格式：
      - 完整 URL: cfg["url"]（如 8088 /v2/infer/{id}/v1/chat/completions）+ cfg["bearer_token"]
      - infer_id: cfg["infer_id"]（8087 csb-inner-service）+ cfg["csb_token"]
    """
    if mock is not None:
        return '{"explanation": "mock 响应（离线验证）", "criteria_met": true, "evidence": "mock"}', None

    if cfg.get("url"):
        url = cfg["url"]
        headers = {"Content-Type": "application/json"}
        if cfg.get("bearer_token"):
            headers["Authorization"] = f"Bearer {cfg['bearer_token']}"
        elif cfg.get("csb_token"):
            headers["csb-token"] = cfg["csb_token"]
    else:
        url = build_url(cfg["infer_id"])
        headers = {
            "Content-Type": "application/json",
            "csb-token": cfg["csb_token"],
        }
    messages = [{"role": "user", "content": prompt}]
    stream = cfg.get("stream", True)
    async with sem:
        content, _ = await call_api_stream(
            session, url, headers, cfg["model_name"], messages,
            cfg.get("max_tokens", 8192), cfg.get("timeout", 300), cfg.get("max_retries", 3),
            stream=stream,
        )
    if content is None:
        return None, "judge API returned None (retries exhausted)"
    return content, None


async def run_judge(config, target_models, limit_rows=0, limit_rubrics=0, mock=None, rubric_source=None, answers=None):
    paths = config["paths"]
    jcfg = config["judge"]
    run_tag = resolve_run_tag(config)

    if rubric_source is None:
        rubric_source = config.get("rubric_source", "all")
    # answers: 自定义答案源文件（如爬取结果 data/crawled/result_X.jsonl）；默认用测试集
    rows = load_jsonl(answers) if answers else load_jsonl(paths["test_set"])
    if limit_rows:
        rows = rows[:limit_rows]
    rubrics = load_rubrics(paths["rubric_sets"], rubric_source, limit_rubrics)
    template = load_judge_template(paths["judge_template"])

    out_root = os.path.join(paths["out_dir"], "raw_judge", run_tag)
    fail_root = os.path.join(paths["out_dir"], "judge_failures", run_tag)
    os.makedirs(out_root, exist_ok=True)
    os.makedirs(fail_root, exist_ok=True)

    total_models = len(target_models)
    total_calls = len(rows) * len(rubrics) * total_models
    print(f"=== Rubric Judge ===")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"模型: {target_models}")
    print(f"用例: {len(rows)} × Rubrics: {len(rubrics)} × 模型: {total_models} = {total_calls} 次调用")
    print(f"并发: {jcfg.get('concurrency', 6)} | 输出: {out_root}")
    if mock is not None:
        print(f"[MOCK 模式] 使用本地假 judge 响应，不调真实 API")

    sem = asyncio.Semaphore(jcfg.get("concurrency", 6))
    connector = aiohttp.TCPConnector(limit=jcfg.get("concurrency", 6) * 2,
                                     limit_per_host=jcfg.get("concurrency", 6) * 2)

    async with aiohttp.ClientSession(connector=connector) as session:
        for model in target_models:
            out_f = os.path.join(out_root, f"{model}.jsonl")
            fail_f = os.path.join(fail_root, f"{model}.jsonl")
            done = existing_keys(out_f)

            pending = []
            for row in rows:
                case_id = row.get("id", "")
                user_prompt = row.get("user_prompt", "")
                mo, mo_err = load_model_outputs(row, model)
                for r_idx, rubric in enumerate(rubrics):
                    rid = rubric_id_of(rubric, r_idx)
                    key = format_key(model, case_id, rid)
                    if key in done:
                        continue
                    if mo_err:
                        pending.append({
                            "key": key, "model": model, "case_id": case_id, "rubric_id": rid,
                            "extract_status": "missing_model_output", "error": mo_err,
                            "prompt": "", "row": row, "rubric": rubric, "pre_error": True,
                            "excerpt_preview": "",
                            "rubric_criterion": (rubric.get("criterion") or "")[:60],
                        })
                        continue
                    excerpt, estatus = extract_excerpt(mo, rubric.get("extract_key_path", ""), fallback="raw")
                    conv = build_conversation(user_prompt, excerpt)
                    prompt = render_judge_prompt(template, conv, rubric)
                    pending.append({
                        "key": key, "model": model, "case_id": case_id, "rubric_id": rid,
                        "extract_status": estatus, "error": "", "prompt": prompt,
                        "row": row, "rubric": rubric, "pre_error": False,
                        "excerpt_preview": excerpt[:80],
                        "rubric_criterion": (rubric.get("criterion") or "")[:60],
                    })

            print(f"\n[{model}] 待判 {len(pending)} 条（跳过已完成 {len(done)}）")

            ok_count = 0
            fail_count = 0
            start = time.time()

            async def process(task):
                nonlocal ok_count, fail_count
                base = {
                    "key": task["key"], "model": model, "case_id": task["case_id"],
                    "rubric_id": task["rubric_id"], "extract_status": task["extract_status"],
                    "rubric_criterion": task.get("rubric_criterion", ""),
                    "excerpt_preview": task.get("excerpt_preview", ""),
                    "judge_prompt": task.get("prompt", ""),  # 完整 judge 输入
                }
                if task["pre_error"]:
                    rec = {
                        **base,
                        "criteria_met": None, "explanation": "", "evidence": "",
                        "raw_judge": "", "ok": False, "error": task["error"],
                    }
                    append_jsonl(fail_f, rec)
                    fail_count += 1
                    return rec

                content = None
                last_err = ""
                # 整条重试（解析失败也算）
                for attempt in range(1, jcfg.get("max_retries_total", 2) + 1):
                    content, err = await judge_one(session, sem, jcfg, task["prompt"], mock=mock)
                    if err:
                        last_err = err
                        continue
                    parsed = parse_judge_response(content)
                    if parsed["ok"]:
                        rec = {
                            **base,
                            "criteria_met": parsed["criteria_met"],
                            "explanation": parsed["explanation"],
                            "evidence": parsed["evidence"],
                            "raw_judge": parsed["raw"],  # 完整 judge 输出，不截断
                            "ok": True, "error": "",
                        }
                        append_jsonl(out_f, rec)
                        ok_count += 1
                        return rec
                    last_err = parsed["error"]

                rec = {
                    **base,
                    "criteria_met": None, "explanation": "", "evidence": "",
                    "raw_judge": content or "",  # 完整 judge 输出，不截断
                    "ok": False, "error": last_err or "unparsable",
                }
                append_jsonl(fail_f, rec)
                fail_count += 1
                return rec

            batch = []
            for task in pending:
                batch.append(process(task))
            for coro in asyncio.as_completed(batch):
                await coro
                if (ok_count + fail_count) % 50 == 0:
                    el = time.time() - start
                    print(f"  [{model}] 完成 {ok_count + fail_count}/{len(pending)} (成功 {ok_count}, 失败 {fail_count}), 耗时 {el:.0f}s")

            print(f"[{model}] 完成：成功 {ok_count}，失败 {fail_count}，耗时 {time.time() - start:.1f}s")
            print(f"  结果: {out_f}")
            if fail_count:
                print(f"  失败: {fail_f}")

    print(f"\n=== 全部完成，run_tag={run_tag} ===")


def main():
    parser = argparse.ArgumentParser(description="Rubric Judge 核心趟")
    parser.add_argument("--config", default=r"D:\PycharmProject\git\Health_Management_Diabetes\local_eval\config.json")
    parser.add_argument("--model", default="", help="单个模型备注名")
    parser.add_argument("--all-models", action="store_true", help="对所有有输出的模型")
    parser.add_argument("--limit-rows", type=int, default=0, help="只处理前 N 条用例")
    parser.add_argument("--limit-rubrics", type=int, default=0, help="只用前 N 条 rubric")
    parser.add_argument("--mock", action="store_true", help="离线 mock 模式（不调真实 API）")
    parser.add_argument("--rubrics", default="", choices=["", "all", "grading", "doctor_pref"],
                        help="rubric 集合：all / grading / doctor_pref；默认读 config.rubric_source")
    parser.add_argument("--answers", default="", help="自定义答案源 JSONL（如爬取结果）；默认用测试集")
    args = parser.parse_args()

    config = load_config(args.config)
    rows = load_jsonl(args.answers) if args.answers else load_jsonl(config["paths"]["test_set"])
    all_models = []
    for row in rows:
        all_models.extend((row.get("four_models_outputs") or {}).keys())
    all_models = sorted(set(all_models))

    if args.all_models:
        target = all_models
    elif args.model:
        target = [args.model]
    else:
        print(f"可用模型: {all_models}")
        parser.error("请指定 --model X 或 --all-models")

    missing = [m for m in target if m not in all_models]
    if missing:
        print(f"[WARN] 以下模型在测试集中无输出: {missing}")
        target = [m for m in target if m in all_models]
        if not target:
            return

    rubric_source = args.rubrics or config.get("rubric_source", "all")

    if args.mock:
        asyncio.run(run_judge(config, target, args.limit_rows, args.limit_rubrics, mock=True, rubric_source=rubric_source, answers=args.answers or None))
    else:
        jcfg = config["judge"]
        has_endpoint = bool(jcfg.get("url") or jcfg.get("infer_id"))
        has_auth = bool(jcfg.get("bearer_token") or jcfg.get("csb_token"))
        if not has_endpoint or not jcfg.get("model_name") or not has_auth:
            print("[ERROR] config.judge 未完整配置：需要 url(或 infer_id) + model_name + bearer_token(或 csb_token)。")
            print("       请先填写，或使用 --mock 进行离线验证。")
            return
        asyncio.run(run_judge(config, target, args.limit_rows, args.limit_rubrics, mock=None, rubric_source=rubric_source, answers=args.answers or None))


if __name__ == "__main__":
    main()
