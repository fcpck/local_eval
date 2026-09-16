# -*- coding: utf-8 -*-
"""爬取新模型回答。

对 data/crawl_dataset.jsonl 逐条请求待测模型接口（8088 /v2/infer 网关，Bearer 鉴权，非流式），
写回 {id, ..., four_models_outputs: {备注名: {model_output, reasoning_parts}}}。

端点格式（config.crawl）：
  - url: 完整推理 URL（8088 /v2/infer/{id}/v1/chat/completions）+ bearer_token
  - 或 infer_id（8087 csb-inner-service）+ csb_token
"""

import argparse
import asyncio
import json
import os

from api_client import build_url, call_api_stream
from result_store import load_jsonl


def _endpoint(cfg):
    """返回 (url, headers, stream)。支持 8088 url+bearer 或 8087 infer_id+csb_token。"""
    if cfg.get("url"):
        headers = {"Content-Type": "application/json"}
        if cfg.get("bearer_token"):
            headers["Authorization"] = f"Bearer {cfg['bearer_token']}"
        elif cfg.get("csb_token"):
            headers["csb-token"] = cfg["csb_token"]
        return cfg["url"], headers, cfg.get("stream", False)
    headers = {
        "Content-Type": "application/json",
        "csb-token": cfg["csb_token"],
    }
    return build_url(cfg["infer_id"]), headers, cfg.get("stream", True)


async def crawl_one_model(session, cfg, dataset, model_name, note_name, out_path):
    url, headers, stream = _endpoint(cfg)
    sem = asyncio.Semaphore(cfg.get("concurrency", 30))

    existing = load_jsonl(out_path)
    existing_ids = {r.get("id") for r in existing}
    print(f"[{note_name}] 已有 {len(existing_ids)} 条，待爬 {len(dataset) - len(existing_ids)} 条，并发 {cfg.get('concurrency', 30)}")

    done = 0
    fail = 0

    async def process(row):
        nonlocal done, fail
        rid = row.get("id", "")
        if rid in existing_ids:
            return
        async with sem:
            messages = []
            if row.get("system_prompt"):
                messages.append({"role": "system", "content": row["system_prompt"]})
            messages.append({"role": "user", "content": row.get("user_prompt", "")})
            content, reasoning = await call_api_stream(
                session, url, headers, model_name, messages,
                cfg.get("max_tokens", 65536), cfg.get("timeout", 600), cfg.get("max_retries", 3),
                stream=stream,
            )
        if content is None:
            print(f"[FAIL] {rid}: 重试耗尽")
            fail += 1
            return
        out_row = dict(row)
        out_row.setdefault("four_models_outputs", {})[note_name] = {
            "model_output": content,
            "reasoning_parts": reasoning or "",
        }
        with open(out_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(out_row, ensure_ascii=False) + "\n")
        done += 1
        if done % cfg.get("save_interval", 10) == 0:
            print(f"  [{note_name}] 进度: {done}")

    await asyncio.gather(*(process(row) for row in dataset))
    return done, fail


def main():
    parser = argparse.ArgumentParser(description="爬取新模型回答")
    parser.add_argument("--config", default=r"D:\PycharmProject\git\Health_Management_Diabetes\local_eval\config.json")
    parser.add_argument("--model", default="", help="要爬取的模型备注名（必须在 config.crawl.models_to_crawl 中）")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)
    paths = config["paths"]
    crawl_cfg = config["crawl"]

    has_endpoint = bool(crawl_cfg.get("url") or crawl_cfg.get("infer_id"))
    if not has_endpoint or not crawl_cfg.get("model_name"):
        print("[ERROR] config.crawl 未完整配置：需要 url(或 infer_id) + model_name。")
        return

    models_to_crawl = crawl_cfg.get("models_to_crawl") or []
    if not models_to_crawl:
        print("[INFO] config.crawl.models_to_crawl 为空，爬取入口未启用。")
        return

    if args.model:
        if args.model not in models_to_crawl:
            print(f"[ERROR] {args.model} 不在 models_to_crawl 中: {models_to_crawl}")
            return
        models = [(args.model, args.model)]
    else:
        models = [(m, m) for m in models_to_crawl]
        print(f"[WARN] 未指定 --model，将爬取全部: {models_to_crawl}")

    dataset_path = os.path.join(paths["data_dir"], "crawl_dataset.jsonl")
    if not os.path.exists(dataset_path):
        print(f"[ERROR] 爬取用数据集不存在: {dataset_path}，请先运行 build")
        return
    dataset = load_jsonl(dataset_path)
    print(f"加载 {len(dataset)} 条用例")

    crawled_dir = os.path.join(paths["data_dir"], "crawled")
    os.makedirs(crawled_dir, exist_ok=True)

    async def run_all():
        import aiohttp
        concurrency = crawl_cfg.get("concurrency", 30)
        connector = aiohttp.TCPConnector(limit=concurrency, limit_per_host=concurrency)
        async with aiohttp.ClientSession(connector=connector) as session:
            for model_name, note in models:
                out_path = os.path.join(crawled_dir, f"result_{note}.jsonl")
                print(f"\n爬取 [{note}] (model={model_name}) ...")
                n, f_ = await crawl_one_model(session, crawl_cfg, dataset, model_name, note, out_path)
                print(f"[{note}] 完成 {n} 条，失败 {f_} 条 -> {out_path}")

    asyncio.run(run_all())


if __name__ == "__main__":
    main()
