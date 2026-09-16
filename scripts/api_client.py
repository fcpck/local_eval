# -*- coding: utf-8 -*-
"""CSB 网关调用封装（流式 + 非流式）。

- 流式（stream=True）：解析 SSE data: 行，区分 content / reasoning_content
  （8087 csb-inner-service 非流式有 180s 超时限制，长回复必须流式）
- 非流式（stream=False）：直接解析 JSON 响应（8088 /v2/infer 网关短回复适用）
- 指数退避重试（1s, 2s, 4s ... 上限 30s）
"""

import asyncio
import json

import aiohttp

# 8087 csb-inner-service 格式（crawl 用，csb-token 鉴权）
DEFAULT_HEADERS = {
    "Content-Type": "application/json",
    "csb-token": "a79011dc-ea86-4dee-ba25-98729590d77c",
}

URL_TEMPLATE = "http://onlineservice-green.cn-east-4.roma.huawei.com:8087/csb-inner-service/rest/infers/{infer_id}?path=/v1/chat/completions"


def build_url(infer_id: str) -> str:
    """按 infer_id 构造 8087 CSB 推理服务 URL。"""
    return URL_TEMPLATE.format(infer_id=infer_id)


async def call_api_stream(
    session: aiohttp.ClientSession,
    url: str,
    headers: dict,
    model_name: str,
    messages: list,
    max_tokens: int,
    timeout: int,
    max_retries: int,
    stream: bool = True,
) -> tuple:
    """调用 chat/completions，返回 (content, reasoning_content)。

    - messages: OpenAI 格式消息列表（crawl 传 [system, user]，judge 传 [user]）
    - stream=True: SSE 流式解析；stream=False: JSON 响应解析
    - 重试耗尽返回 (None, None)
    """
    payload = {
        "model": model_name,
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": stream,
    }

    for attempt in range(1, max_retries + 1):
        content_parts = []
        reasoning_parts = []
        try:
            async with session.post(
                url, json=payload, headers=headers,
                timeout=aiohttp.ClientTimeout(total=timeout),
            ) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    print(f"\n[HTTP {resp.status}] attempt {attempt}/{max_retries}: {text[:200]}")
                    if attempt < max_retries:
                        await asyncio.sleep(min(2 ** attempt, 30))
                    continue

                if not stream:
                    # 非流式：解析 JSON 响应
                    try:
                        data = await resp.json()
                    except Exception:
                        text = await resp.text()
                        print(f"\n[JSON parse error] attempt {attempt}/{max_retries}: {text[:200]}")
                        if attempt < max_retries:
                            await asyncio.sleep(min(2 ** attempt, 30))
                        continue
                    choices = data.get("choices", [])
                    if not choices:
                        print(f"\n[No choices] attempt {attempt}/{max_retries}")
                        if attempt < max_retries:
                            await asyncio.sleep(min(2 ** attempt, 30))
                        continue
                    message = choices[0].get("message", {})
                    content = message.get("content") or ""
                    reasoning = message.get("reasoning_content") or ""
                    return content, reasoning

                async for raw_line in resp.content:
                    line = raw_line.decode("utf-8", errors="ignore").strip()
                    if not line or not line.startswith("data:"):
                        continue
                    data_str = line[len("data:"):].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue
                    choices = chunk.get("choices", [])
                    if not choices:
                        continue
                    delta = choices[0].get("delta", {})
                    if delta.get("reasoning_content"):
                        reasoning_parts.append(delta["reasoning_content"])
                    if delta.get("content"):
                        content_parts.append(delta["content"])

            return "".join(content_parts), "".join(reasoning_parts)

        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            print(f"\n[Error] attempt {attempt}/{max_retries}: {type(e).__name__}: {e}")
        if attempt < max_retries:
            await asyncio.sleep(min(2 ** attempt, 30))

    return None, None
