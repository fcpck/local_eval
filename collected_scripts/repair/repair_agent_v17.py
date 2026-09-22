#!/usr/bin/env python3
"""
Training Data JSONL Repair Agent - Multi-node concurrent v17
按 Group A -> B -> C -> D -> E 串行修复每条样本的管理方案 JSON
多节点并发处理不同样本
"""

import asyncio
import aiohttp
import json
import time
import os
import sys
import hashlib
import re
import argparse
from datetime import datetime

# ===== Configuration =====
INPUT_JSONL_PATH = r"<REPLACE_WITH_INPUT_JSONL_PATH>"
OUTPUT_DIR = r"<REPLACE_WITH_OUTPUT_DIR>"
RULES_FILE = r"<REPLACE_WITH_RULES_MD_PATH>"

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_JSONL_PATH = os.path.join(OUTPUT_DIR, f"训练集_260825_rag_p1_deepseek-v4-pro-0813_newrecipes_fixed_{TIMESTAMP}.jsonl")
LOG_PATH = os.path.join(OUTPUT_DIR, f"训练集_260825_rag_p1_deepseek-v4-pro-0813_newrecipes_fixed_log_{TIMESTAMP}.jsonl")
CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, f"训练集_260825_rag_p1_deepseek-v4-pro-0813_newrecipes_fixed_ckpt_{TIMESTAMP}.resume")

RESUME = True
PROGRESS_INTERVAL = 30
REQUEST_TIMEOUT = 360
MAX_RETRIES = 10
RETRY_BACKOFF = 5
MAX_TOKENS = 8192
LIMIT_ROWS = 0  # 0 = process all

LLM_NODES_CONFIG = [
    {"name": "NODE_1", "url": "http://<NODE_1_IP>:8080/v1/chat/completions", "model": "qwen36_35b", "max_concurrency": 40},
    {"name": "NODE_2", "url": "http://<NODE_2_IP>:8080/v1/chat/completions", "model": "qwen36_35b", "max_concurrency": 40},
]

GROUP_NAMES = ["Group A", "Group B", "Group C", "Group D", "Group E"]


# ===== Rules Loading =====
def load_group_rules(rules_file):
    with open(rules_file, "r", encoding="utf-8") as f:
        content = f.read()
    rules = {}
    for i, name in enumerate(GROUP_NAMES):
        header = f"## {name}"
        start = content.find(header)
        if start == -1:
            raise ValueError(f"Cannot find {header} in rules file")
        if i + 1 < len(GROUP_NAMES):
            next_header = f"## {GROUP_NAMES[i+1]}"
            end = content.find(next_header, start + 1)
            if end == -1:
                end = len(content)
        else:
            end = len(content)
        rules[name] = content[start:end].strip()
    return rules


# ===== JSON Pointer Utilities =====
def parse_pointer(ptr):
    if not ptr or ptr == "/":
        return []
    if not ptr.startswith("/"):
        raise ValueError(f"Invalid JSON Pointer: {ptr}")
    parts = ptr[1:].split("/")
    return [p.replace("~1", "/").replace("~0", "~") for p in parts]


def get_value(obj, ptr):
    parts = parse_pointer(ptr)
    cur = obj
    for p in parts:
        if isinstance(cur, list):
            idx = int(p)
            if idx < 0 or idx >= len(cur):
                raise KeyError(f"Array index {idx} out of range")
            cur = cur[idx]
        elif isinstance(cur, dict):
            if p not in cur:
                raise KeyError(f"Key '{p}' not found")
            cur = cur[p]
        else:
            raise KeyError(f"Cannot traverse into {type(cur)}")
    return cur


def apply_patch_op(obj, op, path, value=None):
    parts = parse_pointer(path)
    if not parts:
        return value if op in ("add", "replace") else None
    cur = obj
    for p in parts[:-1]:
        if isinstance(cur, list):
            cur = cur[int(p)]
        elif isinstance(cur, dict):
            cur = cur[p]
        else:
            raise KeyError(f"Cannot traverse {p} into {type(cur)}")
    last = parts[-1]
    if isinstance(cur, list):
        idx = int(last)
        if op == "add":
            cur.insert(idx, value)
        elif op == "replace":
            if idx < 0 or idx >= len(cur):
                raise IndexError(f"Array index {idx} out of range for replace")
            cur[idx] = value
        elif op == "remove":
            if idx < 0 or idx >= len(cur):
                raise IndexError(f"Array index {idx} out of range for remove")
            cur.pop(idx)
    elif isinstance(cur, dict):
        if op in ("add", "replace"):
            cur[last] = value
        elif op == "remove":
            cur.pop(last, None)
    return obj


# ===== Patch Validation =====
GROUP_FIELD_RULES = {
    "Group A": {
        "allowed": ["/userSituation", "/summary", "/guidelineText", "/managementGoals"],
        "forbidden": ["/dietaryAdvice", "/sportAdvice", "/dailyTasks"],
        "forbidden_suffixes": ["targetMinValue", "targetMaxValue", "targetValue", "targetMode"],
    },
    "Group B": {
        "allowed": ["/managementGoals"],
        "forbidden": ["/userSituation", "/summary", "/guidelineText", "/dietaryAdvice", "/sportAdvice", "/dailyTasks"],
    },
    "Group C": {
        "allowed": ["/dailyTasks"],
        "forbidden": ["/userSituation", "/summary", "/guidelineText", "/managementGoals", "/dietaryAdvice", "/sportAdvice"],
    },
    "Group D": {
        "allowed": ["/dietaryAdvice"],
        "forbidden": ["/userSituation", "/summary", "/guidelineText", "/managementGoals", "/dailyTasks", "/sportAdvice"],
    },
    "Group E": {
        "allowed": ["/sportAdvice"],
        "forbidden": ["/userSituation", "/summary", "/guidelineText", "/managementGoals", "/dailyTasks", "/dietaryAdvice"],
    },
}


def check_path(path, group):
    rules = GROUP_FIELD_RULES[group]
    for fp in rules.get("forbidden", []):
        if path == fp or path.startswith(fp + "/"):
            return False, f"forbidden: {fp}"
    for suffix in rules.get("forbidden_suffixes", []):
        if path.endswith("/" + suffix):
            return False, f"forbidden field: {suffix}"
    for ap in rules["allowed"]:
        if path == ap or path.startswith(ap + "/"):
            return True, ""
    return False, f"not in allowed set for {group}"


# ===== LLM Response Parser =====
def extract_json_array(text):
    text = text.strip()
    # Try direct parse
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result, None
        if isinstance(result, dict):
            plan_keys = ["userSituation", "managementGoals", "dietaryAdvice", "sportAdvice", "dailyTasks", "summary"]
            if any(k in result for k in plan_keys):
                return None, "Response appears to be full plan JSON, not a JSON Patch array"
            return None, "Response is a JSON object, not an array"
    except:
        pass
    # Try markdown code block
    m = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if m:
        try:
            result = json.loads(m.group(1))
            if isinstance(result, list):
                return result, None
        except:
            pass
    # Try finding [ ... ]
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        try:
            result = json.loads(text[start : end + 1])
            if isinstance(result, list):
                return result, None
        except:
            pass
    return None, f"Could not extract JSON array. Response preview: {text[:300]}"


# ===== LLM Node =====
class LLMNode:
    def __init__(self, config):
        self.name = config["name"]
        self.url = config["url"]
        self.model = config["model"]
        self.max_concurrency = config["max_concurrency"]
        self.semaphore = asyncio.Semaphore(self.max_concurrency)
        self.active = 0
        self.timeout_count = 0
        self.fail_count = 0
        self.consecutive_failures = 0
        self.total_requests = 0
        self.total_latency = 0.0
        self.priority = 1

    async def call(self, prompt, session):
        async with self.semaphore:
            self.active += 1
            self.total_requests += 1
            start_time = time.time()
            try:
                payload = {
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": MAX_TOKENS,
                    "temperature": 0.7,
                    "stream": False,
                }
                timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT, connect=30)
                async with session.post(self.url, json=payload, timeout=timeout) as resp:
                    if resp.status != 200:
                        error_text = await resp.text()
                        self.consecutive_failures += 1
                        self.fail_count += 1
                        return None, f"HTTP {resp.status}: {error_text[:200]}"
                    data = await resp.json()
                    choices = data.get("choices", [])
                    if not choices:
                        self.consecutive_failures += 1
                        self.fail_count += 1
                        return None, "No choices in response"
                    content = choices[0].get("message", {}).get("content", "")
                    if not content:
                        self.consecutive_failures += 1
                        self.fail_count += 1
                        return None, "Empty content in response"
                    finish_reason = choices[0].get("finish_reason", "")
                    if finish_reason == "length":
                        print(f"  ⚠️ {self.name}: response truncated (finish_reason=length)")
                    self.consecutive_failures = 0
                    self.total_latency += time.time() - start_time
                    return content, None
            except asyncio.TimeoutError:
                self.timeout_count += 1
                self.consecutive_failures += 1
                return None, f"Timeout after {REQUEST_TIMEOUT}s"
            except Exception as e:
                self.fail_count += 1
                self.consecutive_failures += 1
                return None, str(e)[:200]
            finally:
                self.active -= 1


# ===== Node Pool =====
class NodePool:
    def __init__(self, configs):
        self.nodes = [LLMNode(c) for c in configs]

    def select_node(self):
        available = [n for n in self.nodes if n.active < n.max_concurrency and n.consecutive_failures < 5]
        if not available:
            available = [n for n in self.nodes if n.active < n.max_concurrency]
        if not available:
            return None
        return min(available, key=lambda n: (n.priority, n.active))

    async def call_llm(self, prompt, session):
        last_error = None
        last_node = None
        for attempt in range(MAX_RETRIES):
            node = self.select_node()
            if node is None:
                await asyncio.sleep(1)
                continue
            result, error = await node.call(prompt, session)
            last_node = node.name
            if result is not None:
                if node.consecutive_failures == 0:
                    node.priority = 1
                return result, None, node.name
            last_error = error
            if node.consecutive_failures >= 3:
                print(f"⚠️ 节点 {node.name} 连续失败 {node.consecutive_failures} 次，已降低优先级")
                node.priority = 10
            await asyncio.sleep(RETRY_BACKOFF)
        return None, last_error, last_node


# ===== Build LLM Prompt =====
def build_prompt(group_name, group_rule, system_prompt, user_prompt, current_plan_json):
    # Truncate system_prompt if too long (keep first 15K chars)
    if len(system_prompt) > 15000:
        system_prompt = system_prompt[:15000] + "\n...(system_prompt 截断)..."

    prompt = f"""你是训练数据修复 Agent。你现在只执行【{group_name}】。

下面是本组完整规则。你必须严格遵守本组允许修改字段和禁止修改字段：

{group_rule}

下面是用户基本信息、体检信息和 system prompt：

System Prompt:
{system_prompt}

User Information:
{user_prompt}

下面是当前版本的管理方案 JSON。注意：这是经过前面 Group 修改后的当前版本，你必须在此基础上继续局部修复：

{current_plan_json}

任务：
请根据【{group_name}】的规则，只修改当前管理方案 JSON 中本组允许修改的字段。

强制要求：
- 只输出 JSON Patch 数组。
- 如果不需要修改，输出空数组 []。
- 不要输出解释性正文。
- 不要输出 Markdown 代码块。
- 不要输出完整 JSON。
- 不要修改本组禁止修改的字段。
- 不要删除原 JSON 中正确的信息。
- 不要引入用户输入中没有的指标、疾病、病史或数值。
- 每个 patch 项必须包含 op、path、value、reason、group。
- path 必须是 JSON Pointer 格式，例如 /userSituation 或 /managementGoals/0/metrics/1/displayText。
- op 只能使用 add、replace、remove。
- 如果需要修改数组中的对象，应使用准确的数组下标 path。

输出格式示例：
[
  {{
    "op": "replace",
    "path": "/userSituation",
    "value": "修改后的文本",
    "reason": "修正血脂分类名称，保留原有血压、血糖和家族史信息",
    "group": "{group_name}"
  }}
]"""
    return prompt


# ===== Process Sample =====
async def process_sample(line_index, record, group_rules, node_pool, session,
                          log_file, log_lock, checkpoint_path, ckpt_lock):
    sample_id = record.get("id", f"line_{line_index}")

    # Parse model_output
    try:
        mo_str = record.get("model_output", "")
        if not mo_str:
            return {"line_index": line_index, "status": "failed", "record": record,
                    "completed_groups": [], "updated_at": datetime.now().isoformat(),
                    "processed_by": "", "error": "model_output is empty"}
        plan_json = json.loads(mo_str)
    except (json.JSONDecodeError, TypeError) as e:
        return {"line_index": line_index, "status": "failed", "record": record,
                "completed_groups": [], "updated_at": datetime.now().isoformat(),
                "processed_by": "", "error": f"model_output parse error: {e}"}

    completed_groups = []
    processed_by_nodes = []
    first_prompt_size = None

    for group_name in GROUP_NAMES:
        group_rule = group_rules[group_name]
        system_prompt = record.get("system_prompt", "")
        user_prompt = record.get("user_prompt", "")
        current_plan_str = json.dumps(plan_json, ensure_ascii=False)
        input_hash = hashlib.md5(current_plan_str.encode("utf-8")).hexdigest()

        prompt = build_prompt(group_name, group_rule, system_prompt, user_prompt, current_plan_str)
        if first_prompt_size is None:
            first_prompt_size = len(prompt)

        llm_call_count = 0
        group_success = False
        last_error = None
        node_name = None
        patches = []

        for content_attempt in range(2):  # 2 content-layer retries
            result, error, node_name = await node_pool.call_llm(prompt, session)
            llm_call_count += 1
            if node_name and node_name not in processed_by_nodes:
                processed_by_nodes.append(node_name)

            if error:
                last_error = error
                continue

            patches, parse_error = extract_json_array(result)
            if parse_error:
                last_error = parse_error
                if content_attempt == 0:
                    # Content-layer retry with stricter prompt
                    prompt = f"""你刚才的输出格式错误，必须只输出 JSON Patch 数组，不要输出其他任何内容。

上次错误：{parse_error}

请重新根据当前 Group 规则生成 JSON Patch。

{group_rule}

当前管理方案 JSON：
{current_plan_str}

要求：
- 只输出 JSON Patch 数组。
- 如果不需要修改，输出 []。
- 不要输出 Markdown 代码块。
- 不要输出解释性正文。"""
                    continue
                else:
                    patches = []
                    break

            # Validate and apply patches
            valid_patches = []
            rejected_patches = []
            for patch in patches:
                if not isinstance(patch, dict):
                    rejected_patches.append({"patch": str(patch)[:100], "reason": "not a dict"})
                    continue
                # Auto-fill missing fields (LLM may omit reason/group)
                if "group" not in patch:
                    patch["group"] = group_name
                if "reason" not in patch:
                    patch["reason"] = ""
                # Only require op and path
                if not all(k in patch for k in ["op", "path"]):
                    rejected_patches.append({"patch": patch, "reason": "missing required fields (op/path)"})
                    continue
                if patch.get("group") != group_name:
                    rejected_patches.append({"patch": patch, "reason": f"group mismatch: {patch.get('group')} != {group_name}"})
                    continue
                if patch["op"] not in ("add", "replace", "remove"):
                    rejected_patches.append({"patch": patch, "reason": f"invalid op: {patch['op']}"})
                    continue
                if patch["op"] in ("add", "replace") and "value" not in patch:
                    rejected_patches.append({"patch": patch, "reason": "add/replace without value"})
                    continue
                allowed, reason = check_path(patch["path"], group_name)
                if not allowed:
                    rejected_patches.append({"patch": patch, "reason": f"forbidden path: {reason}"})
                    continue
                valid_patches.append(patch)

            # Apply valid patches
            applied_count = 0
            async with log_lock:
                for patch in valid_patches:
                    try:
                        before_value = None
                        try:
                            before_value = get_value(plan_json, patch["path"])
                        except:
                            pass

                        plan_json = apply_patch_op(plan_json, patch["op"], patch["path"], patch.get("value"))
                        applied_count += 1

                        after_value = None
                        try:
                            after_value = get_value(plan_json, patch["path"])
                        except:
                            pass

                        log_entry = {
                            "log_type": "patch_detail",
                            "line_index": line_index, "record_id": sample_id,
                            "group": group_name, "status": "applied",
                            "op": patch["op"], "path": patch["path"],
                            "reason": patch.get("reason", ""), "rule_source": group_name,
                            "before_value": before_value, "after_value": after_value,
                            "before_value_preview": str(before_value)[:300] if before_value else "",
                            "after_value_preview": str(after_value)[:300] if after_value else "",
                            "rejected_reason": "", "error": "",
                        }
                        log_file.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
                        log_file.flush()
                    except Exception as e:
                        log_entry = {
                            "log_type": "patch_detail",
                            "line_index": line_index, "record_id": sample_id,
                            "group": group_name, "status": "failed",
                            "op": patch["op"], "path": patch["path"],
                            "reason": patch.get("reason", ""),
                            "error": str(e)[:200],
                        }
                        log_file.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
                        log_file.flush()

                # Log rejected patches
                for rp in rejected_patches:
                    p = rp["patch"]
                    log_entry = {
                        "log_type": "patch_detail",
                        "line_index": line_index, "record_id": sample_id,
                        "group": group_name, "status": "rejected",
                        "op": p.get("op", "") if isinstance(p, dict) else "",
                        "path": p.get("path", "") if isinstance(p, dict) else "",
                        "reason": p.get("reason", "") if isinstance(p, dict) else "",
                        "rejected_reason": rp["reason"], "error": "",
                    }
                    log_file.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
                    log_file.flush()

            # Update plan string
            current_plan_str = json.dumps(plan_json, ensure_ascii=False)
            output_hash = hashlib.md5(current_plan_str.encode("utf-8")).hexdigest()

            # Log group summary
            async with log_lock:
                group_summary = {
                    "log_type": "group_summary",
                    "line_index": line_index, "record_id": sample_id,
                    "group": group_name,
                    "status": "success" if applied_count > 0 else "no_change",
                    "llm_call_count": llm_call_count,
                    "raw_patch_count": len(patches),
                    "patch_count": applied_count,
                    "paths": [p["path"] for p in valid_patches],
                    "rejected_paths": [rp["patch"].get("path", "") for rp in rejected_patches if isinstance(rp["patch"], dict)],
                    "error": "",
                    "group_input_plan_hash": input_hash,
                    "group_output_plan_hash": output_hash,
                    "processed_by": node_name or "",
                }
                log_file.write(json.dumps(group_summary, ensure_ascii=False) + "\n")
                log_file.flush()

            completed_groups.append(group_name)
            group_success = True
            break

        if not group_success:
            # Log error
            async with log_lock:
                group_summary = {
                    "log_type": "group_summary",
                    "line_index": line_index, "record_id": sample_id,
                    "group": group_name, "status": "llm_error",
                    "llm_call_count": llm_call_count,
                    "raw_patch_count": 0, "patch_count": 0,
                    "paths": [], "rejected_paths": [],
                    "error": last_error or "All retries failed",
                    "group_input_plan_hash": input_hash,
                    "group_output_plan_hash": input_hash,
                    "processed_by": node_name or "",
                }
                log_file.write(json.dumps(group_summary, ensure_ascii=False) + "\n")
                log_file.flush()
            completed_groups.append(f"{group_name}(FAILED)")

    # Write back model_output
    record["model_output"] = json.dumps(plan_json, ensure_ascii=False)

    has_failure = any("FAILED" in g for g in completed_groups)
    all_failed = all("FAILED" in g for g in completed_groups)
    if all_failed:
        status = "failed"
    elif has_failure:
        status = "partial"
    else:
        status = "completed"

    result = {
        "line_index": line_index,
        "status": status,
        "record": record,
        "completed_groups": completed_groups,
        "updated_at": datetime.now().isoformat(),
        "processed_by": ",".join(processed_by_nodes) if processed_by_nodes else "",
    }

    if first_prompt_size:
        result["prompt_size"] = first_prompt_size

    # Write checkpoint
    async with ckpt_lock:
        with open(checkpoint_path, "a", encoding="utf-8") as ckpt:
            ckpt.write(json.dumps(result, ensure_ascii=False) + "\n")
            ckpt.flush()

    return result


# ===== Progress Reporter =====
class ProgressReporter:
    def __init__(self, total, pool):
        self.total = total
        self.pool = pool
        self.completed = 0
        self.failed = 0
        self.start_time = time.time()
        self.last_report = time.time()
        self.current_tasks = {}

    def update(self, line_index, group):
        self.current_tasks[line_index] = group

    def remove(self, line_index):
        self.current_tasks.pop(line_index, None)

    def maybe_report(self):
        now = time.time()
        if now - self.last_report >= PROGRESS_INTERVAL:
            self.report()
            self.last_report = now

    def report(self):
        elapsed = time.time() - self.start_time
        print(f"\n{'='*60}")
        print(f"进度: total={self.total}, completed={self.completed}, failed={self.failed}, elapsed={elapsed:.0f}s")
        active = sum(n.active for n in self.pool.nodes)
        max_conc = sum(n.max_concurrency for n in self.pool.nodes)
        print(f"active_workers={active}/{max_conc}")
        for node in self.pool.nodes:
            avg = node.total_latency / max(node.total_requests, 1)
            print(f"  {node.name}: {node.active}/{node.max_concurrency}, timeout={node.timeout_count}, fail={node.fail_count}, avg={avg:.1f}s, total_req={node.total_requests}")
        if self.current_tasks:
            tasks_str = ", ".join(f"line {idx}: {grp}" for idx, grp in list(self.current_tasks.items())[:10])
            print(f"正在处理: {tasks_str}")
        print(f"checkpoint: {CHECKPOINT_PATH}")
        print(f"log: {LOG_PATH}")
        print(f"{'='*60}")
        sys.stdout.flush()


# ===== Main =====
async def main():
    global CHECKPOINT_PATH

    parser = argparse.ArgumentParser()
    parser.add_argument("--limit-rows", type=int, default=LIMIT_ROWS, help="Only process first N rows")
    parser.add_argument("--no-resume", action="store_true", help="Don't resume from checkpoint")
    args = parser.parse_args()

    print(f"=== 训练数据 JSONL 修复 Agent 启动 ===")
    print(f"时间: {TIMESTAMP}")
    print(f"输入: {INPUT_JSONL_PATH}")
    print(f"输出: {OUTPUT_JSONL_PATH}")
    print(f"日志: {LOG_PATH}")
    print(f"检查点: {CHECKPOINT_PATH}")
    print(f"节点: {[(n['name'], n['max_concurrency']) for n in LLM_NODES_CONFIG]}")
    print(f"全局并发: {sum(n['max_concurrency'] for n in LLM_NODES_CONFIG)}")
    print(f"max_tokens: {MAX_TOKENS}")

    # Load rules
    group_rules = load_group_rules(RULES_FILE)
    for name in GROUP_NAMES:
        print(f"  {name} 规则: {len(group_rules[name])} 字符")

    # Read input
    records = []
    with open(INPUT_JSONL_PATH, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))

    if args.limit_rows > 0:
        records = records[: args.limit_rows]
    print(f"读取 {len(records)} 条样本")

    # Resume from checkpoint
    checkpoint = {}
    do_resume = RESUME and not args.no_resume
    if do_resume:
        ckpt_path = CHECKPOINT_PATH
        if not os.path.exists(ckpt_path):
            ckpt_dir = os.path.dirname(ckpt_path)
            if os.path.exists(ckpt_dir):
                ckpt_files = [f for f in os.listdir(ckpt_dir) if f.endswith(".resume")]
                if ckpt_files:
                    latest = max(ckpt_files, key=lambda f: os.path.getmtime(os.path.join(ckpt_dir, f)))
                    ckpt_path = os.path.join(ckpt_dir, latest)
                    CHECKPOINT_PATH = ckpt_path
                    print(f"使用最近检查点: {ckpt_path}")

        if os.path.exists(ckpt_path):
            with open(ckpt_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                        checkpoint[entry["line_index"]] = entry
                    except:
                        pass
            print(f"检查点: {len(checkpoint)} 条已有记录")

    # Create node pool
    pool = NodePool(LLM_NODES_CONFIG)

    # Open log file
    log_file = open(LOG_PATH, "a", encoding="utf-8")
    log_lock = asyncio.Lock()
    ckpt_lock = asyncio.Lock()

    # Results dict (start with completed checkpoint entries)
    results = {}
    for idx, entry in checkpoint.items():
        if entry["status"] == "completed":
            results[idx] = entry

    # Determine pending samples
    pending = []
    for i in range(len(records)):
        if i in results:
            continue
        pending.append(i)

    print(f"待处理: {len(pending)} 条")

    # Progress reporter
    reporter = ProgressReporter(len(pending), pool)

    # Process concurrently
    connector = aiohttp.TCPConnector(limit=100, limit_per_host=50)
    async with aiohttp.ClientSession(connector=connector) as session:
        sem = asyncio.Semaphore(sum(n["max_concurrency"] for n in LLM_NODES_CONFIG))

        async def process_with_sem(idx):
            async with sem:
                reporter.update(idx, "Group A")
                try:
                    result = await process_sample(
                        idx, records[idx], group_rules, pool, session,
                        log_file, log_lock, CHECKPOINT_PATH, ckpt_lock
                    )
                except Exception as e:
                    result = {
                        "line_index": idx, "status": "failed", "record": records[idx],
                        "completed_groups": [], "updated_at": datetime.now().isoformat(),
                        "processed_by": "", "error": str(e)[:200],
                    }
                    async with ckpt_lock:
                        with open(CHECKPOINT_PATH, "a", encoding="utf-8") as ckpt:
                            ckpt.write(json.dumps(result, ensure_ascii=False) + "\n")
                            ckpt.flush()
                reporter.remove(idx)
                reporter.completed += 1
                if result["status"] == "failed":
                    reporter.failed += 1
                results[idx] = result
                # Print prompt size for first sample
                if "prompt_size" in result and reporter.completed == 1:
                    print(f"  首条 prompt 大小: {result['prompt_size']} 字符")
                reporter.maybe_report()

        # Start periodic reporter
        running = [True]

        async def periodic_report():
            while running[0]:
                await asyncio.sleep(PROGRESS_INTERVAL)
                if running[0]:
                    reporter.report()

        reporter_task = asyncio.create_task(periodic_report())

        # Create and run tasks
        tasks = [process_with_sem(idx) for idx in pending]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        # Stop periodic reporter
        running[0] = False
        reporter_task.cancel()
        try:
            await reporter_task
        except asyncio.CancelledError:
            pass

    reporter.report()

    # Write output
    print(f"\n写入输出文件...")
    with open(OUTPUT_JSONL_PATH, "w", encoding="utf-8") as f:
        for i in range(len(records)):
            if i in results and "record" in results[i]:
                f.write(json.dumps(results[i]["record"], ensure_ascii=False) + "\n")
            else:
                f.write(json.dumps(records[i], ensure_ascii=False) + "\n")

    log_file.close()

    # Final summary
    completed = sum(1 for r in results.values() if r["status"] == "completed")
    partial = sum(1 for r in results.values() if r["status"] == "partial")
    failed = sum(1 for r in results.values() if r["status"] == "failed")

    print(f"\n=== 完成 ===")
    print(f"总样本: {len(records)}")
    print(f"成功: {completed}, 部分成功: {partial}, 失败: {failed}")
    print(f"输出: {OUTPUT_JSONL_PATH}")
    print(f"日志: {LOG_PATH}")
    print(f"检查点: {CHECKPOINT_PATH}")

    if failed > 0 or partial > 0:
        print(f"\n失败/部分成功的样本:")
        for idx, r in sorted(results.items()):
            if r["status"] in ("failed", "partial"):
                print(f"  line {idx} ({r.get('record_id', '')}): {r['status']}, completed_groups={r.get('completed_groups', [])}")

    # Node statistics
    print(f"\n节点统计:")
    for node in pool.nodes:
        avg = node.total_latency / max(node.total_requests, 1)
        print(f"  {node.name}: requests={node.total_requests}, timeout={node.timeout_count}, fail={node.fail_count}, avg_latency={avg:.1f}s")


if __name__ == "__main__":
    asyncio.run(main())
