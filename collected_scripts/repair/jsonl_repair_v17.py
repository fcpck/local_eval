import asyncio
import aiohttp
import json
import os
import sys
import time
import hashlib
import copy
import traceback
from datetime import datetime
from typing import Optional

INPUT_JSONL = r"<REPLACE_WITH_INPUT_JSONL_PATH>"
RULE_FILE = r"<REPLACE_WITH_RULES_MD_PATH>"

OUTPUT_DIR = r"<REPLACE_WITH_OUTPUT_DIR>"
NOW_STR = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_JSONL = os.path.join(OUTPUT_DIR, f"训练集_260825_rag_p1_deepseek-v4-pro-0813_newrecipes_fixed_{NOW_STR}.jsonl")
LOG_PATH = os.path.join(OUTPUT_DIR, f"训练集_260825_rag_p1_deepseek-v4-pro-0813_newrecipes_fixed_log_{NOW_STR}.txt")
CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, f"训练集_260825_rag_p1_deepseek-v4-pro-0813_newrecipes_fixed_ckpt_{NOW_STR}.resume")

RESUME = True
PROGRESS_INTERVAL = 30
REQUEST_TIMEOUT = 360
MAX_RETRIES = 10
RETRY_BACKOFF = 5

LLM_NODES = [
    {
        "name": "云道4卡1",
        "url": "http://onlineservice-green.cn-east-4.roma.huawei.com:8088/v2/infer/06d0324b-a095-403a-b621-8251b5d9b1f4/v1/chat/completions",
        "model": "qwen36_35b",
        "max_concurrency": 15,
        "request_timeout_seconds": 600,
        "max_retries": 10,
        "retry_backoff_seconds": 5,
        "auth_token": "REPLACE_WITH_BEARER_TOKEN",
    },
    {
        "name": "云道8卡1",
        "url": "http://onlineservice-green.cn-east-4.roma.huawei.com:8088/v2/infer/c89f2d2d-80c1-4aec-8b0f-996da5bda678/v1/chat/completions",
        "model": "qwen36_35b",
        "max_concurrency": 30,
        "request_timeout_seconds": 600,
        "max_retries": 10,
        "retry_backoff_seconds": 5,
        "auth_token": "REPLACE_WITH_BEARER_TOKEN",
    },
]

GROUPS = ["Group A", "Group B", "Group C", "Group D", "Group E"]

GROUP_RULES = {}

GROUP_ALLOWED_FIELDS = {
    "Group A": {
        "allow_prefixes": [
            "/userSituation", "/summary",
            "/managementGoals", "/managementGoals/",
        ],
        "allow_keywords": ["goalDesc", "displayText"],
        "deny_prefixes": [
            "/dietaryAdvice",
            "/sportAdvice",
        ],
        "deny_field_keywords": [
            "frequencyUnitText", "frequencyValue", "frequencyDesc",
            "targetMinValue", "targetMaxValue", "targetValue", "targetMode",
        ],
    },
    "Group B": {
        "allow_prefixes": [
            "/managementGoals", "/managementGoals/",
            "/summary", "/userSituation",
        ],
        "allow_keywords": ["goalDesc", "displayText", "targetMode", "targetMinValue", "targetMaxValue", "targetValue", "metrics"],
        "deny_prefixes": [
            "/dietaryAdvice",
            "/sportAdvice",
        ],
        "deny_field_keywords": [
            "frequencyUnitText", "frequencyValue", "frequencyDesc",
        ],
    },
    "Group C": {
        "allow_prefixes": [
            "/dailyTasks", "/dailyTasks/",
        ],
        "allow_keywords": ["监测", "monitor", "frequency", "taskName", "taskDesc"],
        "deny_prefixes": [
            "/userSituation", "/summary",
            "/managementGoals", "/managementGoals/",
            "/dietaryAdvice",
            "/sportAdvice",
        ],
        "deny_field_keywords": [],
    },
    "Group D": {
        "allow_prefixes": [
            "/dietaryAdvice", "/dietaryAdvice/",
            "/dailyTasks", "/dailyTasks/",
        ],
        "allow_keywords": ["diet", "nutrition", "recipe", "meal", "food", "energy", "weight"],
        "deny_prefixes": [
            "/userSituation", "/summary",
            "/managementGoals", "/managementGoals/",
            "/sportAdvice",
        ],
        "deny_field_keywords": [
            "targetMinValue", "targetMaxValue", "targetValue", "targetMode",
        ],
    },
    "Group E": {
        "allow_prefixes": [
            "/dailyTasks", "/dailyTasks/",
            "/sportAdvice", "/sportAdvice/",
        ],
        "allow_keywords": ["运动", "sport", "exercise", "taskName", "taskDesc", "items", "frequency"],
        "deny_prefixes": [
            "/userSituation", "/summary",
            "/managementGoals", "/managementGoals/",
            "/dietaryAdvice",
        ],
        "deny_field_keywords": [
            "targetMinValue", "targetMaxValue", "targetValue", "targetMode",
        ],
    },
}


def load_rule_file():
    global GROUP_RULES
    with open(RULE_FILE, "r", encoding="utf-8") as f:
        content = f.read()

    group_headers = {
        "Group A": "## Group A：总体结论与疾病分类口径",
        "Group B": "## Group B：管理目标",
        "Group C": "## Group C：监测任务频次",
        "Group D": "## Group D：饮食营养处方",
        "Group E": "## Group E：运动处方",
    }

    positions = {}
    for gname, header in group_headers.items():
        idx = content.find(header)
        if idx == -1:
            raise ValueError(f"Cannot find header: {header}")
        positions[gname] = idx

    sorted_groups = sorted(positions.items(), key=lambda x: x[1])
    for i, (gname, start) in enumerate(sorted_groups):
        if i + 1 < len(sorted_groups):
            end = sorted_groups[i + 1][1]
        else:
            end = len(content)
        GROUP_RULES[gname] = content[start:end].strip()

    print(f"[INIT] Loaded {len(GROUP_RULES)} group rules from rule file.")
    for g in GROUPS:
        print(f"  {g}: {len(GROUP_RULES[g])} chars")


def load_input_jsonl():
    records = []
    with open(INPUT_JSONL, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                records.append((line_idx, obj))
            except json.JSONDecodeError as e:
                print(f"[WARN] Line {line_idx} JSON parse error: {e}")
    print(f"[INIT] Loaded {len(records)} records from input JSONL.")
    return records


def load_checkpoint():
    ckpt = {}
    if RESUME and os.path.exists(CHECKPOINT_PATH):
        with open(CHECKPOINT_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    ckpt[obj["line_index"]] = obj
                except Exception:
                    pass
        print(f"[RESUME] Loaded checkpoint with {len(ckpt)} entries.")
    return ckpt


def save_checkpoint_entry(line_index, status, record, completed_groups, llm_node_used):
    entry = {
        "line_index": line_index,
        "status": status,
        "record": record,
        "completed_groups": completed_groups,
        "updated_at": datetime.now().isoformat(),
        "llm_node_used": llm_node_used,
    }
    with open(CHECKPOINT_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return entry


def write_log(log_entry):
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")


def json_hash(obj):
    return hashlib.md5(json.dumps(obj, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def resolve_json_pointer(obj, pointer):
    parts = pointer.strip("/").split("/")
    current = obj
    for part in parts:
        if isinstance(current, list):
            try:
                if part == "-":
                    return None, False
                idx = int(part)
                current = current[idx]
            except (ValueError, IndexError):
                return None, False
        elif isinstance(current, dict):
            if part in current:
                current = current[part]
            else:
                return None, False
        else:
            return None, False
    return current, True


def set_json_pointer(obj, pointer, value):
    parts = pointer.strip("/").split("/")
    current = obj
    for i, part in enumerate(parts[:-1]):
        if isinstance(current, list):
            try:
                if part == "-":
                    idx = len(current) - 1
                else:
                    idx = int(part)
                current = current[idx]
            except (ValueError, IndexError):
                return False
        elif isinstance(current, dict):
            if part not in current:
                current[part] = {}
            current = current[part]
        else:
            return False
    last = parts[-1]
    if isinstance(current, list):
        try:
            if last == "-":
                idx = len(current)
                current.append(value)
                return True
            idx = int(last)
            if idx < len(current):
                current[idx] = value
            else:
                return False
        except ValueError:
            return False
    elif isinstance(current, dict):
        current[last] = value
    else:
        return False
    return True


def apply_patch(obj, patch_item):
    op = patch_item.get("op")
    path = patch_item.get("path", "")
    value = patch_item.get("value")

    if op == "replace":
        old_val, exists = resolve_json_pointer(obj, path)
        if not exists:
            return False, old_val, None
        set_json_pointer(obj, path, value)
        return True, old_val, value
    elif op == "add":
        old_val, exists = resolve_json_pointer(obj, path)
        set_json_pointer(obj, path, value)
        return True, old_val if exists else None, value
    elif op == "remove":
        old_val, exists = resolve_json_pointer(obj, path)
        if not exists:
            return False, None, None
        parts = path.strip("/").split("/")
        current = obj
        for part in parts[:-1]:
            if isinstance(current, list):
                if part == "-":
                    current = current[-1]
                else:
                    current = current[int(part)]
            else:
                current = current[part]
        last = parts[-1]
        if isinstance(current, dict):
            del current[last]
        elif isinstance(current, list):
            if last == "-":
                current.pop(-1)
            else:
                current.pop(int(last))
        return True, old_val, None
    return False, None, None


def check_patch_allowed(path, group):
    rules = GROUP_ALLOWED_FIELDS.get(group, {})
    deny_prefixes = rules.get("deny_prefixes", [])
    for dp in deny_prefixes:
        if path == dp or path.startswith(dp + "/"):
            return False, f"path {path} is in deny list for {group}"

    deny_field_keywords = rules.get("deny_field_keywords", [])
    path_lower = path.lower()
    for kw in deny_field_keywords:
        if kw.lower() in path_lower:
            if group == "Group A":
                allow_in_context = False
                for ap in rules.get("allow_prefixes", []):
                    if path == ap or path.startswith(ap.rstrip("/") + "/"):
                        if kw.lower() in ["goaldesc", "displaytext"]:
                            allow_in_context = True
                            break
                if not allow_in_context:
                    return False, f"path {path} contains denied field keyword '{kw}' for {group}"
            else:
                return False, f"path {path} contains denied field keyword '{kw}' for {group}"

    return True, ""


def parse_llm_json_patch(text):
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        start = 0
        end = len(lines)
        for i, line in enumerate(lines):
            if line.strip().startswith("```") and i == 0:
                start = 1
            elif line.strip().startswith("```"):
                end = i
                break
        text = "\n".join(lines[start:end]).strip()

    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result
        elif isinstance(result, dict):
            return [result]
        return None
    except json.JSONDecodeError:
        import re
        match = re.search(r'\[[\s\S]*\]', text)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        return None


def build_prompt(group_name, user_info_and_system, current_plan_json):
    group_rule_text = GROUP_RULES.get(group_name, "")
    current_plan_str = json.dumps(current_plan_json, ensure_ascii=False, indent=2)

    prompt = f"""你是训练数据修复 Agent。你现在只执行【{group_name}】。

下面是本组完整规则。你必须严格遵守本组允许修改字段和禁止修改字段：

{group_rule_text}

下面是用户基本信息、体检信息和 system prompt：

{user_info_and_system}

下面是当前版本的管理方案 JSON。注意：这是经过前面 Group 修改后的当前版本，你必须在此基础上继续局部修复：

{current_plan_str}

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


class NodeTracker:
    def __init__(self, node_config):
        self.name = node_config["name"]
        self.url = node_config["url"]
        self.model = node_config["model"]
        self.max_concurrency = node_config["max_concurrency"]
        self.timeout = node_config.get("request_timeout_seconds", REQUEST_TIMEOUT)
        self.max_retries = node_config.get("max_retries", MAX_RETRIES)
        self.backoff = node_config.get("retry_backoff_seconds", RETRY_BACKOFF)
        self.auth_token = node_config.get("auth_token", None)
        self.active = 0
        self.total_failures = 0
        self.total_timeouts = 0
        self.consecutive_failures = 0
        self.consecutive_timeouts = 0
        self.recent_times = []
        self.degraded = False

    def record_success(self, elapsed):
        self.active -= 1
        self.consecutive_failures = 0
        self.consecutive_timeouts = 0
        self.recent_times.append(elapsed)
        if len(self.recent_times) > 20:
            self.recent_times = self.recent_times[-20:]
        if self.degraded and self.consecutive_failures == 0:
            self.degraded = False

    def record_failure(self, is_timeout=False):
        self.active -= 1
        self.total_failures += 1
        self.consecutive_failures += 1
        if is_timeout:
            self.total_timeouts += 1
            self.consecutive_timeouts += 1
        else:
            self.consecutive_timeouts = 0
        if self.consecutive_failures >= 3 or self.consecutive_timeouts >= 3:
            self.degraded = True

    def has_capacity(self):
        return self.active < self.max_concurrency and not self.degraded

    def avg_time(self):
        if not self.recent_times:
            return 0
        return sum(self.recent_times) / len(self.recent_times)


class RepairEngine:
    def __init__(self):
        self.nodes = [NodeTracker(n) for n in LLM_NODES]
        self.semaphore = asyncio.Semaphore(sum(n.max_concurrency for n in self.nodes))
        self.node_semaphores = {n.name: asyncio.Semaphore(n.max_concurrency) for n in self.nodes}
        self.records = []
        self.checkpoint = {}
        self.results = {}
        self.completed = 0
        self.failed = 0
        self.in_progress = {}
        self.start_time = time.time()
        self.log_file = None
        self.failed_samples = []
        self.sample_llm_nodes = {}

    def select_node(self):
        best = None
        best_active = float('inf')
        for node in self.nodes:
            if node.has_capacity() and node.active < best_active:
                best = node
                best_active = node.active
        if best is None:
            for node in self.nodes:
                if node.active < node.max_concurrency:
                    return node
        return best

    async def call_llm(self, prompt, line_index, group_name):
        node = self.select_node()
        if node is None:
            await asyncio.sleep(1)
            node = self.select_node()
            if node is None:
                raise Exception("No available LLM node")

        node_name = node.name
        last_error = None

        for attempt in range(node.max_retries):
            if not node.has_capacity() and node.degraded:
                alt = self.select_node()
                if alt and alt.name != node.name:
                    node = alt
                    node_name = node.name

            async with self.node_semaphores[node.name]:
                node.active += 1
                t0 = time.time()
                try:
                    timeout = aiohttp.ClientTimeout(total=node.timeout)
                    headers = {"Content-Type": "application/json"}
                    if node.auth_token:
                        headers["Authorization"] = f"Bearer {node.auth_token}"
                    async with aiohttp.ClientSession(timeout=timeout) as session:
                        payload = {
                            "model": node.model,
                            "messages": [{"role": "user", "content": prompt}],
                            "max_tokens": 30000,
                            "temperature": 0.7,
                            "stream": False,
                        }
                        async with session.post(node.url, json=payload, headers=headers) as resp:
                            elapsed = time.time() - t0
                            if resp.status == 200:
                                data = await resp.json()
                                content = data["choices"][0]["message"]["content"]
                                node.record_success(elapsed)
                                return content, node_name, elapsed
                            else:
                                text = await resp.text()
                                node.record_failure(is_timeout=False)
                                last_error = f"HTTP {resp.status}: {text[:200]}"
                                print(f"  [LLM ERR] node={node_name} line={line_index} {group_name} attempt={attempt+1} {last_error}")
                            await asyncio.sleep(node.backoff)
                except asyncio.TimeoutError:
                    elapsed = time.time() - t0
                    node.record_failure(is_timeout=True)
                    last_error = f"Timeout after {elapsed:.1f}s"
                    print(f"  [LLM TIMEOUT] node={node_name} line={line_index} {group_name} attempt={attempt+1} {last_error}")
                    if node.consecutive_timeouts >= 3:
                        print(f"\n*** ALERT: Node '{node_name}' has {node.consecutive_timeouts} consecutive timeouts! Please check this node. ***\n")
                    await asyncio.sleep(node.backoff)
                except Exception as e:
                    elapsed = time.time() - t0
                    node.record_failure(is_timeout=False)
                    last_error = str(e)
                    print(f"  [LLM ERR] node={node_name} line={line_index} {group_name} attempt={attempt+1} {last_error}")
                    await asyncio.sleep(node.backoff)

            if attempt < node.max_retries - 1:
                alt = self.select_node()
                if alt and alt.name != node.name:
                    node = alt
                    node_name = node.name

        raise Exception(f"LLM call failed after {node.max_retries} retries: {last_error}")

    async def process_group(self, line_index, record_id, group_name, plan_json, user_info_and_system, llm_node_used_set):
        input_hash = json_hash(plan_json)
        prompt = build_prompt(group_name, user_info_and_system, plan_json)

        llm_call_count = 0
        patches_applied = 0
        patches_rejected = 0
        patch_details = []
        status = "no_change"
        error_msg = ""

        try:
            content, node_name, elapsed = await self.call_llm(prompt, line_index, group_name)
            llm_call_count += 1
            llm_node_used_set.add(node_name)

            patches = parse_llm_json_patch(content)
            if patches is None:
                status = "patch_error"
                error_msg = f"LLM returned invalid JSON patch: {content[:200]}"
                content2, node_name2, elapsed2 = await self.call_llm(
                    prompt + "\n\n[IMPORTANT] Your last output was not valid JSON Patch. You MUST output ONLY a JSON Patch array. Do NOT output any explanation or markdown. Just output the JSON array.",
                    line_index, group_name
                )
                llm_call_count += 1
                llm_node_used_set.add(node_name2)
                patches = parse_llm_json_patch(content2)
                if patches is None:
                    error_msg = f"LLM retry also returned invalid JSON: {content2[:200]}"
                    return plan_json, {
                        "status": status,
                        "llm_call_count": llm_call_count,
                        "raw_patch_count": 0,
                        "patch_count": 0,
                        "paths": [],
                        "rejected_paths": [],
                        "error": error_msg,
                        "patch_details": [],
                    }

            raw_patch_count = len(patches)
            allowed_paths = []
            rejected_paths = []

            for patch_item in patches:
                if not isinstance(patch_item, dict):
                    continue
                op = patch_item.get("op", "")
                path = patch_item.get("path", "")
                value = patch_item.get("value")
                reason = patch_item.get("reason", "")
                patch_group = patch_item.get("group", group_name)

                allowed, reject_reason = check_patch_allowed(path, group_name)
                if not allowed:
                    patches_rejected += 1
                    rejected_paths.append(path)
                    before_val, _ = resolve_json_pointer(plan_json, path)
                    patch_details.append({
                        "log_type": "patch_detail",
                        "line_index": line_index,
                        "record_id": record_id,
                        "group": group_name,
                        "status": "rejected",
                        "op": op,
                        "path": path,
                        "reason": reason,
                        "rule_source": group_name,
                        "before_value": before_val,
                        "after_value": value,
                        "before_value_preview": str(before_val)[:300] if before_val else "",
                        "after_value_preview": str(value)[:300] if value else "",
                        "rejected_reason": reject_reason,
                        "error": "",
                    })
                    continue

                before_val, exists = resolve_json_pointer(plan_json, path)
                success, old_val, new_val = apply_patch(plan_json, patch_item)

                if success:
                    patches_applied += 1
                    allowed_paths.append(path)
                    if isinstance(value, str) and isinstance(old_val, str):
                        if len(value) < len(old_val) * 0.6:
                            print(f"  [WARN] line={line_index} {group_name} path={path} value shortened significantly ({len(old_val)} -> {len(value)})")

                    patch_details.append({
                        "log_type": "patch_detail",
                        "line_index": line_index,
                        "record_id": record_id,
                        "group": group_name,
                        "status": "applied",
                        "op": op,
                        "path": path,
                        "reason": reason,
                        "rule_source": group_name,
                        "before_value": old_val,
                        "after_value": new_val,
                        "before_value_preview": str(old_val)[:300] if old_val is not None else "",
                        "after_value_preview": str(new_val)[:300] if new_val is not None else "",
                        "rejected_reason": "",
                        "error": "",
                    })
                else:
                    patch_details.append({
                        "log_type": "patch_detail",
                        "line_index": line_index,
                        "record_id": record_id,
                        "group": group_name,
                        "status": "failed",
                        "op": op,
                        "path": path,
                        "reason": reason,
                        "rule_source": group_name,
                        "before_value": old_val,
                        "after_value": new_val,
                        "before_value_preview": str(old_val)[:300] if old_val is not None else "",
                        "after_value_preview": str(new_val)[:300] if new_val is not None else "",
                        "rejected_reason": "",
                        "error": "Patch application failed",
                    })

            if patches_applied > 0:
                status = "success"
            elif patches_rejected > 0 and patches_applied == 0:
                status = "rejected_patch"
            else:
                status = "no_change"

        except Exception as e:
            status = "llm_error"
            error_msg = str(e)
            print(f"  [ERR] line={line_index} {group_name}: {e}")

        output_hash = json_hash(plan_json)

        summary = {
            "log_type": "group_summary",
            "line_index": line_index,
            "record_id": record_id,
            "group": group_name,
            "status": status,
            "llm_call_count": llm_call_count,
            "raw_patch_count": len(patches) if patches else 0,
            "patch_count": patches_applied,
            "paths": allowed_paths if patches_applied > 0 else [],
            "rejected_paths": rejected_paths,
            "error": error_msg,
            "group_input_plan_hash": input_hash,
            "group_output_plan_hash": output_hash,
        }

        return plan_json, {
            "status": status,
            "llm_call_count": llm_call_count,
            "raw_patch_count": raw_patch_count if patches else 0,
            "patch_count": patches_applied,
            "paths": allowed_paths if patches_applied > 0 else [],
            "rejected_paths": rejected_paths,
            "error": error_msg,
            "patch_details": patch_details,
            "summary": summary,
        }

    async def process_sample(self, line_index, original_record):
        record_id = original_record.get("id", f"line_{line_index}")
        self.in_progress[line_index] = "Group A"

        try:
            model_output_str = original_record.get("model_output", "")
            try:
                plan_json = json.loads(model_output_str)
            except json.JSONDecodeError:
                plan_json = json.loads(model_output_str.encode('utf-8').decode('unicode_escape'))

            plan_json = copy.deepcopy(plan_json)
        except Exception as e:
            print(f"[ERR] line={line_index}: Cannot parse model_output: {e}")
            self.failed += 1
            self.failed_samples.append({"line_index": line_index, "record_id": record_id, "error": str(e), "failed_group": "parse"})
            save_checkpoint_entry(line_index, "failed", original_record, [], [])
            del self.in_progress[line_index]
            return

        user_info_and_system = (
            f"【System Prompt】\n{original_record.get('system_prompt', '')}\n\n"
            f"【User Prompt (用户基本信息/体检信息)】\n{original_record.get('user_prompt', '')}"
        )

        completed_groups = []
        llm_node_used_set = set()
        any_group_failed = False

        for group_name in GROUPS:
            self.in_progress[line_index] = group_name
            plan_json, result = await self.process_group(
                line_index, record_id, group_name, plan_json, user_info_and_system, llm_node_used_set
            )

            write_log(result.get("summary", {}))
            for pd in result.get("patch_details", []):
                write_log(pd)

            if result["status"] in ("success", "no_change"):
                completed_groups.append(group_name)
            elif result["status"] in ("llm_error", "patch_error"):
                any_group_failed = True
                completed_groups.append(group_name)
            elif result["status"] == "rejected_patch":
                completed_groups.append(group_name)

        final_record = copy.deepcopy(original_record)
        final_record["model_output"] = json.dumps(plan_json, ensure_ascii=False)

        status = "failed" if any_group_failed else "completed"
        if any_group_failed:
            self.failed += 1
            self.failed_samples.append({
                "line_index": line_index,
                "record_id": record_id,
                "error": "Some groups had LLM errors",
                "failed_groups": [g for g in GROUPS if g not in completed_groups or any_group_failed],
            })
        else:
            self.completed += 1

        llm_nodes_list = list(llm_node_used_set)
        self.sample_llm_nodes[str(line_index)] = llm_nodes_list
        save_checkpoint_entry(line_index, status, final_record, completed_groups, llm_nodes_list)
        self.results[line_index] = final_record

        if line_index % 10 == 0:
            print(f"  [DONE] line={line_index} status={status} patches_applied={result['patch_count']} groups={completed_groups}")

        del self.in_progress[line_index]

    def report_progress(self):
        total = len(self.records)
        elapsed = time.time() - self.start_time

        node_info = []
        for n in self.nodes:
            avg = n.avg_time()
            node_info.append(f"  {n.name}: active={n.active}/{n.max_concurrency} fail={n.total_failures} timeout={n.total_timeouts} avg={avg:.1f}s degraded={n.degraded}")

        in_progress_info = []
        for li, g in self.in_progress.items():
            in_progress_info.append(f"line {li}: {g}")

        print(f"\n{'='*60}")
        print(f"PROGRESS [{elapsed:.0f}s]")
        print(f"  total={total} completed={self.completed} failed={self.failed} in_progress={len(self.in_progress)}")
        print(f"  active_workers={sum(n.active for n in self.nodes)}/{sum(n.max_concurrency for n in self.nodes)}")
        for ni in node_info:
            print(ni)
        if in_progress_info:
            print(f"  Processing: {', '.join(in_progress_info[:10])}")
        print(f"  checkpoint: {CHECKPOINT_PATH}")
        print(f"  log: {LOG_PATH}")
        print(f"{'='*60}\n")

    async def progress_reporter(self):
        while True:
            await asyncio.sleep(PROGRESS_INTERVAL)
            self.report_progress()

    async def run(self):
        load_rule_file()
        self.records = load_input_jsonl()
        self.checkpoint = load_checkpoint()

        if self.checkpoint:
            completed_indices = {
                idx for idx, entry in self.checkpoint.items()
                if entry.get("status") == "completed"
            }
            print(f"[RESUME] Skipping {len(completed_indices)} already completed samples.")
            for idx, entry in self.checkpoint.items():
                if entry.get("status") == "completed":
                    self.results[idx] = entry["record"]
                    self.completed += 1
        else:
            completed_indices = set()

        os.makedirs(OUTPUT_DIR, exist_ok=True)

        print(f"\n[START] Processing {len(self.records)} samples with {len(LLM_NODES)} LLM nodes.")
        print(f"  Output: {OUTPUT_JSONL}")
        print(f"  Log: {LOG_PATH}")
        print(f"  Checkpoint: {CHECKPOINT_PATH}")
        print(f"  Global max concurrency: {sum(n.max_concurrency for n in self.nodes)}")

        self.report_progress()

        tasks = []
        reporter = asyncio.create_task(self.progress_reporter())

        for line_index, original_record in self.records:
            if line_index in completed_indices:
                continue
            task = asyncio.create_task(self.process_sample(line_index, original_record))
            tasks.append(task)

        batch_size = sum(n.max_concurrency for n in self.nodes)
        for i in range(0, len(tasks), batch_size):
            batch = tasks[i:i + batch_size]
            await asyncio.gather(*batch, return_exceptions=True)
            if i + batch_size < len(tasks):
                self.report_progress()

        reporter.cancel()
        try:
            await reporter
        except asyncio.CancelledError:
            pass

        self.write_output()
        self.final_report()

    def write_output(self):
        sorted_indices = sorted(self.results.keys())
        with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
            for idx in sorted_indices:
                f.write(json.dumps(self.results[idx], ensure_ascii=False) + "\n")
        print(f"\n[OUTPUT] Written {len(sorted_indices)} records to {OUTPUT_JSONL}")

    def final_report(self):
        total = len(self.records)
        print(f"\n{'='*60}")
        print(f"FINAL REPORT")
        print(f"  Total: {total}")
        print(f"  Completed: {self.completed}")
        print(f"  Failed: {self.failed}")
        print(f"  Output: {OUTPUT_JSONL}")
        print(f"  Log: {LOG_PATH}")
        print(f"  Checkpoint: {CHECKPOINT_PATH}")
        if self.failed_samples:
            print(f"\n  Failed samples ({len(self.failed_samples)}):")
            for fs in self.failed_samples[:20]:
                print(f"    line={fs['line_index']} id={fs['record_id']} error={fs['error']}")
        for n in self.nodes:
            print(f"  Node '{n.name}': total_fail={n.total_failures} total_timeout={n.total_timeouts} degraded={n.degraded}")
        print(f"{'='*60}")


if __name__ == "__main__":
    engine = RepairEngine()
    asyncio.run(engine.run())
