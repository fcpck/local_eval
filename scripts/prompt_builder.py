# -*- coding: utf-8 -*-
"""Prompt 构造与解析（纯函数，核心）。

- extract_excerpt: 按 rubric 的 extract_key_path 从模型回答 JSON 提取片段（健壮解析 + 降级）
- build_conversation: 构造 {{conversation}} 内容（用户输入 + 提取片段）
- render_judge_prompt: 填充 judge prompt 模板
- parse_judge_response: 解析 judge 返回的 {explanation, criteria_met, evidence}

提取/截断逻辑沿用旧版 judge_runner.py 的 extract_nested / extract_fields。
"""

import json
import re

# 与旧版 judge_runner.py 一致
MAX_FIELD_CHARS = 2000
MAX_PLAN_CHARS = 7000


# ==================== 提取 ====================

def extract_nested(plan, path):
    """按 'a.b.c' / 'a[0].b' 形式的 path 从 plan 中提取值。"""
    if not path:
        return None
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


def _serialize_value(v):
    """将提取值序列化为字符串，list/dict 用 JSON。"""
    if isinstance(v, (dict, list)):
        return json.dumps(v, ensure_ascii=False)
    return str(v)


def extract_fields(plan, extract_key_path):
    """按 extract_key_path（逗号分隔）从已解析的 plan 提取内容。

    列表限 5 条；字段截断 MAX_FIELD_CHARS；整体截断 MAX_PLAN_CHARS。
    键缺失时标注 "[path] (字段不存在)"。
    """
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
            items = v[:5]
            for i, item in enumerate(items):
                s = _serialize_value(item)
                if len(s) > MAX_FIELD_CHARS:
                    s = s[:MAX_FIELD_CHARS] + "...(截断)"
                parts.append(f"[{path}][{i}] {s}")
        else:
            s = _serialize_value(v)
            if len(s) > MAX_FIELD_CHARS:
                s = s[:MAX_FIELD_CHARS] + "...(截断)"
            parts.append(f"[{path}] {s}")

    joined = "\n".join(parts)
    if len(joined) > MAX_PLAN_CHARS:
        joined = joined[:MAX_PLAN_CHARS] + "\n...(整体截断)"
    return joined


def _strip_code_fence(text):
    """剥离首尾可能包裹的 ```json ... ``` 代码围栏。"""
    text = text.strip()
    m = re.match(r"^```(?:json)?\s*\n?(.*?)\n?```$", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return text


def _balance_brackets(text):
    """用栈模拟 JSON 括号平衡，若仅缺少尾部闭合括号则补全。

    返回 (fixed_text, ok)。字符串内容中的括号（含转义）被正确跳过。
    修复思路：扫描全部字符（跳过字符串内的内容），若括号配对最终差 n 个
    右括号（且过程中没有多余右括号），则在末尾补 n 个右括号。
    """
    stack = []
    in_str = False
    i = 0
    n = len(text)
    while i < n:
        c = text[i]
        if in_str:
            if c == "\\":
                i += 2
                continue
            if c == '"':
                in_str = False
            i += 1
            continue
        if c == '"':
            in_str = True
        elif c in "{[":
            stack.append(c)
        elif c in "}]":
            if stack:
                stack.pop()
            else:
                return None, False  # 多余闭合括号，无法简单修复
        i += 1
    if in_str:
        return None, False  # 字符串未闭合
    if not stack:
        return text, True  # 已平衡（无缺失）
    # 缺失 len(stack) 个右括号
    closes = {"{": "}", "[": "]"}[stack[-1]]
    fixed = text + closes * len(stack)
    return fixed, True


def parse_model_output(raw_output):
    """健壮解析 model_output 字符串为 dict。

    依次尝试：
      1. 直接 json.loads
      2. 剥离 markdown 代码围栏后 json.loads
      3. 括号平衡补全后 json.loads（修复尾部缺失闭合括号）
      4. json.JSONDecoder.raw_decode 取第一个完整 JSON（忽略尾部冗余）
    返回 (plan_or_None, parse_status)，status ∈ {"ok", "code_fence", "balance_fix", "raw_decode", "fail"}。
    """
    if not raw_output or not isinstance(raw_output, str):
        return None, "fail"

    text = raw_output.strip()
    if not text:
        return None, "fail"

    decoder = json.JSONDecoder()

    # 1. 直接解析
    try:
        return json.loads(text), "ok"
    except json.JSONDecodeError:
        pass

    # 2. 剥围栏后解析
    fenced = _strip_code_fence(text)
    if fenced != text:
        try:
            return json.loads(fenced), "ok"
        except json.JSONDecodeError:
            pass

    # 3. 括号平衡补全
    for candidate in (text, fenced):
        fixed, ok = _balance_brackets(candidate)
        if ok and fixed != candidate:
            try:
                return json.loads(fixed), "balance_fix"
            except json.JSONDecodeError:
                pass

    # 4. raw_decode 取第一个完整 JSON（跳过前导空白、忽略尾部冗余字符）
    for candidate in (text, fenced):
        try:
            obj, _ = decoder.raw_decode(candidate)
            if obj is not None:
                return obj, "raw_decode"
        except (json.JSONDecodeError, ValueError):
            continue

    return None, "fail"


def extract_excerpt(raw_output, extract_key_path, fallback="raw"):
    """从模型回答原始字符串中按 extract_key_path 提取片段。

    返回 (excerpt, status)：
      - status ∈ {"ok", "json_fallback_whole", "json_fallback_raw", "empty"}
      - fallback="raw": 解析失败时整个 raw_output 作为片段
      - fallback="whole": 解析失败但前缀修复成功时用修复后的整体 JSON 序列化
    """
    plan, parse_status = parse_model_output(raw_output)
    if plan is None:
        if fallback == "raw" and raw_output:
            return raw_output.strip(), "json_fallback_raw"
        return "", "empty"

    excerpt = extract_fields(plan, extract_key_path)
    if not excerpt or excerpt.strip() == "":
        return "", "empty"

    if parse_status != "ok":
        # 解析有降级但成功（code_fence / balance_fix / raw_decode）——仍按路径提取，标注降级来源
        return excerpt, f"json_fallback_whole({parse_status})"

    return excerpt, "ok"


# ==================== 构造 ====================

def build_conversation(user_prompt, excerpt):
    """构造对话：用户输入（含健康数据）+ 助手回复（按 rubric 提取的字段）。"""
    return (
        f"# 用户输入（含健康数据）\n{user_prompt}\n\n"
        f"# 助手回复（已按 rubric 提取相关字段）\n{excerpt}"
    )


def render_judge_prompt(template, conversation, rubric_item):
    """填充 judge prompt 模板：{{conversation}} 和 {{rubric_item}}。"""
    criterion = rubric_item.get("criterion", "") if isinstance(rubric_item, dict) else str(rubric_item)
    return (
        template
        .replace("{{conversation}}", conversation)
        .replace("{{rubric_item}}", criterion)
    )


# ==================== 解析 judge 返回 ====================

def _normalize_criteria_met(val):
    """规范化 criteria_met：true/false/null 各种形态 → True/False/None。"""
    if val is True or val is False or val is None:
        return val
    if isinstance(val, str):
        s = val.strip().lower()
        if s in ("true", "yes", "是", "符合", "合规"):
            return True
        if s in ("false", "no", "否", "不符合", "不合规"):
            return False
        if s in ("null", "none", "n/a", "na", "不适用"):
            return None
    return None


def parse_judge_response(content):
    """从 judge 回复解析 {explanation, criteria_met, evidence}。

    三级兜底：直接 json.loads → markdown 代码块 → 首对花括号子串。
    返回 dict: {ok, criteria_met, explanation, evidence, raw, error}
    """
    if not content:
        return {"ok": False, "criteria_met": None, "explanation": "",
                "evidence": "", "raw": "", "error": "empty content"}
    text = content.strip()

    candidates = []
    # 1. 直接
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            candidates.append(obj)
    except Exception:
        pass

    # 2. markdown 代码块
    if not candidates:
        m = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
        if m:
            try:
                obj = json.loads(m.group(1))
                if isinstance(obj, dict):
                    candidates.append(obj)
            except Exception:
                pass

    # 3. 首对花括号
    if not candidates:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                obj = json.loads(text[start:end + 1])
                if isinstance(obj, dict):
                    candidates.append(obj)
            except Exception:
                pass

    if not candidates:
        return {"ok": False, "criteria_met": None, "explanation": "",
                "evidence": "", "raw": content,
                "error": f"Could not parse JSON. Preview: {text[:300]}"}

    obj = candidates[0]
    return {
        "ok": True,
        "criteria_met": _normalize_criteria_met(obj.get("criteria_met")),
        "explanation": obj.get("explanation") or "",
        "evidence": obj.get("evidence") or "",
        "raw": content,
        "error": "",
    }
