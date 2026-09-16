# 本地可控制模型测评管线（四高一重 Rubric Judge）

在自己电脑上可控地运行"四高一重"健康管理方案的测评任务，不依赖服务器上的测评平台。

测评流程：**加载测评集 → 爬取待测模型回答 → 按 rubric 提取回答片段 → 组合 conversation 交给 judge 模型评分 → 生成报告**。

## 目录结构

```
local_eval\
├── config.json            # 运行时配置（必填项见下文）
├── config.example.json    # 无真实 token 的模板（可分享）
├── data\
│   ├── crawl_dataset.jsonl     # build 产出：爬取用数据集 {id, system_prompt, user_prompt}
│   └── crawled\                # crawl 产出：爬取结果 result_{模型名}.jsonl
├── output\
│   ├── raw_judge\{run_tag}\{模型}.jsonl      # judge 原始判分（每行一条）
│   ├── judge_failures\{run_tag}\{模型}.jsonl # judge 失败记录（可断点续跑补齐）
│   ├── report_single\{模型}.md               # 单模型报告
│   └── report_multi\{run_tag}\all_models.md/xlsx  # 多模型对比报告
└── scripts\
    ├── pipeline.py          # 统一入口（推荐用这个）
    ├── judge_runner.py      # judge 主趟
    ├── crawl_answers.py     # 爬取待测模型回答
    ├── build_prompts.py     # 预处理（生成爬取用数据集）
    ├── report_single_model.py  # 单模型报告
    ├── report_multi_model.py   # 多模型对比报告
    ├── api_client.py        # CSB 网关调用封装
    ├── prompt_builder.py    # 提取/构造/解析（核心纯函数）
    └── result_store.py      # JSONL 读写与断点续跑
```

## 环境依赖

Python 3.11+，依赖：`aiohttp`、`openpyxl`（报告 Excel 用）。均已安装。

## 配置文件（config.json）

### paths —— 数据路径

| 字段 | 说明 |
|------|------|
| `test_set` | 测评集 JSONL（20 条测试集），含 `four_models_outputs`（已有模型回答） |
| `system_prompt` | system prompt 模板（v8.8），含 `{{管理总周期}}` 占位符 |
| `judge_template` | judge prompt 模板，含 `{{conversation}}`、`{{rubric_item}}` 占位符 |
| `rubric_sets` | 命名 rubric 集合：`grading`（分级准确率，5 条）、`doctor_pref`（医生偏好，30 条） |
| `out_dir` / `data_dir` / `logs_dir` | 输出/数据/日志目录 |

### crawl —— 待测模型（爬取回答）

| 字段 | 说明 |
|------|------|
| `url` | 推理服务完整 URL（8088 `/v2/infer/{id}/v1/chat/completions`） |
| `model_name` | 请求体中的模型名（如 `qwen38_27b`） |
| `bearer_token` | Bearer 鉴权 token |
| `models_to_crawl` | 要爬取的模型备注名列表（备注名 = 结果文件与 `four_models_outputs` 的键名） |
| `concurrency` | 爬取并发（8088 网关建议 30） |
| `timeout` / `max_retries` / `max_tokens` | 超时 / 重试 / 最大 token |
| `stream` | 是否流式（8088 网关短回复可 `false`） |
| `save_interval` | 每 N 条落盘一次 |

> 兼容旧格式：也可用 `infer_id` + `csb_token`（8087 csb-inner-service，流式）。

### judge —— 裁判模型

| 字段 | 说明 |
|------|------|
| `url` | judge 服务完整 URL |
| `model_name` | judge 模型名（如 `qwen36_35b`） |
| `bearer_token` | Bearer 鉴权 token |
| `concurrency` | judge 并发（当前 30） |
| `timeout` / `max_retries` | 单次调用超时 / API 内部重试 |
| `max_tokens` | judge 回复最大 token（超长 explanation 会截断 JSON，调大到 16384） |
| `max_retries_total` | 整条判分（含解析失败）的总重试次数 |
| `stream` | 是否流式 |

### 其他

| 字段 | 说明 |
|------|------|
| `report.single_examples` | 单模型报告每条 rubric 展示的 false 样例数 |
| `rubric_source` | 默认 rubric 集合：`all` / `grading` / `doctor_pref` |
| `run_tag` | `auto` = 每次运行按时间戳建目录；固定字符串 = 复用目录实现断点续跑 |

## 命令说明

统一入口（推荐，以下均在 `local_eval\scripts` 目录下执行）：

```bash
python pipeline.py --config ../config.json <command> [选项]
```

### 1. build —— 生成爬取用数据集

```bash
python pipeline.py --config ../config.json build
```

读 `test_set` + `system_prompt`，把 `{{管理总周期}}` 替换为每条用例的 `management_total_period`，产出 `data/crawl_dataset.jsonl`（供爬取使用）。**judge 已有回答时不需要这一步**。

### 2. crawl —— 爬取待测模型回答

```bash
python pipeline.py --config ../config.json crawl --model qwen38_27b
```

- 对 `data/crawl_dataset.jsonl` 的 20 条用例调用待测模型接口，输出 `data/crawled/result_{备注名}.jsonl`
- 断点续跑：已爬过的 id 自动跳过
- `--model` 省略时爬取 `models_to_crawl` 全部

### 3. judge —— 运行 rubric 评分

```bash
# 用测试集里已有的模型回答评分
python pipeline.py --config ../config.json judge --model glm-5.2
python pipeline.py --config ../config.json judge --all-models

# 用爬取到的回答评分（--answers 指定答案源文件）
python pipeline.py --config ../config.json judge --model qwen38_27b --answers ../data/crawled/result_qwen38_27b.jsonl

# 选择 rubric 集合
python pipeline.py --config ../config.json judge --model glm-5.2 --rubrics grading
python pipeline.py --config ../config.json judge --model glm-5.2 --rubrics doctor_pref
python pipeline.py --config ../config.json judge --model glm-5.2 --rubrics all

# 快速冒烟（少量用例/少量 rubric）
python pipeline.py --config ../config.json judge --model glm-5.2 --limit-rows 2 --limit-rubrics 3
```

- 对每个 (模型 × 用例 × rubric) 调用一次 judge，输出 `output/raw_judge/{run_tag}/{模型}.jsonl`
- **断点续跑**：以 `模型|用例|rubric` 为唯一键，已判过的自动跳过；失败记录在 `judge_failures/`，重跑只补失败部分
- `--mock` 用本地假响应离线验证编排（不调真实 API）

### 4. report —— 生成报告

```bash
# 单模型报告（Markdown）
python pipeline.py --config ../config.json report --model glm-5.2

# 多模型对比报告（Markdown + Excel）
python pipeline.py --config ../config.json report-multi --models glm-5.2,qwen38_27b
python pipeline.py --config ../config.json report-multi   # 不加 --models = 全部模型
```

报告自动使用最新一个 run_tag 目录的结果；可用 `--run-tag` 指定。

## 报告口径

- **总通过率%** = true / 有效判分(true+false)，整体维度
- **全通过率%** = 回复所有 rubric 评分无 false 且无 error（全为 true/null）的用例占比
- 单模型报告主视角是**按 20 条回复（用例）明细**：每条回复的 true/false/null/error、用例通过率、是否全对、未通过的 rubric；分 rubric 明细作为附录
- 多模型报告主视角是**模型对比表**（每模型的总通过率% + 全通过率% + 计数），分 rubric 通过率矩阵作为附录

## 典型使用流程

```bash
# 1. 新模型爬取回答
python pipeline.py --config ../config.json build
python pipeline.py --config ../config.json crawl --model qwen38_27b

# 2. 评分（分级准确率）
python pipeline.py --config ../config.json judge --model qwen38_27b --rubrics grading --answers ../data/crawled/result_qwen38_27b.jsonl

# 3. 报告
python pipeline.py --config ../config.json report --model qwen38_27b
```

## 说明与注意

- **curl 直调 8088 网关**：body 必须用 `--data @文件`，内联 `-d '{...}'` 会报 `400 error parsing the body`（本管线用 aiohttp 不受影响）
- **流式 vs 非流式**：8087 csb-inner-service 非流式有 180s 超时，长回复必须流式；8088 `/v2/infer` 网关短回复可非流式
- **提取降级**：个别模型的 model_output 不是合法 JSON（如 4g1z 部分回答），会自动括号补全修复；仍失败的整段原文传给 judge（不截断），报告里以"片段降级(原文)"统计
- **judge 返回超长 explanation**：若 `max_tokens` 太小会截断 JSON 导致解析失败，失败记录在 `judge_failures/`，调大 `max_tokens` 后重跑即可补齐
