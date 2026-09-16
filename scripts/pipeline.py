# -*- coding: utf-8 -*-
"""本地测评管线统一 CLI 入口。

用法:
  python pipeline.py --config config.json judge --model glm-5.2 [--limit-rows 2 --limit-rubrics 3]
  python pipeline.py --config config.json judge --all-models [--mock]
  python pipeline.py --config config.json report --model glm-5.2 [--examples 5]
  python pipeline.py --config config.json report-multi --models glm-5.2,qwen36_35b
  python pipeline.py --config config.json build
  python pipeline.py --config config.json crawl --model NEW_MODEL_NAME
"""

import argparse
import json
import sys


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def cmd_judge(args):
    from judge_runner import main as judge_main
    sys.argv = [sys.argv[0], "--config", args.config]
    if args.all_models:
        sys.argv += ["--all-models"]
    if args.model:
        sys.argv += ["--model", args.model]
    if args.limit_rows:
        sys.argv += ["--limit-rows", str(args.limit_rows)]
    if args.limit_rubrics:
        sys.argv += ["--limit-rubrics", str(args.limit_rubrics)]
    if args.mock:
        sys.argv += ["--mock"]
    if args.rubrics:
        sys.argv += ["--rubrics", args.rubrics]
    if args.answers:
        sys.argv += ["--answers", args.answers]
    judge_main()


def cmd_report(args):
    from report_single_model import main as report_main
    sys.argv = [sys.argv[0], "--config", args.config, "--model", args.model]
    if args.examples:
        sys.argv += ["--examples", str(args.examples)]
    report_main()


def cmd_report_multi(args):
    from report_multi_model import main as multi_main
    sys.argv = [sys.argv[0], "--config", args.config]
    if args.models:
        sys.argv += ["--models", args.models]
    if args.run_tag:
        sys.argv += ["--run-tag", args.run_tag]
    multi_main()


def cmd_build(args):
    from build_prompts import main as build_main
    sys.argv = [sys.argv[0], "--config", args.config]
    build_main()


def cmd_crawl(args):
    from crawl_answers import main as crawl_main
    sys.argv = [sys.argv[0], "--config", args.config]
    if args.model:
        sys.argv += ["--model", args.model]
    crawl_main()


def main():
    parser = argparse.ArgumentParser(description="本地可控制模型测评管线")
    parser.add_argument("--config", default=r"D:\PycharmProject\git\Health_Management_Diabetes\local_eval\config.json")
    sub = parser.add_subparsers(dest="command", required=True)

    p_judge = sub.add_parser("judge", help="运行 rubric judge")
    p_judge.add_argument("--model", default="")
    p_judge.add_argument("--all-models", action="store_true")
    p_judge.add_argument("--limit-rows", type=int, default=0)
    p_judge.add_argument("--limit-rubrics", type=int, default=0)
    p_judge.add_argument("--mock", action="store_true", help="离线 mock，不调真实 API")
    p_judge.add_argument("--rubrics", default="", choices=["", "all", "grading", "doctor_pref"],
                         help="rubric 集合：all/grading/doctor_pref；默认读 config.rubric_source")
    p_judge.add_argument("--answers", default="", help="自定义答案源 JSONL（如爬取结果），默认用测试集")

    p_report = sub.add_parser("report", help="生成单模型 Markdown 报告")
    p_report.add_argument("--model", required=True)
    p_report.add_argument("--examples", type=int, default=0)

    p_multi = sub.add_parser("report-multi", help="生成多模型汇总报告")
    p_multi.add_argument("--models", default="", help="逗号分隔模型名；空=全部")
    p_multi.add_argument("--run-tag", default="", help="指定 run_tag 目录")

    p_build = sub.add_parser("build", help="预处理产出爬取用数据集")
    p_crawl = sub.add_parser("crawl", help="爬取新模型回答（预留）")
    p_crawl.add_argument("--model", default="")

    args = parser.parse_args()
    load_config(args.config)  # 校验配置可读

    if args.command == "judge":
        cmd_judge(args)
    elif args.command == "report":
        cmd_report(args)
    elif args.command == "report-multi":
        cmd_report_multi(args)
    elif args.command == "build":
        cmd_build(args)
    elif args.command == "crawl":
        cmd_crawl(args)


if __name__ == "__main__":
    main()
