#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修复 frequencyValue 频次表达错误
问题：模型把"一周7次"写成 frequencyValue=7 + frequencyUnitText=每天
正确：frequencyUnitText=周 + frequencyValue=7，或 frequencyUnitText=天 + frequencyValue=1
"""

import json
import re
import sys


def fix_frequency_value(daily_tasks):
    """修复 dailyTasks 中的 frequencyValue 频次表达"""
    if not isinstance(daily_tasks, list):
        return daily_tasks, False

    fixed = False
    for task in daily_tasks:
        if not isinstance(task, dict):
            continue

        freq_value = task.get('frequencyValue')
        freq_unit = task.get('frequencyUnitText', '')
        task_type = task.get('taskTypeText', '')

        # 检查"每天N次"的错误表达（饮食/运动任务）
        # 条件：frequencyUnitText='天' 且 frequencyValue>=1，改为 frequencyUnitText='周'
        if freq_unit == '天' and isinstance(freq_value, int) and freq_value >= 1 and task_type in ('饮食', '运动'):
            task['frequencyUnitText'] = '周'
            fixed = True

    return daily_tasks, fixed


def process_file(input_path, output_path):
    """处理整个文件"""
    total = 0
    fixed_count = 0

    with open(input_path, 'r', encoding='utf-8') as fin, \
         open(output_path, 'w', encoding='utf-8') as fout:

        for line in fin:
            line = line.strip()
            if not line:
                continue

            total += 1
            record = json.loads(line)

            # 提取 model_output
            if 'model_output' in record:
                model_output_str = record['model_output']
            elif 'sessions' in record:
                # 从 sessions 中提取
                sessions = record.get('sessions', [])
                if sessions:
                    conv = sessions[0].get('conversation_history', [])
                    for msg in conv:
                        if msg.get('role') == 'assistant':
                            model_output_str = msg.get('content', '')
                            break
                    else:
                        fout.write(line + '\n')
                        continue
                else:
                    fout.write(line + '\n')
                    continue
            else:
                fout.write(line + '\n')
                continue

            # 尝试解析 JSON
            try:
                model_output = json.loads(model_output_str)
            except json.JSONDecodeError:
                # 解析失败，保持原样
                fout.write(line + '\n')
                continue

            # 修复 dailyTasks
            if 'dailyTasks' in model_output:
                daily_tasks, was_fixed = fix_frequency_value(model_output['dailyTasks'])
                if was_fixed:
                    fixed_count += 1
                    # 更新 record
                    if 'model_output' in record:
                        record['model_output'] = json.dumps(model_output, ensure_ascii=False)
                    elif 'sessions' in record:
                        # 更新 sessions 中的 content
                        sessions = record.get('sessions', [])
                        if sessions:
                            conv = sessions[0].get('conversation_history', [])
                            for msg in conv:
                                if msg.get('role') == 'assistant':
                                    msg['content'] = json.dumps(model_output, ensure_ascii=False)
                                    break

            # 写回文件
            fout.write(json.dumps(record, ensure_ascii=False) + '\n')

    print(f'总记录数: {total}')
    print(f'修复记录数: {fixed_count}')
    print(f'输出文件: {output_path}')


def main():
    if len(sys.argv) < 2:
        print('用法: python fix_frequency_value.py <input.jsonl> [output.jsonl]')
        sys.exit(1)

    input_path = sys.argv[1]
    if len(sys.argv) >= 3:
        output_path = sys.argv[2]
    else:
        # 自动生成输出文件名
        base = input_path.rsplit('.', 1)[0]
        output_path = f'{base}_fixed.jsonl'

    process_file(input_path, output_path)


if __name__ == '__main__':
    main()
