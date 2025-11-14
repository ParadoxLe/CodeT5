from human_eval.data import read_problems, write_jsonl, stream_jsonl
import glob
from tqdm import tqdm
import argparse
import ast  # 导入语法解析模块

# 非Python语言的标志性关键字（可根据实际情况补充）
NON_PYTHON_KEYWORDS = {
    # ------------------------------
    # 1. 不应出现在函数体内的Python相关声明（位置错误）
    # ------------------------------
    '#!/usr/bin/env python',    # 脚本头部声明，不能在函数内
    '#!/usr/bin/python',        # 类似变体
    '# -*- coding: utf-8 -*-',  # 编码声明，应在文件头部
    'if __name__ == "__main__":',  # 主程序入口，不应在函数内

    # ------------------------------
    # 2. 其他编程语言的标志性语法
    # ------------------------------
    # C/C++
    '#include',         '#define',        'using namespace',
    'std::',            'int main',       'cout',             'endl',
    'printf(',          'scanf(',         'class ',           'public:',
    # // Java
    'package ',         'import java.',   'public class',     'extends ',
    'implements ',      'new ',           'System.out.println',
    # // JavaScript
    'function ',        'console.log',    'let ',             'const ',
    'document.getElementById',
    # // PHP
    '<?php',            'echo ',          '$_',               'require_once',
    # // Shell
    '#!/bin/bash',      'export ',        'cd ',              'ls ',
    # // 其他
    'print(',           'def main():',    #// 混淆语法（Python中print()合法，但结合其他特征时过滤）

    # ------------------------------
    # 3. 格式错误/无关标记
    # ------------------------------
    '```',              '```python',      '```py',            #// markdown代码块标记
    '<!--',             '-->',           # // HTML注释
    '/*',               '*/',             #// C风格注释（Python中用#）
}

def is_valid_python(completion):
    """检查是否包含非Python关键字"""
    for keyword in NON_PYTHON_KEYWORDS:
        if keyword in completion:
            return False
    return True

def check_python_syntax(code):
    """检查代码是否符合Python语法"""
    try:
        ast.parse(code)  # 解析成功 = 语法合法
        return True
    except SyntaxError:
        return False

parser = argparse.ArgumentParser()

# Inputs
parser.add_argument(
    '--path',
    type=str,
    default="preds/codet5p-770M_T0.2_N200",
    help="")
parser.add_argument(
    '--out_path',
    type=str,
    default='preds/merged_results.jsonl',
    help="")
parser.add_argument(
    '--add_prompt',
    action='store_true',
    help='')

args = parser.parse_args()

files = sorted(glob.glob(args.path + '/*.jsonl'))
print("{} files in {}".format(len(files), args.path))

problems = read_problems('data/HumanEval.jsonl.gz')

output = []
for code_file in tqdm(files, total=len(files)):
    codes = [c for c in stream_jsonl(code_file)]
    if args.add_prompt:
        for code in codes:
            task_id = code['task_id']
            prompt = problems[task_id]['prompt']
            completion = code['completion'].strip()  # 去除首尾空格

            # 过滤1：包含非Python关键字 → 跳过
            if not is_valid_python(completion):
                print(f"跳过 {task_id}：包含非Python语法关键字")
                continue

            # 拼接完整代码（prompt + completion）
            all_code = prompt.rstrip('\n') + '\n' + completion  # 确保格式正确

            # 过滤2：语法错误 → 跳过
            if not check_python_syntax(all_code):
                print(f"跳过 {task_id}：Python语法错误")
                continue

            # 保留合法代码
            code['all_code'] = all_code
            output.append(code)
    else:
        # 若不拼接prompt，直接检查completion
        for code in codes:
            completion = code['completion'].strip()
            if is_valid_python(completion) and check_python_syntax(completion):
                output.append(code)

    output += codes

print("save to {}".format(args.out_path))
write_jsonl(args.out_path, output)
