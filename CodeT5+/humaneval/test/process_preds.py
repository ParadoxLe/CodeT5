import sys
import re
import os

# 1. 确保human_eval模块可导入（之前的修改，保留）
current_dir = os.path.dirname(os.path.abspath(__file__))  # test文件夹路径
parent_dir = os.path.dirname(current_dir)  # humaneval文件夹路径（human_eval所在目录）
sys.path.append(parent_dir)
from human_eval.data import read_problems, write_jsonl, stream_jsonl
import glob
from tqdm import tqdm
import argparse
import ast  # 导入语法解析模块

# 非Python语言的标志性关键字（可根据实际情况补充）
NON_PYTHON_PATTERNS = [
    # 1. 不应出现在函数体内的Python声明（行首出现）
    re.compile(r'^#!/usr/bin/env python', re.MULTILINE),
    re.compile(r'^#!/usr/bin/python', re.MULTILINE),
    re.compile(r'^# -\*- coding: utf-8 -\*-', re.MULTILINE),
    re.compile(r'^if __name__ == "__main__":', re.MULTILINE),

    # 2. 其他语言关键字（完整单词/语法，不匹配子串）
    re.compile(r'\b#include\b', re.IGNORECASE),  # C/C++
    re.compile(r'\busing namespace\b', re.IGNORECASE),
    re.compile(r'\bstd::\b', re.IGNORECASE),
    re.compile(r'\bpublic class\b', re.IGNORECASE),  # Java
    re.compile(r'\bfunction\s+\w+\(', re.IGNORECASE),  # JavaScript函数定义
    re.compile(r'\bconsole\.log\b', re.IGNORECASE),
    re.compile(r'^<\?php', re.MULTILINE | re.IGNORECASE),  # PHP

    # 3. 格式错误标记（整行匹配或独立出现）
    re.compile(r'^```[a-z]*$', re.MULTILINE),  # markdown代码块
    re.compile(r'\/\*|\*\/'),  # C风格注释（Python用#）
]


def is_valid_python(completion):
    """检查是否包含非Python关键字"""
    for keyword in NON_PYTHON_PATTERNS:
        if keyword in completion:
            return False
    return True


def check_code_completeness(code):
    """检查代码完整性（括号、引号闭合等）"""
    # 检查括号平衡
    stack = []
    brackets = {'(': ')', '{': '}', '[': ']'}
    for char in code:
        if char in brackets:
            stack.append(char)
        elif char in brackets.values():
            if not stack or brackets[stack.pop()] != char:
                return False  # 括号不匹配
    if stack:
        return False  # 存在未闭合的括号

    # 检查引号闭合
    for quote in ['"', "'", '"""', "'''"]:
        if code.count(quote) % 2 != 0:
            return False  # 引号未闭合

    return True


def check_python_syntax(code):
    """检查语法合法性 + 确保包含有效函数定义"""
    if not code.strip():  # 过滤空代码
        return False
    try:
        tree = ast.parse(code)
        # 检查是否至少包含一个函数定义（符合HumanEval任务要求）
        has_function = any(
            isinstance(node, ast.FunctionDef)  # 函数定义
            or isinstance(node, ast.AsyncFunctionDef)  # 异步函数
            for node in tree.body
        )
        # 检查是否有明显无效结构（如单独的return语句）
        has_invalid_top_level = any(
            isinstance(node, (ast.Return, ast.Break, ast.Continue))
            for node in tree.body
        )
        return has_function and not has_invalid_top_level
    except SyntaxError:
        return False


parser = argparse.ArgumentParser()

# 修改 --path 参数的默认值（原默认值为 "preds/codet5p-770M_T0.2_N200"）
parser.add_argument(
    '--path',
    type=str,
    default="preds/codet5p-770M-native_T0.2_N200",  # 与生成路径一致
    help="预测结果所在目录"
)
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

# 2. 正确定位数据文件路径（新增的修改）
data_file_path = os.path.join(current_dir, "..", "data", "HumanEval.jsonl.gz")  # 拼接路径
data_file_path = os.path.abspath(data_file_path)  # 转为绝对路径
problems = read_problems(data_file_path)

output = []
for code_file in tqdm(files, total=len(files)):
    codes = [c for c in stream_jsonl(code_file)]
    if args.add_prompt:
        for code in codes:

            # ...（过滤和处理逻辑不变）...
            if not check_python_syntax(all_code):
                print(f"跳过 {task_id}：Python语法错误")
                continue
                # 在主逻辑中添加完整性检查
            if not check_code_completeness(all_code):
                print(f"跳过 {task_id}：代码不完整（括号/引号未闭合）")
                continue
            code['all_code'] = all_code
            output.append(code)  # 只添加合法代码
    else:
        for code in codes:
            completion = code['completion'].strip()
            if is_valid_python(completion) and check_python_syntax(completion):
                output.append(code)  # 只添加合法代码

print("save to {}".format(args.out_path))
write_jsonl(args.out_path, output)
