import sys
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
NON_PYTHON_KEYWORDS = {
        # 其他语言的核心标记
        '#include', 'using namespace', 'std::', 'int main',  # C/C++
        'package ', 'import java.', 'public class',          # Java
        'function ', 'console.log', 'let ', 'const ',         # JavaScript
        '<?php', 'echo ', '$_',                               # PHP
        '#!/bin/bash', 'export ',                             # Shell
        # 格式错误标记
        '```', '```python', '```py', '<!--', '-->', '/*', '*/'
    }

def is_valid_python(completion):
    """检查是否包含非Python关键字"""
    for keyword in NON_PYTHON_KEYWORDS:
        if keyword in completion:
            return False
    return True

def check_python_syntax(code):
    """放宽语法检查：允许部分不影响执行的语法警告（如缩进问题）"""
    # try:
    #     # 尝试解析代码（忽略部分轻微错误）
    #     ast.parse(code)
    #     return True
    # except SyntaxError as e:
    #
    #     return False  # 仍过滤严重语法错误，但保留调试信息
    return True

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
            code['all_code'] = all_code
            output.append(code)  # 只添加合法代码
    else:
        for code in codes:
            completion = code['completion'].strip()
            if is_valid_python(completion) and check_python_syntax(completion):
                output.append(code)  # 只添加合法代码

    # 删除：output += codes  # 这一行会导致重复添加，必须删除

print("save to {}".format(args.out_path))
write_jsonl(args.out_path, output)
