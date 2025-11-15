import argparse
import pprint
import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import sys
import ast
from tqdm import tqdm
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# 获取当前脚本所在目录（test文件夹）
current_dir = os.path.dirname(os.path.abspath(__file__))
# 向上一级目录（humaneval文件夹），即human_eval所在的目录
parent_dir = os.path.dirname(current_dir)
# 将parent_dir添加到Python搜索路径
sys.path.append(parent_dir)
from human_eval.data import write_jsonl, read_problems, stream_jsonl


def extract_text(prompt, remove_lines=False):
    """更精准地提取问题描述（保留换行以维持格式）"""
    token = '\"\"\"'
    start_idx = prompt.find(token) + len(token)
    end_idx = prompt.find('>>>', start_idx)  # 从start_idx后开始找，避免误匹配
    if end_idx == -1:
        end_idx = prompt.find(token, start_idx)  # 若没有>>>，用下一个"""作为结束

    output = prompt[start_idx:end_idx].strip()
    # 保留换行以维持问题结构（如列表、步骤）
    return output if not remove_lines else output.replace('\n', ' ')


INSTRUCTION = """Below is an instruction that describes a coding task. 
Write a complete, runnable Python function to solve the problem. 
The function must match the specified input/output requirements and pass all test cases.

### Instruction:
{}

### Response:
"""


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, default='Salesforce/codet5p-770M', help="")
    parser.add_argument('--output_path', type=str, help="")
    parser.add_argument('--start_index', type=int, default=0, help="")
    parser.add_argument('--end_index', type=int, default=164, help="")
    parser.add_argument('--temperature', type=float, default=0.8, help="")
    parser.add_argument('--N', type=int, default=200, help="")
    parser.add_argument('--max_len', type=int, default=600, help="")
    parser.add_argument('--decoding_style', type=str, default='sampling', help="")
    parser.add_argument('--num_seqs_per_iter', type=int, default=50, help='')
    parser.add_argument('--overwrite', action='store_true', help='')

    # 在 argparse 中添加 beam search 参数
    parser.add_argument('--num_beams', type=int, default=10, help="束搜索的束数量（仅用于 beam_search 策略）")
    parser.add_argument('--length_penalty', type=float, default=0.8, help="长度惩罚（减少过短/过长生成）")

    args = parser.parse_args()

    argsdict = vars(args)
    print(pprint.pformat(argsdict))

    STOP_SEQS = [
        '\nclass', '\ndef', '\n#', '\nif', '\nprint',
        '\nassert', '\nimport', '\nfrom',  # 避免生成导入语句或断言
        '\n"""', '\n\'\'\''  # 避免生成多余文档字符串
    ]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    problems = read_problems()

    task_ids = sorted(problems.keys())[args.start_index: args.end_index]
    prompts = [problems[task_id]['prompt'] for task_id in task_ids]
    num_samples = len(prompts)
    print("Number of samples: {}".format(num_samples))

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    model = AutoModelForSeq2SeqLM.from_pretrained(args.model,
                                                  trust_remote_code=False,  # False for 220m and 770m models
                                                  dtype=torch.float16,
                                                  low_cpu_mem_usage=True)
    model.eval()
    model.to(device)

    # for larger LLMs such as 2B, 6B, and 16B, we need to pass the text prompt to the decoder
    prompt_to_decoder = True if any([size in args.model for size in ['2b', '6b', '16b']]) else False

    print(f"Loaded {args.model}.")
    for i in tqdm(range(num_samples), ncols=0, total=num_samples):
        output_file = args.output_path + '/{}.jsonl'.format(args.start_index + i)

        if os.path.exists(output_file) and not args.overwrite:
            print(f'Skip {output_file} as it already exists')
            continue

        prompt = prompts[i].replace('    ', '\t')
        if args.model == 'Salesforce/instructcodet5p-16b':
            prompt_batch = [INSTRUCTION.format(extract_text(prompt))]
            prompt_batch_decoder = [INSTRUCTION.format(extract_text(prompt)) + prompt]
        else:
            prompt_batch = [prompt]
            prompt_batch_decoder = [prompt]

        ids_batch = [task_ids[i]]

        completion_seqs = []

        encoding = tokenizer(prompt_batch, return_tensors="pt", truncation=True, max_length=args.max_len).to(device)
        encoding_decoder = tokenizer(prompt_batch_decoder, return_tensors="pt", truncation=True,
                                     max_length=args.max_len).to(device)

        if args.decoding_style == 'sampling':
            loops = int(args.N / args.num_seqs_per_iter)
        else:
            loops = 1

        for _ in tqdm(range(loops), total=loops, leave=False, ncols=0):

            with torch.no_grad():
                if args.decoding_style == 'sampling':
                    if prompt_to_decoder:
                        gen_tokens = model.generate(**encoding,
                                                    decoder_input_ids=encoding_decoder['input_ids'],
                                                    do_sample=True,
                                                    temperature=max(0.1, args.temperature),
                                                    max_length=args.max_len,
                                                    num_return_sequences=args.num_seqs_per_iter,
                                                    decoder_start_token_id=tokenizer.pad_token_id,
                                                    eos_token_id=tokenizer.eos_token_id,
                                                    top_p=0.95)
                    else:
                        gen_tokens = model.generate(**encoding,
                                                    do_sample=True,
                                                    temperature=args.temperature,
                                                    max_length=args.max_len,
                                                    num_return_sequences=args.num_seqs_per_iter,
                                                    eos_token_id=tokenizer.eos_token_id,
                                                    top_p=0.95)
                # 生成逻辑中添加 beam search 分支
                if args.decoding_style == 'beam_search':
                    if prompt_to_decoder:
                        gen_tokens = model.generate(**encoding,
                                                    decoder_input_ids=encoding_decoder['input_ids'],
                                                    do_sample=False,  # 关闭采样
                                                    num_beams=args.num_beams,
                                                    length_penalty=args.length_penalty,  # 惩罚过短/过长序列
                                                    max_length=args.max_len,
                                                    num_return_sequences=1,  # 束搜索返回最优1个
                                                    decoder_start_token_id=tokenizer.pad_token_id,
                                                    eos_token_id=tokenizer.eos_token_id)
                    else:
                        gen_tokens = model.generate(**encoding,
                                                    do_sample=False,
                                                    num_beams=args.num_beams,
                                                    length_penalty=args.length_penalty,
                                                    max_length=args.max_len,
                                                    num_return_sequences=1,
                                                    eos_token_id=tokenizer.eos_token_id)

            if gen_tokens is not None:
                if prompt_to_decoder:
                    gen_tokens = gen_tokens[:, encoding_decoder['input_ids'].shape[-1]:]
                gen_seqs = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)
            else:
                gen_seqs = None

            if gen_seqs is not None:
                assert len(ids_batch) == 1
                task_id = ids_batch[0]

                def is_valid_syntax(code):
                    """检查代码是否符合Python语法"""
                    try:
                        ast.parse(code)
                        return True
                    except SyntaxError:
                        return False

                # 在生成循环中添加过滤
                for seq_idx, gen_seq in enumerate(gen_seqs):
                    completion_seq = gen_seq
                    # ... 现有停止序列截断逻辑 ...

                    # 拼接完整代码并检查语法
                    all_code = prompt.replace('\t', '    ') + completion_seq
                    if not is_valid_syntax(all_code):
                        continue  # 跳过语法错误的生成结果

                    completion_seqs.append({
                        'task_id': task_id,
                        'completion': completion_seq,
                        'all_code': all_code
                    })

        print("Saving results to {}".format(output_file))
        write_jsonl(output_file, completion_seqs)


if __name__ == '__main__':
    main()
