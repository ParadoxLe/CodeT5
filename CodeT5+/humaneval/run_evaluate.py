import os
import subprocess
import argparse

def main():
    # 解析命令行参数（保持与原脚本的输出路径逻辑一致）
    parser = argparse.ArgumentParser(description="Process predictions and run evaluation")
    parser.add_argument(
        "--model",
        default="codet5p-220m",
        help="Model name (used in output path)"
    )
    parser.add_argument(
        "--temp",
        type=float,
        default=0.2,
        help="Temperature parameter (used in output path)"
    )
    parser.add_argument(
        "--pred_num",
        type=int,
        default=200,
        help="Number of predictions (used in output path)"
    )
    args = parser.parse_args()

    # 构建输出路径（与原脚本完全一致）
    output_path = f"preds/{args.model}_T{args.temp}_N{args.pred_num}"
    jsonl_path = f"{output_path}.jsonl"

    print(f"Output path: {output_path}")

    # 1. 调用process_preds.py处理预测结果
    print("Processing predictions with process_preds.py...")
    process_preds_cmd = [
        "python", "process_preds.py",
        "--path", output_path,
        "--out_path", jsonl_path
    ]
    # 执行处理命令，检查是否成功
    result = subprocess.run(
        process_preds_cmd,
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        print(f"Error in process_preds.py: {result.stderr}")
        return

    # 2. 调用evaluate_functional_correctness进行评估
    print("Running functional correctness evaluation...")
    eval_cmd = ["evaluate_functional_correctness", jsonl_path]
    # 执行评估命令，直接输出评估过程
    result = subprocess.run(eval_cmd)
    if result.returncode != 0:
        print("Evaluation failed!")
        return

    print("Evaluation completed successfully.")

if __name__ == "__main__":
    main()