import os
import subprocess
import shutil

def main():
    # ==================== 配置参数（对应原 bash 脚本中的变量）====================
    model = "codet5p-220m"  # 模型名称（可改为 codet5p-220m 等小模型）
    temp = 0.2                     # 温度参数
    max_len = 800                  # 生成代码最大长度
    pred_num = 200                 # 每个问题生成的样本数
    num_seqs_per_iter = 2          # 每次迭代生成的序列数（控制内存使用）
    gpu_num = 1                    # GPU 数量（根据你的设备修改，如 1 个 GPU）
    total_problems = 164           # HumanEval
    output_dir = f"preds/{model}_T{temp}_N{pred_num}"  # 输出目录

    # ==================== 创建输出目录 ====================
    if os.path.exists(output_dir):
        # 若目录已存在，可选择删除旧文件或保留（这里选择删除后重建）
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    print(f"输出目录已创建：{output_dir}")

    # ==================== 计算每个 GPU 处理的问题范围 ====================
    # 平均分配问题（最后一个 GPU 处理剩余部分）
    problems_per_gpu = total_problems // gpu_num
    processes = []  # 存储进程对象

    for gpu_idx in range(gpu_num):
        # 计算当前 GPU 处理的问题索引（左闭右开）
        start_idx = gpu_idx * problems_per_gpu
        # 最后一个 GPU 处理剩余所有问题
        end_idx = total_problems if gpu_idx == gpu_num - 1 else (gpu_idx + 1) * problems_per_gpu

        # 构造命令行参数（调用 generate_codet5p.py）
        cmd = [
            "python", "generate_codet5p.py",
            "--model", f"Salesforce/{model}",
            "--start_index", str(start_idx),
            "--end_index", str(end_idx),
            "--temperature", str(temp),
            "--num_seqs_per_iter", str(num_seqs_per_iter),
            "--N", str(pred_num),
            "--max_len", str(max_len),
            "--output_path", output_dir
        ]

        # 设置当前进程使用的 GPU（Windows 用 CUDA_VISIBLE_DEVICES，Linux 相同）
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)  # 指定 GPU 索引

        # 启动子进程（并行运行）
        print(f"启动 GPU {gpu_idx}：处理问题 {start_idx}-{end_idx}")
        process = subprocess.Popen(cmd, env=env)
        processes.append(process)

    # 等待所有进程完成
    for process in processes:
        process.wait()
        if process.returncode != 0:
            print(f"进程出错，返回码：{process.returncode}")

    print("所有生成任务完成！")

if __name__ == "__main__":
    main()