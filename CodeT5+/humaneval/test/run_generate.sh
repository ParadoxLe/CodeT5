model="/root/autodl-tmp/CodeT5+/humaneval/train/saved_models/codet5p-770M-native/final_checkpoint"
temp=0.2
max_len=800
pred_num=200
num_seqs_per_iter=25 # 25 for 350M and 770M, 10 for 2B, 8 for 6B, 2 for 16B on A100-40G


output_path="preds/codet5p-770M-native_T${temp}_N${pred_num}"  # 简化输出路径

mkdir -p ${output_path}
echo 'Output path: '$output_path
echo 'Model to eval: '$model

# 164 problems,  164 per GPU if GPU=1
index=0
gpu_num=1
for ((i = 0; i < $gpu_num; i++)); do
  start_index=0
  end_index=164

  gpu=$((i))
  echo 'Running process #' ${i} 'from' $start_index 'to' $end_index 'on GPU' ${gpu}
  ((index++))
  # 子进程命令：正确转义换行符
(
  CUDA_VISIBLE_DEVICES=$gpu \
  PYTHONPATH="/root/autodl-tmp/CodeT5+/humaneval:$PYTHONPATH" \
  python generate_codet5p.py --model "${model}" \
    --start_index ${start_index} \
    --end_index ${end_index} \
    --temperature ${temp} \
    --num_seqs_per_iter ${num_seqs_per_iter} \
    --N ${pred_num} \
    --max_len ${max_len} \
    --output_path ${output_path}
) &
  if (($index % $gpu_num == 0)); then wait; fi
done