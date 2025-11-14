# 明确指定已有的合并文件路径（你的 merged_results.jsonl）
merged_file=preds/merged_results.jsonl
# 评估结果保存路径（自定义）
eval_result_file=preds/eval_results.json

# 检查合并文件是否存在
if [ ! -f "$merged_file" ]; then
  echo "错误：合并文件 $merged_file 不存在！"
  exit 1
fi

echo "开始评估文件：$merged_file"

# 直接运行评估脚本，读取 merged_results.jsonl
PYTHONPATH="./CodeT5+/humaneval:$PYTHONPATH" \
python human_eval/evaluate_functional_correctness.py \
  "$merged_file" 

echo "评估完成，结果保存到：$eval_result_file"