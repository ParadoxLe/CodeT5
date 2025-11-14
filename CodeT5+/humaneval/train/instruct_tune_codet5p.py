import os
# 设置国内镜像加速模型下载
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import pprint
import argparse
import numpy as np
import copy
import torch
from datasets import load_dataset, load_from_disk
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    DataCollatorForSeq2Seq
)


# 指令模板（适配代码生成任务）
PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a code task, paired with an input that provides context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a code task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}


def get_model_size(model):
    """计算可训练参数数量（单位：M）"""
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    model_size = sum([np.prod(p.size()) for p in model_parameters])
    return f"{round(model_size / 1e+6)}M"


def freeze_decoder_except_xattn_codegen(model):
    """冻结解码器部分参数，只训练交叉注意力层（适合小模型加速训练）"""
    print(f'冻结前参数: 总参数 {model.num_parameters()}, 可训练参数 {get_model_size(model)}')
    for param in model.decoder.parameters():
        param.requires_grad = False  # 冻结整个解码器

    # 解冻交叉注意力层（CodeT5+ 解码器的关键层）
    num_decoder_layers = model.decoder.config.num_layers
    for i in range(num_decoder_layers):
        decoder_layer = model.decoder.block[i]
        if hasattr(decoder_layer, 'crossattention'):
            for param in decoder_layer.crossattention.parameters():
                param.requires_grad = True
            decoder_layer.crossattention.to(torch.float32)  # 保持高精度训练
        if hasattr(decoder_layer, 'alpha_xattn'):
            decoder_layer.alpha_xattn.requires_grad = True  # 解冻注意力权重参数
    print(f'冻结后参数: 总参数 {model.num_parameters()}, 可训练参数 {get_model_size(model)}')


def run_training(args, model, train_data, tokenizer):
    """运行训练（单卡原生训练，无 DeepSpeed）"""
    print(f"\n===== 开始训练主循环 =====")

    # 自定义日志回调
    class TrainingLogger(TrainerCallback):
        def on_epoch_begin(self, args, state, control, **kwargs):
            print(
                f"\n===================== Epoch {int(state.epoch) + 1}/{args.num_train_epochs} 开始 =====================")

        def on_epoch_end(self, args, state, control, **kwargs):
            avg_loss = state.log_history[-1]['loss'] if state.log_history else "N/A"
            print(
                f"===================== Epoch {int(state.epoch) + 1}/{args.num_train_epochs} 结束 =====================")
            print(f"当前步数: {state.global_step}, 平均损失: {avg_loss:.4f}\n")

        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step % args.logging_steps == 0:
                current_loss = state.log_history[-1]['loss'] if state.log_history else "N/A"
                # 移除学习率打印（避免访问 state.optimizer）
                print(f"Step {state.global_step}/{state.max_steps} | 损失: {current_loss:.4f}")

    # 训练参数配置（移除所有 DeepSpeed 相关项）
    training_args = TrainingArguments(
        output_dir=args.save_dir,
        overwrite_output_dir=False,

        # 训练核心参数
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size_per_replica,
        gradient_accumulation_steps=args.grad_acc_steps,

        # 优化器参数
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_steps=args.lr_warmup_steps,

        # 日志与保存
        logging_dir=os.path.join(args.save_dir, "logs"),
        logging_first_step=True,
        logging_steps=args.log_freq,
        save_strategy="epoch",
        save_total_limit=2,

        # 数据加载
        dataloader_drop_last=True,
        dataloader_num_workers=0,

        # 单卡训练配置（移除 deepspeed 参数）
        local_rank=args.local_rank,
        fp16=args.fp16,  # 保留原生 FP16 加速（非 DeepSpeed）
    )

    # 定义数据拼接器
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, pad_to_multiple_of=8)

    # 初始化 Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        callbacks=[TrainingLogger()],
        data_collator=data_collator
    )

    # 计算总训练步数（单卡版本）
    total_batch_size = training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps
    num_training_steps = (len(train_data) // total_batch_size) * training_args.num_train_epochs

    # 打印训练配置
    print(f"训练配置:")
    print(f"  总 epochs: {args.epochs}")
    print(f"  单卡批次: {args.batch_size_per_replica} | 梯度累积: {args.grad_acc_steps}")
    print(f"  总训练步数: {num_training_steps}")
    print(f"  原生 FP16 加速: {'启用' if training_args.fp16 else '禁用'}\n")

    # 检查训练数据设备
    sample = train_data[0]
    input_ids_device = sample["input_ids"].device if hasattr(sample["input_ids"], 'device') else "cpu"
    print(f"训练数据设备: {input_ids_device}")

    # 开始训练
    trainer.train()

    # 保存最终模型
    if args.local_rank in [0, -1]:
        final_checkpoint_dir = os.path.join(args.save_dir, "final_checkpoint")
        model.save_pretrained(final_checkpoint_dir)
        tokenizer.save_pretrained(final_checkpoint_dir)
        print(f'  ==> 训练完成，模型保存至 {final_checkpoint_dir}')


def load_tokenize_data(args, tokenizer):
    """加载并预处理数据（单卡模式）"""
    if os.path.exists(args.cache_data):
        print(f"  ==> 从缓存加载数据: {args.cache_data}")
        train_data = load_from_disk(args.cache_data)
        print(f"  ==> 缓存数据加载完成，共 {len(train_data)} 条样本")
        return train_data
    else:
        print(f"  ==> 缓存不存在，从原始数据加载: {args.instruct_data_path}")
        datasets = load_dataset('json', data_files=args.instruct_data_path)['train']
        print(f"  ==> 原始数据加载完成，共 {len(datasets)} 条样本，开始预处理...")

        def preprocess_function(examples):
            prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
            sources = [
                prompt_input.format_map({'instruction': i, 'input': j}) if j != ''
                else prompt_no_input.format_map({'instruction': i})
                for i, j in zip(examples["instruction"], examples["input"])
            ]
            targets = [f"{src}{output}{tokenizer.eos_token}" for src, output in zip(sources, examples["output"])]

            model_inputs = tokenizer(sources, max_length=args.max_len, padding="max_length", truncation=True)
            labels = tokenizer(targets, max_length=args.max_len, padding="max_length", truncation=True)

            model_inputs["decoder_input_ids"] = copy.deepcopy(labels["input_ids"])
            decoder_start_id = tokenizer.eos_token_id
            for z in model_inputs["decoder_input_ids"]:
                z[1:] = z[:-1]
                z[0] = decoder_start_id

            eos_token_id = tokenizer.eos_token_id
            for x, y in zip(model_inputs["input_ids"], labels["input_ids"]):
                prefix_len = x.index(eos_token_id) if eos_token_id in x else len(x)
                y[:prefix_len] = [-100] * prefix_len
                if eos_token_id in y:
                    pad_start = y.index(eos_token_id) + 1
                    y[pad_start:] = [-100] * (len(y) - pad_start)

            model_inputs["labels"] = labels["input_ids"]
            model_inputs["decoder_attention_mask"] = labels["attention_mask"]
            return model_inputs

        train_data = datasets.map(
            preprocess_function,
            batched=True,
            remove_columns=datasets.column_names,
            num_proc=0,  # 单进程预处理
            load_from_cache_file=False,
        )

        print(f"  ==> 数据预处理完成，共 {len(train_data)} 条样本")
        train_data.save_to_disk(args.cache_data)
        print(f"  ==> 数据缓存至 {args.cache_data}")
        return train_data


def main(args):
    # 打印参数配置
    argsdict = vars(args)
    print("训练参数:")
    print(pprint.pformat(argsdict))

    # 保存参数到文件
    os.makedirs(args.save_dir, exist_ok=True)
    with open(os.path.join(args.save_dir, "command.txt"), 'w') as f:
        f.write(pprint.pformat(argsdict))

    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.load)

    # 加载并预处理数据
    train_data = load_tokenize_data(args, tokenizer)
    if args.data_num != -1:
        train_data = train_data.select(range(args.data_num))
        print(f"  ==> 限制训练数据量为 {args.data_num} 条")

    # 加载模型（FP32 加载，配合原生 FP16 训练）
    print(f"  ==> 加载模型: {args.load}")
    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.load,
        dtype=torch.float32,
        low_cpu_mem_usage=True,
        trust_remote_code=False
    ).to("cuda" if torch.cuda.is_available() else "cpu")  # 强制移到 GPU

    # 冻结部分参数
    freeze_decoder_except_xattn_codegen(model)

    # 开始训练
    run_training(args, model, train_data, tokenizer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CodeT5p-770M 指令微调（单卡原生训练）")
    # 数据配置
    parser.add_argument('--data-num', default=-1, type=int, help="限制训练数据量（-1 为全部）")
    parser.add_argument('--max-len', default=512, type=int, help="最大序列长度")
    parser.add_argument('--instruct-data-path', default='code_alpaca_20k.json', type=str)
    parser.add_argument('--cache-data', default='cache_data/instructions_770m', type=str, help="数据缓存路径")
    parser.add_argument('--load', default='Salesforce/codet5p-770M', type=str, help="模型路径")

    # 训练配置（移除 DeepSpeed 相关参数）
    parser.add_argument('--epochs', default=3, type=int, help="训练轮数")
    parser.add_argument('--lr', default=5e-5, type=float, help="学习率")
    parser.add_argument('--lr-warmup-steps', default=100, type=int, help="预热步数")
    parser.add_argument('--batch-size-per-replica', default=4, type=int, help="单卡批次")
    parser.add_argument('--grad-acc-steps', default=2, type=int, help="梯度累积步数")
    parser.add_argument('--local_rank', default=-1, type=int, help="分布式标识（单卡固定为-1）")
    parser.add_argument('--fp16', default=True, action='store_true', help="启用原生 FP16 加速")

    # 日志与保存
    parser.add_argument('--save-dir', default="saved_models/codet5p-770M-native", type=str)
    parser.add_argument('--log-freq', default=10, type=int, help="日志输出频率")

    args = parser.parse_args()
    main(args)