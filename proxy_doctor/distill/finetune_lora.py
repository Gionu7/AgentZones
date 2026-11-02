"""Step 3 (GPU box) - LoRA fine-tune a small open-weight model on the SFT data.

    pip install -e '.[distill]'
    python -m proxy_doctor.distill.finetune_lora --data data/sft.jsonl \\
        --base Qwen/Qwen2.5-1.5B-Instruct --out models/proxy-doctor-1.5b

Then serve it on laptops with Ollama (see proxy_doctor/distill/Modelfile):
    python llama.cpp/convert_hf_to_gguf.py models/proxy-doctor-1.5b/merged --outtype q8_0 \\
        --outfile models/proxy-doctor-1.5b.gguf
    ollama create proxy-doctor -f proxy_doctor/distill/Modelfile
and set ``model.model: proxy-doctor`` in zone.yaml.

Also runnable as an Azure ML command job (see docs/DISTILLATION.md).
"""

from __future__ import annotations

import argparse


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/sft.jsonl")
    ap.add_argument("--base", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--out", default="models/proxy-doctor-1.5b")
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=1536)
    args = ap.parse_args()

    import torch
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model
    from transformers import (AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq, Trainer,
                              TrainingArguments)

    tok = AutoTokenizer.from_pretrained(args.base)
    tok.pad_token = tok.pad_token or tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32)
    model = get_peft_model(model, LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05,
                                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                                             "gate_proj", "up_proj", "down_proj"],
                                             task_type="CAUSAL_LM"))
    model.print_trainable_parameters()

    def encode(ex):
        prompt = tok.apply_chat_template(ex["messages"][:-1], tokenize=False, add_generation_prompt=True)
        full = prompt + ex["messages"][-1]["content"] + tok.eos_token
        p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
        ids = tok(full, add_special_tokens=False, truncation=True, max_length=args.max_len)["input_ids"]
        labels = [-100] * min(len(p_ids), len(ids)) + ids[len(p_ids):]  # learn the answer only
        return {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": labels}

    ds = load_dataset("json", data_files=args.data)["train"].map(encode, remove_columns=["messages"])
    ds = ds.train_test_split(test_size=0.05, seed=0)
    trainer = Trainer(
        model=model,
        args=TrainingArguments(output_dir=args.out, num_train_epochs=args.epochs, learning_rate=args.lr,
                               per_device_train_batch_size=args.batch, gradient_accumulation_steps=2,
                               logging_steps=20, eval_strategy="epoch", save_strategy="epoch",
                               bf16=torch.cuda.is_available(), report_to=[]),
        train_dataset=ds["train"], eval_dataset=ds["test"],
        data_collator=DataCollatorForSeq2Seq(tok, padding=True),
    )
    trainer.train()
    merged = model.merge_and_unload()
    merged.save_pretrained(f"{args.out}/merged")
    tok.save_pretrained(f"{args.out}/merged")
    print(f"merged model saved to {args.out}/merged")


if __name__ == "__main__":
    main()
