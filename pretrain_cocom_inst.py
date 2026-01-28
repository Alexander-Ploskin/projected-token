import json
import math
import os
import random
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from accelerate import Accelerator
from datasets import load_dataset, Dataset as HFDataset
from datasets.iterable_dataset import IterableDataset as HFIterableDataset

from transformers import get_linear_schedule_with_warmup

from cocom.modeling_cocom import COCOM, COCOMConfig


def select_text_column(ds):
    # Works for both Dataset and IterableDataset; IterableDataset.map is lazy.
    def pick_text(ex):
        return {"text": ex.get("text") or ex.get("content") or ""}
    # remove_columns differs a bit across versions; safest is to just keep "text" in the output
    return ds.map(pick_text)    

class COCOMPretrainDataset(Dataset):
    def __init__(self, texts, model: COCOM, max_context_tokens: int = 128, stage: int = 1, accelerator=None):
        raw_model = accelerator.unwrap_model(model) if accelerator else model
        self.compr_tokenizer = raw_model.compr.tokenizer if getattr(raw_model, "compr", None) is not None else None
        self.decoder_tokenizer = raw_model.decoder_tokenizer
        self.max_context_tokens = max_context_tokens
        self.stage = stage

        self._source = texts
        if not hasattr(self._source, "__len__"):
            raise ValueError("This dataset requires a non-streaming HF Dataset (random access).")
        self._length = len(self._source)

    def __len__(self) -> int:
        return self._length

    def _get_text(self, i: int) -> str:
        ex = self._source[i]
        if isinstance(ex, dict):
            t = ex.get("text") or ex.get("content") or ""
        else:
            t = str(ex)
        return t.strip()

    def _passes_filter(self, t: str) -> bool:
        if not t:
            return False
        if self.compr_tokenizer is not None:
            enc = self.compr_tokenizer.encode(
                t, add_special_tokens=True, truncation=True, max_length=self.max_context_tokens
            )
            return len(enc) <= self.max_context_tokens
        else:
            text_with_tokens = (
                self.decoder_tokenizer.enc_token
                + self.decoder_tokenizer.bos_token
                + t
                + self.decoder_tokenizer.bos_token
            )
            enc = self.decoder_tokenizer.encode(
                text_with_tokens, add_special_tokens=False, truncation=True, max_length=self.max_context_tokens
            )
            return len(enc) <= self.max_context_tokens

    def __getitem__(self, idx: int) -> Dict[str, str]:
        # Retry a few times to avoid rare empty/bad rows without pre-scanning whole dataset
        for _ in range(8):
            t = self._get_text(idx)
            if self._passes_filter(t):
                if self.stage == 2:
                    tokens = self.decoder_tokenizer.encode(
                        t, add_special_tokens=False, truncation=True, max_length=self.max_context_tokens * 2
                    )
                    sp = len(tokens) // 2
                    x1 = self.decoder_tokenizer.decode(tokens[:sp], skip_special_tokens=True)
                    x2 = self.decoder_tokenizer.decode(tokens[sp:], skip_special_tokens=True)
                    return {"x1": x1, "x2": x2, "full_text": t}
                return {"text": t}
            idx = random.randrange(self._length)
        # If your corpus is very noisy, increase retries or raise
        return {"text": "Fallback text."}


import math
import torch
from typing import List

class COCOMPretrainCollator:
    def __init__(
        self, 
        model: COCOM, 
        max_context_tokens: int = 128, 
        stage: int = 1,
        max_decoder_tokens: int = 512,  # NEW: hard cap decoder length
        accelerator: Accelerator = None,
    ):
        raw_model = accelerator.unwrap_model(model) if accelerator else model
        # raw_model = model
        self.model = raw_model
        self.compr_tokenizer = raw_model.compr.tokenizer if raw_model.compr is not None else None
        self.decoder_tokenizer = raw_model.decoder_tokenizer
        self.compr_rate = raw_model.compr_rate
        self.max_context_tokens = max_context_tokens
        self.max_decoder_tokens = max_decoder_tokens
        self.stage = stage
        self.is_qwen = "qwen" in model.config.decoder_model_name.lower()
        self.use_inst_format = not self.is_qwen

    def __call__(self, batch: List[Dict[str, str]]) -> Dict[str, torch.Tensor]:
        if self.stage == 1:
            texts = [item["text"] for item in batch]
            return self._collate_stage1(texts)
        else:
            x1_list = [item["x1"] for item in batch]
            x2_list = [item["x2"] for item in batch]
            return self._collate_stage2(x1_list, x2_list)

    def _create_qwen_chat_prompt(self, system_msg: str, user_msg: str) -> str:
        if system_msg:
            return f"<|im_start|>system\n{system_msg}<|im_end|>\n<|im_start|>user\n{user_msg}<|im_end|>\n<|im_start|>assistant\n"
        else:
            return f"<|im_start|>user\n{user_msg}<|im_end|>\n<|im_start|>assistant\n"

    def _collate_stage1(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        # Encoder unchanged
        if self.compr_tokenizer is not None:
            enc_inputs = self.compr_tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self.max_context_tokens,
                return_tensors="pt",
                pad_to_multiple_of=self.compr_rate,
            )
            seq_len = enc_inputs["input_ids"].size(1)
            num_mem_tokens = math.ceil(seq_len / self.compr_rate)
        else:
            contexts_with_tokens = [
                self.decoder_tokenizer.enc_token
                + self.decoder_tokenizer.bos_token
                + t
                + self.decoder_tokenizer.bos_token
                for t in texts
            ]
            enc_inputs = self.decoder_tokenizer(
                contexts_with_tokens,
                truncation=True,
                return_tensors="pt",
                padding="longest",
                max_length=self.max_context_tokens,
            )
            seq_len = enc_inputs["input_ids"].size(1)
            num_mem_tokens = math.ceil((seq_len - 3) / self.compr_rate)
            mem_tokens = torch.full(
                (enc_inputs["input_ids"].size(0), num_mem_tokens),
                self.decoder_tokenizer.mem_token_id,
                dtype=torch.long,
            )
            enc_inputs["input_ids"] = torch.cat([mem_tokens, enc_inputs["input_ids"]], dim=1)
            enc_inputs["attention_mask"] = torch.cat(
                [torch.ones_like(mem_tokens), enc_inputs["attention_mask"]],
                dim=1,
            )
        batch_size = len(texts)
        mem_tokens_str = self.decoder_tokenizer.mem_token * num_mem_tokens

        decoder_inputs: List[torch.Tensor] = []
        labels_list: List[torch.Tensor] = []

        for text in texts:
            if self.use_inst_format:
                instruction = (
                    "[INST] Reconstruct the following text from compressed context embeddings: "
                    + mem_tokens_str
                    + " [/INST] "
                    + text
                )
                instruction_part = (
                    "[INST] Reconstruct the following text from compressed context embeddings: "
                    + mem_tokens_str
                    + " [/INST] "
                )
                # FIXED: Add truncation + max_length
                instruction_tokenized = self.decoder_tokenizer(
                    instruction_part,
                    return_tensors="pt",
                    add_special_tokens=False,
                    truncation=True,
                    max_length=self.max_decoder_tokens,
                )
                instruction_len = instruction_tokenized["input_ids"].size(1)

                full_instruction_with_bos = self.decoder_tokenizer.bos_token + instruction
            else:
                user_message = f"Reconstruct the following text from compressed context embeddings: {mem_tokens_str}"
                instruction = self._create_qwen_chat_prompt("", user_message) + text
                
                instruction_part = self._create_qwen_chat_prompt("", user_message)
                # FIXED: Add truncation + max_length
                instruction_tokenized = self.decoder_tokenizer(
                    instruction_part,
                    return_tensors="pt",
                    add_special_tokens=False,
                    truncation=True,
                    max_length=self.max_decoder_tokens,
                )
                instruction_len = instruction_tokenized["input_ids"].size(1)
                
                full_instruction_with_bos = instruction

            # FIXED: Add max_length cap here too
            full_tokenized = self.decoder_tokenizer(
                full_instruction_with_bos,
                truncation=True,
                return_tensors="pt",
                padding=False,
                add_special_tokens=False,
                max_length=self.max_decoder_tokens,
            )
            input_ids = full_tokenized["input_ids"].squeeze(0)

            if self.is_qwen and self.decoder_tokenizer.eos_token_id is not None:
                input_ids = torch.cat([input_ids, torch.tensor([self.decoder_tokenizer.eos_token_id])])

            labels = [-100] * instruction_len + input_ids[instruction_len:].tolist()

            if len(labels) < len(input_ids):
                labels += [-100] * (len(input_ids) - len(labels))
            elif len(labels) > len(input_ids):
                labels = labels[: len(input_ids)]

            decoder_inputs.append(input_ids)
            labels_list.append(torch.tensor(labels, dtype=torch.long))

        max_dec_len = max(len(ids) for ids in decoder_inputs) if decoder_inputs else 1
        dec_input_ids = torch.zeros((batch_size, max_dec_len), dtype=torch.long)
        dec_attention_mask = torch.zeros((batch_size, max_dec_len), dtype=torch.long)
        labels = torch.full((batch_size, max_dec_len), -100, dtype=torch.long)

        for i, (ids, lab) in enumerate(zip(decoder_inputs, labels_list)):
            seq_len = len(ids)
            dec_input_ids[i, :seq_len] = ids
            dec_attention_mask[i, :seq_len] = 1
            labels[i, :seq_len] = lab

        return {
            "enc_input_ids": enc_inputs["input_ids"],
            "enc_attention_mask": enc_inputs["attention_mask"],
            "dec_input_ids": dec_input_ids,
            "dec_attention_mask": dec_attention_mask,
            "labels": labels,
        }

    def _collate_stage2(self, x1_list: List[str], x2_list: List[str]) -> Dict[str, torch.Tensor]:
        # Encoder unchanged
        if self.compr_tokenizer is not None:
            enc_inputs = self.compr_tokenizer(
                x1_list,
                padding=True,
                truncation=True,
                max_length=self.max_context_tokens,
                return_tensors="pt",
                pad_to_multiple_of=self.compr_rate,
            )
            seq_len = enc_inputs["input_ids"].size(1)
            num_mem_tokens = math.ceil(seq_len / self.compr_rate)
        else:
            contexts_with_tokens = [
                self.decoder_tokenizer.enc_token
                + self.decoder_tokenizer.bos_token
                + x1
                + self.decoder_tokenizer.bos_token
                for x1 in x1_list
            ]
            enc_inputs = self.decoder_tokenizer(
                contexts_with_tokens,
                truncation=True,
                return_tensors="pt",
                padding="longest",
                max_length=self.max_context_tokens,
            )
            seq_len = enc_inputs["input_ids"].size(1)
            num_mem_tokens = math.ceil((seq_len - 3) / self.compr_rate)
            mem_tokens = torch.full(
                (enc_inputs["input_ids"].size(0), num_mem_tokens),
                self.decoder_tokenizer.mem_token_id,
                dtype=torch.long,
            )
            enc_inputs["input_ids"] = torch.cat([mem_tokens, enc_inputs["input_ids"]], dim=1)
            enc_inputs["attention_mask"] = torch.cat(
                [torch.ones_like(mem_tokens), enc_inputs["attention_mask"]],
                dim=1,
            )

        batch_size = len(x1_list)
        mem_tokens_str = self.decoder_tokenizer.mem_token * num_mem_tokens

        decoder_inputs: List[torch.Tensor] = []
        labels_list: List[torch.Tensor] = []

        for x1, x2 in zip(x1_list, x2_list):
            if self.use_inst_format:
                instruction = (
                    "[INST] Continue the following text based on compressed context embeddings: "
                    + mem_tokens_str
                    + " [/INST] "
                    + x1
                    + " "
                    + x2
                )
                instruction_prefix = (
                    "[INST] Continue the following text based on compressed context embeddings: "
                    + mem_tokens_str
                    + " [/INST] "
                )
                # FIXED: Add truncation + max_length
                instruction_prefix_tokenized = self.decoder_tokenizer(
                    instruction_prefix,
                    return_tensors="pt",
                    add_special_tokens=False,
                    truncation=True,
                    max_length=self.max_decoder_tokens,
                )
                instruction_prefix_len = instruction_prefix_tokenized["input_ids"].size(1)

                x1_with_prefix = instruction_prefix + x1 + " "
                x1_with_bos = self.decoder_tokenizer.bos_token + x1_with_prefix
                # FIXED: Add truncation + max_length
                x1_tokenized = self.decoder_tokenizer(
                    x1_with_bos,
                    return_tensors="pt",
                    add_special_tokens=False,
                    truncation=True,
                    max_length=self.max_decoder_tokens,
                )
                x1_end_len = x1_tokenized["input_ids"].size(1)

                full_instruction_with_bos = self.decoder_tokenizer.bos_token + instruction
            else:
                user_message = f"Continue the following text based on compressed context embeddings: {mem_tokens_str}"
                instruction = self._create_qwen_chat_prompt("", user_message) + x1 + " " + x2
                
                instruction_prefix = self._create_qwen_chat_prompt("", user_message)
                # FIXED: Add truncation + max_length
                instruction_prefix_tokenized = self.decoder_tokenizer(
                    instruction_prefix,
                    return_tensors="pt",
                    add_special_tokens=False,
                    truncation=True,
                    max_length=self.max_decoder_tokens,
                )
                instruction_prefix_len = instruction_prefix_tokenized["input_ids"].size(1)

                x1_with_prefix = instruction_prefix + x1 + " "
                # FIXED: Add truncation + max_length
                x1_tokenized = self.decoder_tokenizer(
                    x1_with_prefix,
                    return_tensors="pt",
                    add_special_tokens=False,
                    truncation=True,
                    max_length=self.max_decoder_tokens,
                )
                x1_end_len = x1_tokenized["input_ids"].size(1)
                
                full_instruction_with_bos = instruction

            # FIXED: Add max_length cap here too
            full_tokenized = self.decoder_tokenizer(
                full_instruction_with_bos,
                truncation=True,
                return_tensors="pt",
                padding=False,
                add_special_tokens=False,
                max_length=self.max_decoder_tokens,
            )
            input_ids = full_tokenized["input_ids"].squeeze(0)

            if self.is_qwen and self.decoder_tokenizer.eos_token_id is not None:
                input_ids = torch.cat([input_ids, torch.tensor([self.decoder_tokenizer.eos_token_id])])

            labels = [-100] * x1_end_len + input_ids[x1_end_len:].tolist()

            if len(labels) < len(input_ids):
                labels += [-100] * (len(input_ids) - len(labels))
            elif len(labels) > len(input_ids):
                labels = labels[: len(input_ids)]

            decoder_inputs.append(input_ids)
            labels_list.append(torch.tensor(labels, dtype=torch.long))

        max_dec_len = max(len(ids) for ids in decoder_inputs) if decoder_inputs else 1
        dec_input_ids = torch.zeros((batch_size, max_dec_len), dtype=torch.long)
        dec_attention_mask = torch.zeros((batch_size, max_dec_len), dtype=torch.long)
        labels = torch.full((batch_size, max_dec_len), -100, dtype=torch.long)

        for i, (ids, lab) in enumerate(zip(decoder_inputs, labels_list)):
            seq_len = len(ids)
            dec_input_ids[i, :seq_len] = ids
            dec_attention_mask[i, :seq_len] = 1
            labels[i, :seq_len] = lab

        return {
            "enc_input_ids": enc_inputs["input_ids"],
            "enc_attention_mask": enc_inputs["attention_mask"],
            "dec_input_ids": dec_input_ids,
            "dec_attention_mask": dec_attention_mask,
            "labels": labels,
        }




from datasets import load_dataset, Dataset, IterableDataset

def load_finewiki_dataset(
    data_path: Optional[str] = None,
    max_docs: Optional[int] = None,
    streaming: bool = True,
) -> IterableDataset | Dataset:
    """
    Load dataset from:
      - local JSONL/JSON file if data_path provided
      - HF 'HuggingFaceFW/finewiki' otherwise.

    Returns a datasets Dataset or IterableDataset with a 'text' column.
    """
    if data_path and os.path.exists(data_path):
        if data_path.endswith(".jsonl") or data_path.endswith(".json"):
            # Local JSON/JSONL using datasets, no manual json.loads
            ds = load_dataset(
                "json",
                data_files={"train": data_path},
                split="train",
                streaming=streaming,
            )
        else:
            # Fallback: use your old JSON logic if you really have non-JSONL
            raise ValueError(f"Unsupported corpus format: {data_path}")
    else:
        # HF Finewiki fallback
        try:
            print("Loading Finewiki from HuggingFace: HuggingFaceFW/finewiki, ...")
            ds = load_dataset(
                "HuggingFaceFW/finewiki",
                split="train",
                streaming=streaming,
            )
        except Exception as e:
            print(f"Could not load Finewiki from HF: {e}")
            print("Using dummy texts for testing; please provide a real corpus for training.")
            # Small in‑memory fallback – safe
            return ["This is a dummy text for COCOM pretraining."] * 1000

    # Normalize to have a 'text' field
    def _ensure_text(example):
        if "text" in example and isinstance(example["text"], str):
            return example
        if "content" in example and isinstance(example["content"], str):
            example["text"] = example["content"]
            return example
        # drop non-string rows by returning empty text
        example["text"] = ""
        return example

    ds = ds.map(_ensure_text)

    if max_docs is not None and not isinstance(ds, IterableDataset):
        # Only possible for non-streaming Dataset
        ds = ds.select(range(min(max_docs, len(ds))))

    return ds


def train_stage(
    model: COCOM,
    train_texts: List[str],
    accelerator: Accelerator,
    stage: int,
    num_epochs: int,
    batch_size: int,
    learning_rate: float,
    warmup_ratio: float,
    weight_decay: float,
    max_context_tokens: int,
    gradient_accumulation_steps: int,
    save_steps: int,
    logging_steps: int,
    output_dir: str,
) -> COCOM:
    accelerator.print("=" * 80)
    accelerator.print(f"Stage {stage}: {'Auto-encoding with Context Embeddings' if stage == 1 else 'Language Modeling from Context Embeddings'}")
    accelerator.print("=" * 80)

    print('load dataset')
    dataset = COCOMPretrainDataset(
        texts=train_texts,
        model=model,
        max_context_tokens=max_context_tokens,
        stage=stage,
        accelerator=accelerator,  # NEW: pass it here
    )
    print('load collator')
    collator = COCOMPretrainCollator(
        model=model,
        max_context_tokens=max_context_tokens,
        stage=stage,
        max_decoder_tokens=512,
        accelerator=accelerator,  # NEW: pass it here too
    )

    print('dataloader')
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=4,
        pin_memory=True,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    num_update_steps_per_epoch = max(1, len(dataloader) // gradient_accumulation_steps)
    num_training_steps = num_update_steps_per_epoch * num_epochs
    num_warmup_steps = int(num_training_steps * warmup_ratio)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    model, optimizer, dataloader, scheduler = accelerator.prepare(
        model, optimizer, dataloader, scheduler
    )

    accelerator.print(f"Dataset size: {len(dataset)}")
    accelerator.print(f"Training for {num_epochs} epoch(s)")
    accelerator.print(f"Total updates: {num_training_steps}, warmup steps: {num_warmup_steps}")
    accelerator.print(f"Batch size: {batch_size}, Gradient accumulation: {gradient_accumulation_steps}")
    accelerator.print("=" * 80)

    global_step = 0
    running_loss = 0.0

    for epoch in range(num_epochs):
        model.train()

        if accelerator.is_main_process:
            progress_bar = tqdm(dataloader, desc=f"Stage {stage} - Epoch {epoch + 1}/{num_epochs}")
        else:
            progress_bar = dataloader

        for step, batch in enumerate(progress_bar):
            outputs = model(
                enc_input_ids=batch["enc_input_ids"],
                enc_attention_mask=batch["enc_attention_mask"],
                dec_input_ids=batch["dec_input_ids"],
                dec_attention_mask=batch["dec_attention_mask"],
                labels=batch["labels"],
            )

            loss = outputs["loss"] / gradient_accumulation_steps
            accelerator.backward(loss)

            running_loss += outputs["loss"].item()

            if (step + 1) % gradient_accumulation_steps == 0:
                accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                global_step += 1

                if global_step % logging_steps == 0:
                    avg_loss = running_loss / (logging_steps * gradient_accumulation_steps)
                    accelerator.print(f"Stage {stage} - Step {global_step}: loss = {avg_loss:.4f}")
                    if accelerator.is_main_process:
                        if hasattr(progress_bar, "set_postfix"):
                            progress_bar.set_postfix({"loss": avg_loss})
                    running_loss = 0.0

                if global_step % save_steps == 0 and accelerator.is_main_process:
                    accelerator.wait_for_everyone()
                    unwrapped = accelerator.unwrap_model(model)
                    ckpt_dir = os.path.join(output_dir, f"stage{stage}_checkpoint-{global_step}")
                    os.makedirs(ckpt_dir, exist_ok=True)
                    unwrapped.save_pretrained(ckpt_dir)
                    accelerator.print(f"Saved checkpoint at step {global_step} -> {ckpt_dir}")

        if accelerator.is_main_process:
            accelerator.wait_for_everyone()
            unwrapped = accelerator.unwrap_model(model)
            epoch_dir = os.path.join(output_dir, f"stage{stage}_epoch-{epoch + 1}")
            os.makedirs(epoch_dir, exist_ok=True)
            unwrapped.save_pretrained(epoch_dir)
            accelerator.print(f"Saved checkpoint at end of epoch {epoch + 1} -> {epoch_dir}")

    accelerator.print(f"Stage {stage} training completed!")
    return model


def pretrain_cocom_two_stage(
    model_config: COCOMConfig,
    train_texts: List[str],
    output_dir: str = "./cocom_pretrain_checkpoints",
    num_epochs_stage1: int = 1,
    num_epochs_stage2: int = 1,
    batch_size: int = 1,
    learning_rate: float = 1e-4,
    warmup_ratio: float = 0.05,
    weight_decay: float = 0.1,
    max_context_tokens: int = 128,
    gradient_accumulation_steps: int = 1,
    save_steps: int = 2000,
    logging_steps: int = 100,
    save_path: Optional[str] = None,
) -> None:
    accelerator = Accelerator()
    accelerator.print("=" * 80)
    accelerator.print("COCOM Two-Stage Pretraining")
    accelerator.print("Stage 1: Auto-encoding with Context Embeddings")
    accelerator.print("Stage 2: Language Modeling from Context Embeddings")
    accelerator.print("=" * 80)
    accelerator.print("Initializing COCOM model...")

    model = COCOM(model_config)
    model.train()

    accelerator.print("\n" + "=" * 80)
    accelerator.print("Starting Stage 1: Auto-encoding with Context Embeddings")
    accelerator.print("=" * 80)
    model = train_stage(
        model=model,
        train_texts=train_texts,
        accelerator=accelerator,
        stage=1,
        num_epochs=num_epochs_stage1,
        batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        max_context_tokens=max_context_tokens,
        gradient_accumulation_steps=gradient_accumulation_steps,
        save_steps=save_steps,
        logging_steps=logging_steps,
        output_dir=output_dir,
    )

    accelerator.print("\n" + "=" * 80)
    accelerator.print("Starting Stage 2: Language Modeling from Context Embeddings")
    accelerator.print("=" * 80)
    model = train_stage(
        model=model,
        train_texts=train_texts,
        accelerator=accelerator,
        stage=2,
        num_epochs=num_epochs_stage2,
        batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        max_context_tokens=max_context_tokens,
        gradient_accumulation_steps=gradient_accumulation_steps,
        save_steps=save_steps,
        logging_steps=logging_steps,
        output_dir=output_dir,
    )

    if accelerator.is_main_process:
        accelerator.wait_for_everyone()
        unwrapped = accelerator.unwrap_model(model)
        
        final_dir = os.path.join(output_dir, "final")
        os.makedirs(final_dir, exist_ok=True)
        unwrapped.save_pretrained(final_dir)
        accelerator.print(f"\nSaved final model -> {final_dir}")
        
        if save_path:
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
            unwrapped.save_pretrained(save_path)
            accelerator.print(f"Saved final model to specified path -> {save_path}")
        
        accelerator.print("=" * 80)
        accelerator.print("Two-stage pretraining completed!")
        accelerator.print("=" * 80)


def main() -> None:
    """
    Example (multi-GPU with accelerate):
        accelerate launch pretrain_cocom_inst.py \\
          --decoder_model Qwen/Qwen2.5-1.5B-Instruct \\
          --compr_model bert-base-uncased \\
          --compr_rate 64 \\
          --max_docs 500000 \\
          --save_path ./pretrained_cocom_model
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Two-stage pretraining for COCOM on Finewiki"
    )
    
    parser.add_argument(
        "--decoder_model",
        type=str,
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="Decoder model name.",
    )
    parser.add_argument(
        "--compr_model",
        type=str,
        default="bert-base-uncased",
        help="Compressor model name (BERT-based). Use 'none' for decoder-based compression.",
    )
    parser.add_argument(
        "--compr_rate",
        type=int,
        default=64,
        help="Compression rate.",
    )
    parser.add_argument(
        "--quantization",
        type=str,
        default="int4",
        choices=["no", "int4", "int8"],
        help="Quantization mode for the decoder.",
    )
    parser.add_argument(
        "--lora",
        action="store_true",
        help="Use LoRA adapters for the decoder during pretraining.",
    )
    parser.add_argument(
        "--lora_r",
        type=int,
        default=16,
        help="LoRA rank.",
    )
    parser.add_argument(
        "--training_form",
        type=str,
        default="both",
        choices=["compressor", "both"],
        help="Which components to train: compressor only or both compressor+decoder.",
    )

    parser.add_argument(
        "--corpus_path",
        type=str,
        default=None,
        help="Optional path to a local corpus file (JSON/JSONL). "
             "If omitted, uses HuggingFaceFW/finewiki from Hugging Face.",
    )
    parser.add_argument(
        "--max_docs",
        type=int,
        default=None,
        help="Maximum number of documents to load for pretraining.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="./cocom_pretrain_checkpoints",
        help="Directory for pretraining checkpoints.",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default=None,
        help="Path to directory to save the final pretrained model.",
    )
    parser.add_argument(
        "--num_epochs_stage1",
        type=int,
        default=1,
        help="Number of epochs for Stage 1 (auto-encoding).",
    )
    parser.add_argument(
        "--num_epochs_stage2",
        type=int,
        default=1,
        help="Number of epochs for Stage 2 (language modeling).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Per-device batch size.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate.",
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.05,
        help="Warmup ratio for the LR scheduler.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.1,
        help="Weight decay.",
    )
    parser.add_argument(
        "--max_context_tokens",
        type=int,
        default=128,
        help="Maximum number of compressor tokens per document.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Gradient accumulation steps.",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=2000,
        help="Steps between checkpoints.",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=100,
        help="Steps between logging.",
    )

    args = parser.parse_args()

    compr_model_name = args.compr_model
    if compr_model_name.lower() in ["none", "null", "decoder"]:
        compr_model_name = None

    cfg = COCOMConfig(
        decoder_model_name=args.decoder_model,
        quantization=args.quantization,
        generation_top_k=1,
        sep=False,
        compr_model_name=compr_model_name,
        compr_rate=args.compr_rate,
        compr_linear_type="concat",
        lora=args.lora,
        training_form=args.training_form,
        lora_r=args.lora_r,
    )

    print("Loading corpus...")
    ds = load_finewiki_dataset(
        args.corpus_path,
        max_docs=args.max_docs,
        streaming=False,  # Arrow + mmap; better for random access & shuffling
    )

    pretrain_cocom_two_stage(
        model_config=cfg,
        train_texts=ds,
        output_dir=args.output_dir,
        num_epochs_stage1=args.num_epochs_stage1,
        num_epochs_stage2=args.num_epochs_stage2,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        max_context_tokens=args.max_context_tokens,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        save_path=args.save_path,
    )


if __name__ == "__main__":
    main()