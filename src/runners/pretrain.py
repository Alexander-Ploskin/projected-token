from __future__ import annotations

import os
import math
import time
import torch

from accelerate.utils import tqdm
from accelerate.logging import get_logger

from src.data.datamodule import build_pretrain_dataloaders
from src.models.factory import build_tokenizer, build_xrag_model
from src.encoders.hf_encoder import HFMeanPoolRetriever
from src.train import (
    build_optimizer,
    build_scheduler,
    nll_loss,
    validate_pretrain_ppl,
    save_checkpoint,
)

logger = get_logger(__name__)


class PretrainRunner:
    def __init__(self, cfg, accelerator, tracker):
        self.cfg = cfg
        self.accelerator = accelerator
        self.tracker = tracker

    def run(self) -> None:
        cfg = self.cfg
        acc = self.accelerator

        # --- run header ---
        logger.info(
            f"Starting pretrain run | distributed={acc.state.distributed_type} "
            f"world_size={acc.num_processes} rank={acc.process_index} local_rank={acc.local_process_index} "
            f"mixed_precision={acc.mixed_precision}",
            main_process_only=False,
        )  # accelerate state fields are standard [web:235]

        # --- tokenizer ---
        t0 = time.time()
        logger.info("Loading tokenizer + adding XRAG token...", main_process_only=True)
        tokenizer, xrag_token_id = build_tokenizer(cfg.model.model_name_or_path, cfg.model.xrag_token)
        logger.info(
            f"Tokenizer loaded in {time.time()-t0:.2f}s | vocab_size={len(tokenizer)} xrag_token_id={xrag_token_id}",
            main_process_only=True,
        )

        # --- retriever ---
        retriever = None
        retriever_tokenizer = None
        retriever_embed_dim = 0

        if cfg.retriever.retriever_name_or_path:
            t0 = time.time()
            logger.info(f"Loading retriever from {cfg.retriever.retriever_name_or_path} ...", main_process_only=True)
            retriever = HFMeanPoolRetriever(cfg.retriever.retriever_name_or_path, torch_dtype=torch.bfloat16)
            retriever_tokenizer = retriever.tokenizer
            retriever_embed_dim = retriever.embed_dim
            logger.info(
                f"Retriever loaded in {time.time()-t0:.2f}s | embed_dim={retriever_embed_dim}",
                main_process_only=True,
            )
        else:
            logger.warning("No retriever configured; running in 'no retrieval' mode.", main_process_only=True)

        # --- model ---
        t0 = time.time()
        logger.info(f"Loading LLM from {cfg.model.model_name_or_path} ...", main_process_only=True)
        model = build_xrag_model(
            model_name_or_path=cfg.model.model_name_or_path,
            xrag_token_id=xrag_token_id,
            retriever_embed_dim=retriever_embed_dim,
            bridge_hidden_dim=cfg.projector.hidden_dim,
            bridge_dropout=cfg.projector.dropout,
            use_flash_attn_2=cfg.model.use_flash_attn_2,
            torch_dtype=torch.bfloat16 if cfg.distributed.mixed_precision == "bf16" else "auto",
        )
        logger.info(f"Model loaded in {time.time()-t0:.2f}s", main_process_only=True)

        # --- freezing ---
        if cfg.model.freeze_llm:
            logger.info("Freezing LLM parameters (projector-only training).", main_process_only=True)
            model.freeze_llm()

        if retriever is not None and cfg.retriever.freeze_retriever:
            logger.info("Freezing retriever parameters.", main_process_only=True)
            for p in retriever.parameters():
                p.requires_grad = False
            retriever.eval()

        # trainable params
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        logger.info(
            f"Parameter counts | trainable={trainable:,} total={total:,} ({trainable/total:.4%})",
            main_process_only=True,
        )

        # --- data ---
        t0 = time.time()
        logger.info("Building dataloaders...", main_process_only=True)
        train_loader, dev_loader = build_pretrain_dataloaders(cfg, tokenizer, retriever_tokenizer=retriever_tokenizer)
        logger.info(
            f"Dataloaders built in {time.time()-t0:.2f}s | train_batches={len(train_loader)} "
            f"dev_batches={len(dev_loader) if dev_loader is not None else 0}",
            main_process_only=True,
        )

        # --- optimizer/scheduler ---
        logger.info("Building optimizer...", main_process_only=True)
        optimizer = build_optimizer(cfg, model)

        steps_per_epoch = math.ceil(len(train_loader) / cfg.distributed.gradient_accumulation_steps)
        max_steps = cfg.train.max_train_steps or (cfg.train.num_train_epochs * steps_per_epoch)
        logger.info(
            f"Scheduling | steps_per_epoch={steps_per_epoch} max_steps={max_steps} "
            f"grad_accum={cfg.distributed.gradient_accumulation_steps}",
            main_process_only=True,
        )

        logger.info("Building LR scheduler...", main_process_only=True)
        scheduler = build_scheduler(cfg, optimizer, num_training_steps=max_steps)

        # --- device placement ---
        if retriever is not None:
            retriever = retriever.to(acc.device)
            logger.info(f"Moved retriever to device={acc.device}", main_process_only=True)

        # --- accelerate prepare ---
        logger.info("Calling accelerator.prepare(...) ...", main_process_only=True)
        model, optimizer, train_loader, scheduler = acc.prepare(model, optimizer, train_loader, scheduler)
        if dev_loader is not None:
            dev_loader = acc.prepare(dev_loader)

        logger.info("accelerator.prepare(...) done.", main_process_only=True)

        if acc.is_main_process:
            self.tracker.log_text(f"Trainable params: {trainable:,}")

        # --- training loop ---
        logger.info(
            f"Training start | epochs={cfg.train.num_train_epochs} batch_size={cfg.train.per_device_train_batch_size}",
            main_process_only=True,
        )

        global_batch = (
            cfg.train.per_device_train_batch_size
            * acc.num_processes
            * cfg.distributed.gradient_accumulation_steps
        )

        steps_per_epoch = math.ceil(len(train_loader) / cfg.distributed.gradient_accumulation_steps)

        max_steps = cfg.train.max_train_steps

        t_start = time.time()
        progress_bar = tqdm(
            total=max_steps,
            desc="pretrain",
            main_process_only=True,   # only rank0 shows it [web:423]
        )

        global_step = 0
        for epoch in range(cfg.train.num_train_epochs):
            model.train()
            logger.info(f"Epoch {epoch+1}/{cfg.train.num_train_epochs} start", main_process_only=True)

            for batch_idx, batch in enumerate(train_loader):
                with acc.accumulate(model):
                    retrieval_embeds = None
                    if retriever is not None:
                        retrieval_embeds = retriever.encode(
                            batch["retriever_input_ids"], batch["retriever_attention_mask"]
                        )

                    out = model(
                        input_ids=batch["xrag_input_ids"],
                        attention_mask=batch["xrag_attention_mask"],
                        retrieval_embeds=retrieval_embeds,
                    )
                    loss = nll_loss(out.logits, batch["xrag_labels"])

                    acc.backward(loss)
                    if acc.sync_gradients and cfg.train.clip_grad_norm and cfg.train.clip_grad_norm > 0:
                        acc.clip_grad_norm_(model.parameters(), cfg.train.clip_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()

                if acc.sync_gradients:
                    global_step += 1

                    elapsed = time.time() - t_start
                    steps_done = max(global_step, 1)
                    sec_per_step = elapsed / steps_done
                    steps_left = max_steps - global_step
                    eta_sec = steps_left * sec_per_step

                    examples_seen = global_step * global_batch
                    examples_left = steps_left * global_batch

                    if acc.is_main_process:
                        progress_bar.update(1)
                        progress_bar.set_postfix({
                            "loss": f"{loss.item():.4f}",
                            "lr": f"{scheduler.get_last_lr()[0]:.2e}",
                            "seen_ex": f"{examples_seen/1e6:.2f}M",
                            "left_ex": f"{examples_left/1e6:.2f}M",
                            "eta_min": f"{eta_sec/60:.1f}",
                        })

                    # periodic scalar logs
                    if global_step % cfg.logging.log_every_steps == 0 and acc.is_main_process:
                        self.tracker.log_metrics({"train_loss": float(loss.item())}, step=global_step)

                    # periodic eval
                    if dev_loader is not None and global_step % cfg.train.eval_every_steps == 0:
                        logger.info(f"Running dev eval at step={global_step} ...", main_process_only=True)
                        ppl = validate_pretrain_ppl(acc, model, retriever, dev_loader)
                        if acc.is_main_process:
                            self.tracker.log_metrics({"dev_ppl": ppl}, step=global_step)
                        logger.info(f"Dev eval done | ppl={ppl:.4f}", main_process_only=True)

                    # periodic checkpoint
                    if global_step % cfg.train.checkpoint_every_steps == 0 and acc.is_main_process:
                        ckpt_dir = os.path.join(cfg.train.output_dir, f"step_{global_step}")
                        logger.info(f"Saving checkpoint: {ckpt_dir}", main_process_only=True)
                        save_checkpoint(acc, model, tokenizer, ckpt_dir)
                        logger.info("Checkpoint saved.", main_process_only=True)

                    if global_step >= max_steps:
                        logger.info("Reached max_steps, stopping.", main_process_only=True)
                        break

            if global_step >= max_steps:
                break

        # --- final checkpoint ---
        if acc.is_main_process:
            progress_bar.close()
            last_dir = os.path.join(cfg.train.output_dir, "last")
            logger.info(f"Saving final checkpoint: {last_dir}", main_process_only=True)
            save_checkpoint(acc, model, tokenizer, last_dir)
            logger.info("Final checkpoint saved.", main_process_only=True)

        logger.info("Prƒetrain run finished.", main_process_only=False)
