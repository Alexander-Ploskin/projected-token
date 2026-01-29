from __future__ import annotations

import math
import torch
import time
import re
import os
from typing import Optional
from accelerate.utils import tqdm
from accelerate.logging import get_logger

from src.models.factory import build_tokenizer, build_xrag_model
from src.encoders.hf_encoder import HFMeanPoolRetriever
from src.runners.checkpoint_manager import CheckpointManager
from src.train.utils import (
    build_optimizer,
    build_scheduler,
    save_checkpoint,
    load_projector_checkpoint,
)
from src.distributed.utils import mean_across_processes
from src.train.objectives import BaseObjective


logger = get_logger(__name__)


class TrainRunner:
    def __init__(self, cfg, accelerator, tracker, *, stage_name: str, objective: BaseObjective, build_dataloaders_fn, validate_fn=None):
        self.cfg = cfg
        self.acc = accelerator
        self.tracker = tracker
        self.stage_name = stage_name
        self.objective = objective
        self.build_dataloaders_fn = build_dataloaders_fn
        self.validate_fn = validate_fn

        self.checkpoint_manager = CheckpointManager(
            output_dir=cfg.train.output_dir,
            save_total_limit=cfg.train.save_total_limit,
            best_metric_higher_better=False  # Lower metric (e.g. PPL) is better
        )

    def run(self) -> None:
        cfg, acc = self.cfg, self.acc

        # --- run header ---
        logger.info(
            f"Starting {self.stage_name} run | distributed={acc.state.distributed_type} "
            f"world_size={acc.num_processes} rank={acc.process_index} local_rank={acc.local_process_index} "
            f"mixed_precision={acc.mixed_precision}",
            main_process_only=False,
        )  # accelerate logger supports main_process_only [web:802]

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

        # --- optional projector init ---
        projector_checkpoint_path = getattr(cfg.projector, "checkpoint", None)
        if projector_checkpoint_path is not None:
            t0 = time.time()
            logger.info(f"Loading projector checkpoint: {projector_checkpoint_path}", main_process_only=True)
            load_projector_checkpoint(model, projector_checkpoint_path, projector_ckpt_name=cfg.train.projector_ckpt_name)
            logger.info(f"Projector checkpoint loaded in {time.time()-t0:.2f}s", main_process_only=True)

        # --- resumption state ---
        initial_step = 0
        if cfg.train.resume_step is not None:
            initial_step = cfg.train.resume_step
        elif projector_checkpoint_path is not None:
            # Try to parse from path e.g. runs/pretrain/checkpoint-100 or runs/pretrain/checkpoint-100/projector.pt
            match = re.search(r"checkpoint-(\d+)", projector_checkpoint_path)
            if match:
                initial_step = int(match.group(1))

        if initial_step > 0:
            logger.info(f"Resumption configured | starting from step {initial_step}", main_process_only=True)

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
        train_loader, dev_loader = self.build_dataloaders_fn(cfg, tokenizer, retriever_tokenizer=retriever_tokenizer)
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

        t_start = time.time()
        progress = tqdm(
            total=max_steps,
            desc=self.stage_name,
            main_process_only=True,
        )

        global_step = 0
        
        # Calculate resumption epoch and step within epoch
        resume_epoch = 0
        resume_step_in_epoch = 0
        if initial_step > 0:
            resume_epoch = initial_step // steps_per_epoch
            resume_step_in_epoch = initial_step % steps_per_epoch
            
            logger.info(f"Resumption: skipping to epoch {resume_epoch} step {resume_step_in_epoch}", main_process_only=True)
            logger.info(f"Fast-forwarding scheduler and progress to global step {initial_step}...", main_process_only=True)
            for _ in range(initial_step):
                scheduler.step()
            progress.update(initial_step)
            global_step = initial_step

        rolling = {"loss": 0.0, "nll": 0.0, "kl": 0.0, "nll_sum": 0.0, "ntokens": 0.0}
        eval_rolling = {"nll_sum": 0.0, "ntokens": 0.0}

        for epoch in range(cfg.train.num_train_epochs):
            if epoch < resume_epoch:
                continue

            model.train()
            logger.info(f"Epoch {epoch+1}/{cfg.train.num_train_epochs} start", main_process_only=True)

            # Efficiently skip batches in the current epoch if needed
            if epoch == resume_epoch and resume_step_in_epoch > 0:
                batches_to_skip_in_epoch = resume_step_in_epoch * cfg.distributed.gradient_accumulation_steps
                logger.info(f"Skipping first {batches_to_skip_in_epoch} micro-batches in this epoch...", main_process_only=True)
                active_loader = acc.skip_first_batches(train_loader, batches_to_skip_in_epoch)
                logger.info(f"Active loader (after skipping) has {len(active_loader)} batches and will be ready in a few minutes", main_process_only=True)
            else:
                active_loader = train_loader

            for batch in active_loader:
                with acc.accumulate(model):
                    retrieval_kwargs = {}
                    if retriever is not None:
                        retrieval_kwargs["retrieval_embeds"] = retriever.encode(
                            batch["retriever_input_ids"], batch["retriever_attention_mask"]
                        )

                    student_outputs = model(
                        input_ids=batch["xrag_input_ids"],
                        attention_mask=batch["xrag_attention_mask"],
                        **retrieval_kwargs,
                    )

                    teacher_outputs = None
                    if cfg.objective.get("alpha_kl", 0.0) > 0.0:
                        with torch.no_grad():
                            model.eval()
                            teacher_outputs = model(
                                input_ids=batch["input_ids"],
                                attention_mask=batch["attention_mask"],
                            )
                            model.train()

                    bundle = self.objective.compute(
                        cfg=cfg,
                        acc=acc,
                        model=model,
                        batch=batch,
                        student_outputs=student_outputs,
                        teacher_outputs=teacher_outputs,
                    )

                    acc.backward(bundle.loss)
                    if acc.sync_gradients and cfg.train.clip_grad_norm and cfg.train.clip_grad_norm > 0:
                        acc.clip_grad_norm_(model.parameters(), cfg.train.clip_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                    
                    # Accumulate metrics across micro-batches
                    for k, v in bundle.logs.items():
                        if k in rolling:
                            rolling[k] += v
                        if k in eval_rolling:
                            eval_rolling[k] += v

                if not acc.sync_gradients:
                    continue

                global_step += 1

                elapsed = time.time() - t_start
                steps_done = max(global_step, 1)
                sec_per_step = elapsed / steps_done
                steps_left = max_steps - global_step
                eta_sec = steps_left * sec_per_step

                examples_seen = global_step * global_batch
                examples_left = steps_left * global_batch

                if acc.is_main_process:
                    progress.update(1)
                    progress.set_postfix({
                        "loss": f"{bundle.logs.get('loss', float(bundle.loss.item())):.4f}",
                        "lr": f"{scheduler.get_last_lr()[0]:.2e}",
                        "seen_ex": f"{examples_seen/1e6:.2f}M",
                        "left_ex": f"{examples_left/1e6:.2f}M",
                        "eta_min": f"{eta_sec/60:.1f}",
                    })

                # periodic scalar logs
                if global_step % cfg.logging.log_every_steps == 0 and acc.is_main_process:
                    denom = float(cfg.logging.log_every_steps) * cfg.distributed.gradient_accumulation_steps
                    to_log = {
                        "learning_rate": scheduler.get_last_lr()[0],
                        "train_loss": rolling["loss"] / denom,
                    }
                    if "nll" in rolling and rolling["nll"] > 0:
                        to_log["train_nll"] = rolling["nll"] / denom
                    if "kl" in rolling and rolling["kl"] > 0:
                        to_log["train_kl"] = rolling["kl"] / denom
                    self.tracker.log_metrics(to_log, step=global_step)
                    rolling = {"loss": 0.0, "nll": 0.0, "kl": 0.0, "nll_sum": 0.0, "ntokens": 0.0}

                # periodic eval
                if dev_loader is not None and self.validate_fn is not None and global_step % cfg.train.eval_every_steps == 0:
                    logger.info(
                        f"Running dev eval at step={global_step} on device={acc.device}... "
                        f"Model device={next(model.parameters()).device} "
                        f"Retriever device={next(retriever.parameters()).device if retriever is not None else 'N/A'}",
                        main_process_only=True
                    )
                    metric = self.validate_fn(acc, model, retriever, dev_loader)
                    if acc.is_main_process:
                        # Synchronize training metrics for accurate PPL
                        t_nll_sum = torch.tensor(eval_rolling["nll_sum"], device=acc.device)
                        t_ntokens = torch.tensor(eval_rolling["ntokens"], device=acc.device)
                        
                        # Use mean_across_processes and multiply by num_processes to get sum
                        t_nll_sum = mean_across_processes(acc, t_nll_sum) * acc.num_processes
                        t_ntokens = mean_across_processes(acc, t_ntokens) * acc.num_processes
                        
                        metrics_to_log = {"dev_ppl": metric}
                        if t_ntokens > 0:
                            train_ppl = math.exp(t_nll_sum.item() / t_ntokens.item())
                            metrics_to_log["train_ppl"] = train_ppl
                            logger.info(f"Train evaluation over last {cfg.train.eval_every_steps} steps | ppl={train_ppl:.4f}", main_process_only=True)
                        
                        self.tracker.log_metrics(metrics_to_log, step=global_step)
                        self.checkpoint_manager.save(acc, model, tokenizer, save_checkpoint, metric=metric)

                    else:
                        # Non-main processes also need to participate in synchronization
                        t_nll_sum = torch.tensor(eval_rolling["nll_sum"], device=acc.device)
                        t_ntokens = torch.tensor(eval_rolling["ntokens"], device=acc.device)
                        mean_across_processes(acc, t_nll_sum)
                        mean_across_processes(acc, t_ntokens)

                    logger.info(f"Dev eval done | ppl={metric:.4f}", main_process_only=True)
                    eval_rolling = {"nll_sum": 0.0, "ntokens": 0.0}

                # periodic checkpoint (save latest N)
                if global_step % cfg.train.checkpoint_every_steps == 0 and acc.is_main_process:
                    self.checkpoint_manager.save(acc, model, tokenizer, save_checkpoint, step=global_step)

                if global_step >= max_steps:
                    logger.info("Reached max_steps, stopping.", main_process_only=True)
                    break

            if global_step >= max_steps:
                break

        # --- final checkpoint ---
        if acc.is_main_process:
            progress.close()
            self.checkpoint_manager.save(acc, model, tokenizer, save_checkpoint, is_final=True)

        logger.info(f"{self.stage_name.capitalize()} run finished.", main_process_only=False)
