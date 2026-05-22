# Stage B (HN) run on free GPUs 0,3,4 (Docker + tmux)

## Goal
- First generate teacher embeddings for `BAAI/bge-base-en-v1.5`.
- Then train with `configs/training/stage_b_distill_bge_base_hn.yaml`.
- Use `CUDA_VISIBLE_DEVICES=0,3,4`.
- Keep OSCAR split across two cards and run async validation on the third card.

## GPU mapping
- Logical `cuda:0` -> physical GPU `0` (main train device, OSCAR compressor).
- Logical `cuda:1` -> physical GPU `3` (OSCAR decoder).
- Logical `cuda:2` -> physical GPU `4` (async validation subprocess).

## Prerequisites
```bash
cd /home/a-ploskin/repos/ms-thesis/projected-token
docker ps --format '{{.Names}}' | grep '^pt-exp-mgpu$'
```

The runner expects the container to have:
- repository mounted at `/workspace/projected-token`
- `tmux` installed in container
- enough free disk space for `pip install -e .`

Dependencies are installed directly in the container python (no `venv`).

## Launch in tmux
```bash
cd /home/a-ploskin/repos/ms-thesis/projected-token
HF_TOKEN=<your_hf_token> \
./run_docker_tmux_bge_teacher_then_stage_b_hn.sh
```

Optional overrides:
```bash
CUDA_VISIBLE_DEVICES_VALUE=0,3,4 \
OSCAR_COMPRESSOR_DEVICE_VALUE=cuda:0 \
OSCAR_DECODER_DEVICE_VALUE=cuda:1 \
ASYNC_VALIDATION_DEVICE_VALUE=cuda:2 \
TEACHER_MIXED_DATASET_PATH=/data/mixed_train_dataset_stageb_small_hardneg.json \
TEACHER_OUTPUT_DIR=/data/teacher-embeddings \
TEACHER_OUTPUT_NAME=bge-base-en-v1.5_teacher_embeddings_mixed.h5 \
SESSION_NAME=stage_b_hn_with_bge_teacher \
INSTALL_DEPS=1 \
HF_TOKEN=<your_hf_token> \
./run_docker_tmux_bge_teacher_then_stage_b_hn.sh
```

Teacher embeddings are generated into `/data/teacher-embeddings` in container.
Because `/data` is mounted from host, the output is available on host at:
`data/teacher-embeddings/bge-base-en-v1.5_teacher_embeddings_mixed.h5`.

## Monitor
```bash
docker exec -it pt-exp-mgpu tmux attach -t stage_b_hn_with_bge_teacher
```

Run GPU monitor from host (or use existing watcher session):
```bash
nvidia-smi
```

```bash
ls -t artifacts/logs/stage_b_hn_with_bge_teacher_*.log | head -n 1
```

```bash
tail -f artifacts/logs/<latest-log>.log
```

## Success criteria
- Train process is running on logical `cuda:0`.
- Logs contain async-validation launch messages for `cuda:2`.
- `nvidia-smi` shows activity on physical GPUs `0`, `3`, `4`.
