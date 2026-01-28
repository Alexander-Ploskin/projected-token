# COCOM pretraining on Finewiki

Usage example:

```
accelerate config

accelerate launch pretrain_cocom_inst.py \
  --decoder_model Qwen/Qwen2.5-1.5B-Instruct \
  --compr_model none \
  --compr_rate 64 \
  --lora \
  --max_docs 500000 \
  --num_epochs_stage1 1 \
  --num_epochs_stage2 1 \
  --save_path ./pretrained_cocom_model
```
