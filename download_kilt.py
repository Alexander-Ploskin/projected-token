from datasets import load_dataset
import os

# Ensure the target directory exists
os.makedirs("/mnt/raid/a-ploskin", exist_ok=True)

print("Starting KILT dataset download to /mnt/raid/a-ploskin...")
# Use 'cache_dir' argument to explicitly set the cache location
ds = load_dataset("s-nlp/kilt", split="train", cache_dir="/mnt/raid/a-ploskin/hf_cache")
print("KILT dataset download complete.")
