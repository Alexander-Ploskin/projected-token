# projected-token

### Build & Run

```
docker build -t projected-token .
```

```
docker run -it --gpus all \
  --name projected-token-eval \
  -v ~/hf_cache:/workspace/data/hf_cache \
  -v $(pwd):/workspace \
  projected-token-image \
  --help
```
