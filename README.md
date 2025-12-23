# dinov3-clip

## Install

CPU-only (default):
```bash
pip install -e .
```

CUDA (Linux, choose the CUDA version you have):
```bash
pip install -e .[cuda] --index-url https://download.pytorch.org/whl/cu128
```

If you use uv:
```bash
uv pip install -e .[cuda] --index-url https://download.pytorch.org/whl/cu128
```
