# Qwen2.5-VL smoke test

This output checks that the Qwen2.5-VL model can caption a tiny synthetic test video without returning the old `@@@` failure pattern.

Run this from this output directory to reproduce it:

```bash
./reproduce.sh
```

The script keeps durable result files here and stores the synthetic video plus extracted frames under `work/vl-smoke-test/`.

Expected result files:

- `captions.md`
- `captions.json`
- `reproduce.sh`
