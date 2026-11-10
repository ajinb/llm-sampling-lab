# llm-sampling-lab

Temperature, top-k, top-p and min-p implemented from scratch in numpy, measured on a real small model, plus a repeatability test of greedy decoding.

Companion to [What Temperature Actually Does, Measured, and Why 0 Was Never Deterministic](https://cloudandsre.com/blog/llm-sampling-temperature-top-p-min-p-measured/).

## Results

SmolLM2-135M-Instruct, 20 prompts, 16 samples each, 64 tokens. "Tail-token rate" is the share of sampled tokens that the untempered model rated under 1%.

| Setting | Tail-token rate | Distinct openings |
|---|---:|---:|
| T=0.7 | 1.8% | 95.6% |
| T=1.0 | 11.4% | 99.1% |
| T=1.0, top-p 0.9 | 4.4% | 97.8% |
| T=1.0, min-p 0.1 | 0.1% | 97.2% |
| T=1.5 | 79.8% | 100% |
| T=1.5, top-p 0.9 | 67.3% | 100% |
| T=1.5, min-p 0.1 | 2.4% | 99.1% |

Greedy decoding was identical across repeat runs, padded batches and CPU versus Apple GPU for all 20 prompts at 128 tokens. The smallest gap between the top two tokens over 2,316 steps was 0.00041 in log-probability, about 11 times the largest CPU versus GPU noise (0.000036).

Raw results: [`results/`](results/).

## Run it

```bash
pip install -r requirements-ci.txt
pytest -q                          # samplers only, numpy

pip install torch transformers
python experiment.py               # sampling and determinism, downloads the model (about 270 MB)
python experiment.py --margins-only
```

## Use the samplers

```python
import numpy as np
import samplers

probs = samplers.filtered(logits, temperature=1.5, min_p=0.1)
token = np.random.default_rng().choice(probs.size, p=probs)
```

Temperature is applied first, then truncation, the same order Hugging Face transformers uses. Other servers may differ.

## Limits

This is a 135M-parameter model, so treat the absolute numbers as illustrative. The determinism test runs on one machine, so it can't reproduce the batch-size effects of a shared inference server.

## License

Apache-2.0
