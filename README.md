# FIONNA

This repository contains the Python implementation of Functional
Inference with Orthogonal Neural Network Adjustment (FIONNA).

## Method

`fionna.fit_fionna()` uses five-fold cross-fitting, training-fold FPCA for each
functional mediator, three-fold inner validation over a fixed ReLU architecture
grid, unregularized residualized-exposure projection, cross-validated ridge
stabilization for mediator-score coefficients, and influence-function standard
errors for the total, direct, and indirect effects. 

## Install

```bash
python -m pip install -r requirements.txt
```

## Fit FIONNA

The input object provides `Y`, `X`, `Z`, `grid`, and either `M` or `M_list`.
For four mediators with two retained scores per mediator:

```python
from fionna import fit_fionna

fit = fit_fionna(data, K=[2, 2, 2, 2], seed=20260702)
print(fit["theta_hat"], fit["se_theta"])
print(fit["alpha_hat"], fit["se_alpha"])
print(fit["delta_hat"], fit["se_delta"], fit["ci_delta"])
```

## Size examples

`examples/run_size_examples.py` contains two null designs from the completed
diagnostic run:

- linear confounding, one mediator, `n=400`, `K=5`;
- shared-profile confounding, four mediators, `n=400`, `K=2` per mediator.

Run a quick check with:

```bash
python examples/run_size_examples.py
```

The saved `R=200` evidence is in `results/size_examples_r200.csv`. Indirect-effect
coverage was 0.950 for the linear design and 0.935 for the shared-profile design.
Reproduce it with:

```bash
python examples/run_size_examples.py --replications 200 --workers 12
```

The generators and seed sequences are the same as those used for the saved
results.
