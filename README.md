# Predictive Multiplicity in Ranking and Fairness Implications

## Reproducing the experiments

The project uses Python 3.12+ and [uv](https://docs.astral.sh/uv/).

Use `--help` with each script for available options.

### Finding fair top-K sets

1. **Preprocess datasets** (COMPAS, law school, astronauts, SAT/GPA, or HMDA):
   ```bash
   uv run preprocess.py hmda
   ```

2. **Run fairness-aware top-\(K\) optimization.** Example:
   ```bash
   uv run lp.py --dataset data/german_categorical-binsensitive.csv \
     --sens-attr age --score-attr month --k 40 --eps 0.1 \
     --solver glpk --save-csv results/example.csv
   ```
  Note that for using the MOSEK solver, you have to install it on your system first.

3. **Run all the jobs:**
   ```bash
   bash run.sh
   ```
## Synthetic Rashomon-set size experiment

```bash
uv run toy_rashomon_set_size.py
```

## Plotting

`plot.py` provides three plotting options:

- **`lp_results`**: Fairness improvement across different values of \(K\) and \(\epsilon\).
- **`score_distrib`**: Score distributions and group representation across top-\(K\) cutoffs.
- **`instability`**: Candidate-set composition across different values of \(\epsilon\).

Plot rendering requires a LaTeX installation for Matplotlib's text support.

## License

[GNU General Public License v3.0](LICENSE).
