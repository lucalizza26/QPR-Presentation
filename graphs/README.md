# QPR presentation graphs

One self-contained Python script per figure in the deck, so you can tweak each
one independently and drop the PNG back into PowerPoint.

## Files

| Script | Slide | Figure |
|--------|-------|--------|
| `slide03_reliability_drift.py`   | 3  | Conservative prediction vs. observed in-orbit (the systematic gap) |
| `slide04_zero_failures.py`       | 4  | Zero failures ≠ zero failure rate (posterior with a usable upper bound) |
| `slide05_prior_data_posterior.py`| 5  | Prior × Data → Posterior |
| `slide07_more_evidence.py`       | 7  | More evidence → tighter posterior (conjugate update) |
| `slide08_one_component.py`       | 8  | One component, start to finish (727 FIT, 90% CI [129, 1725]) |
| `slide09_prior_sensitivity.py`   | 9  | Prior matters when data is sparse; washes out as evidence grows |
| `slide10_system_montecarlo.py`   | 10 | RBD + Monte-Carlo system reliability distribution |
| `slide11_scale_tradeoff.py`      | 11 | Statistical certainty vs. design resolution across scales |
| `bayes_style.py`                 | —  | Shared palette, fonts, and helpers (imported by all of the above) |

## Setup

```bash
pip install numpy scipy matplotlib
```

## Run

From inside this `graphs/` folder:

```bash
python slide08_one_component.py
```

Each script pops up the figure **and** saves a transparent, high-res
`slideNN_*.png` next to it — ready to paste straight onto the slide.

## Tweaking

- **Colours / fonts / line weights** for the whole set live once in
  `bayes_style.py`. Change `TEAL`, `AMBER`, font size, etc. there and every
  figure follows. (`font.family` is `DejaVu Sans`; switch it to `"Arial"` if you
  want the figures to match Office exactly.)
- **The numbers behind each figure** live in a clearly marked `# knobs` block at
  the top of each script — prior strength `alpha`, prior mean, observed failures
  `k`, operating hours `T`, axis limits, annotation text/positions. Edit those;
  the maths (conjugate Gamma update, exponential reliability, Monte-Carlo
  propagation) updates automatically.
- `gamma_fit(alpha, beta_per_hour)` returns a scipy gamma frozen over the failure
  rate **in FIT** (1 FIT = 1e-9 / h), so `.mean()`, `.ppf(0.95)`, `.pdf(x)` all
  read in FIT directly.

All FIT values and credible intervals reproduce the numbers quoted in the
speaker script (e.g. slide 8: mean 727 FIT, 90% CI [129, 1725] FIT,
R: 0.940 → 0.956).
