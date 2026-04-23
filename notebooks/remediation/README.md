# Remediation Experiments

These scripts implement the three remediation strategies evaluated in Chapter 5.4 of the dissertation.

## Files

- `per_subject_normalisation.py` — applies per-subject z-score normalisation before LOSO training, removing inter-subject amplitude differences. Results reported in Table 7.
- `dynamic_content_filter.py` — implements a post-hoc motion-energy filter that suppresses fall predictions on low-energy windows. Threshold sweep results reported in Appendix A.3 and Table 7.
- `subject_exclusion.py` — reruns LOSO with S10 excluded, then S10 and S13 excluded, to quantify how much these data-quality outliers depress mean F1. Results reported in Table 7.

## Prerequisites

Run `src/prepare_upfall.py` first to generate the windowed data. Each script loads from `prepared_upfall/` and saves results to `outputs/lstm/`.