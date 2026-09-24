# score.rs: from label log-probabilities to typed answers

- `softmax` over the labels only (restricted), at a temperature.
- `Calibration` (the conditioning): a temperature per bucket, the bucket
  being the question kind and option count, fitted by `xks condition` or
  `xks replay --fit` to minimise log loss on labelled cases.
- Confidence, as TypeSafe computes it in its MIT-licensed
  `system-one-adapter` (`_utils/confidence_metrics.py`), with its test
  cases carried over as Rust tests:
  - Choice: the peak probability scaled from uniform to certainty,
    `(p_max - 1/n) / (1 - 1/n)`. The docs' three-option demo,
    `(3 p_max - 1) / 2`, is this formula.
  - Score: concentration around the modal level,
    `1 - E|level - mode| / MAD(uniform)`, so a near miss costs less than a
    far one. An earlier version used the Choice formula for Score too; the
    adapter's test `[0.01, 0.02, 0.07, 0.3, 0.6] -> 0.55` is what caught it.
- `expected_level`: a Score's value, which can land between levels.
