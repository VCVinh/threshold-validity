# threshold-validity

Reproduction code for **"How Many Positive Cases Does Your Validation Set Need?"** —
a closed-form Beta-distribution result giving finite-sample guarantees on validation-set
sizing for diagnostic-AI sensitivity claims.

**Author**: Vinh C. Van (Independent Researcher)
**Paper**: CC BY 4.0 manuscript (artifact: https://claude.ai/code/artifact/9065a11b-936e-4da9-b7db-eb5ba2fd5ba5).
Code in this repository: AGPL-3.0, dual-licensed
(see [COMMERCIAL-LICENSE.md](COMMERCIAL-LICENSE.md)).

## What this gives you

Given a target operating point ("the model must hold 90% sensitivity with confidence c"),
the closed form answers the minimum number of **positive** cases a validation set must
contain — before anyone trains anything. The paper's Table 1 tabulates minimum-n by
target sensitivity and confidence level; consult the manuscript for the table itself
(numbers here are never restated without their derivation).

## Contents

```
threshold_validity.py          # main implementation incl. self_test() regression suite
independent_beta_check.py      # independent re-derivation cross-check of the closed form
outputs/                       # run outputs of the independent check
```

## Reproduction

```bash
python threshold_validity.py --self-test   # runs the built-in regression suite
python independent_beta_check.py           # independent numeric cross-check
```

Environment: Python ≥ 3.10, `numpy`, `scipy`. No data download, no credentials,
no GPU — everything is analytic or Monte-Carlo over the Beta distribution.

## Provenance notes

- The manuscript went through an adversarial review pass that caught and corrected an
  off-by-one in the defining equation and an unstated rounding convention; both fixes are
  visible in the published revision note.
- §3.3 of the manuscript contains a self-correction withdrawing an earlier arithmetic
  claim about a third-party study; it is retained in-body deliberately, as part of the
  measurement record.

## Licence

AGPL-3.0. Commercial licensing available — see [COMMERCIAL-LICENSE.md](COMMERCIAL-LICENSE.md).
Contributions subject to a CLA — see [CONTRIBUTING.md](CONTRIBUTING.md).
