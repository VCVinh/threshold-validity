"""ADVERSARIAL CHECK (item 7): independent re-derivation of Paper B's Beta
closed-form, NOT trusting threshold_validity.py's own self_test().

Three independent checks, none of which import threshold_validity.py's own
verification machinery:
  1. scipy.stats.beta as ground truth for the closed-form parameters claimed
     (Beta(n-k+1, k)) at n in {8, 30, 100, 169, 921} (the paper's own headline
     "minimum n" figures) and target 0.90.
  2. A from-scratch Monte Carlo (own RNG stream, own order-statistic logic,
     no shared code with threshold_validity.py) at the same n values.
  3. Cross-check against the module's own achieved_sensitivity_interval() to
     see if the module's numbers actually match the independent derivation
     -- this is the real test, not the module's internal self-test.
"""
from __future__ import annotations

import math
import sys
import numpy as np
from scipy.stats import beta as scipy_beta

# path-independent; requires only numpy/scipy
import threshold_validity as tv  # noqa: E402


def order_stat_k(n: int, target: float) -> int:
    """Independent re-derivation of which order statistic a target implies.

    k = round((1-target)*n), clamped to [1, n-1] -- same formula the paper
    states in Sec 2.1, re-derived here rather than imported.
    """
    return max(1, min(n - 1, round((1.0 - target) * n)))


def independent_monte_carlo(n: int, target: float, trials: int = 200_000,
                             seed: int = 12345) -> dict:
    """From-scratch MC: draw n Gaussian validation scores, take the k-th
    order statistic as tau, draw a fresh large evaluation sample, measure
    the fraction >= tau. Repeat `trials` times. Uses numpy's Generator
    (PCG64), independent of threshold_validity.py's random.Random.
    """
    rng = np.random.default_rng(seed)
    k = order_stat_k(n, target)
    achieved = np.empty(trials)
    EVAL_N = 5000
    for t in range(trials):
        val = np.sort(rng.normal(0.0, 1.0, size=n))
        tau = val[k - 1]
        ev = rng.normal(0.0, 1.0, size=EVAL_N)
        achieved[t] = np.mean(ev >= tau)
    return {
        "k": k,
        "mc_mean": float(achieved.mean()),
        "mc_lo": float(np.quantile(achieved, 0.025)),
        "mc_hi": float(np.quantile(achieved, 0.975)),
        "mc_sd": float(achieved.std(ddof=1)),
    }


def main() -> int:
    target = 0.90
    ns = [8, 30, 100, 169, 921]
    print(f"{'n':>5} {'k':>4} | {'scipy mean':>11} {'scipy lo':>9} {'scipy hi':>9} "
          f"| {'MC mean':>8} {'MC lo':>8} {'MC hi':>8} "
          f"| {'module mean':>12} {'module lo':>10} {'module hi':>10}")
    all_ok = True
    for n in ns:
        k = order_stat_k(n, target)
        a, b = n - k + 1, k
        # 1. scipy ground truth
        scipy_mean = scipy_beta.mean(a, b)
        scipy_lo = scipy_beta.ppf(0.025, a, b)
        scipy_hi = scipy_beta.ppf(0.975, a, b)

        # 2. independent Monte Carlo (fewer trials for large n to keep runtime sane)
        trials = 200_000 if n <= 100 else 60_000
        mc = independent_monte_carlo(n, target, trials=trials, seed=1000 + n)

        # 3. the module's own function
        mod = tv.achieved_sensitivity_interval(n, target)

        print(f"{n:>5} {k:>4} | {scipy_mean:>11.4f} {scipy_lo:>9.4f} {scipy_hi:>9.4f} "
              f"| {mc['mc_mean']:>8.4f} {mc['mc_lo']:>8.4f} {mc['mc_hi']:>8.4f} "
              f"| {mod['expected']:>12.4f} {mod['lo']:>10.4f} {mod['hi']:>10.4f}")

        # agreement checks
        ok_scipy_mod = (abs(scipy_mean - mod["expected"]) < 1e-3 and
                        abs(scipy_lo - mod["lo"]) < 5e-3 and
                        abs(scipy_hi - mod["hi"]) < 5e-3)
        ok_mc_scipy = (abs(mc["mc_mean"] - scipy_mean) < 0.02 and
                       abs(mc["mc_lo"] - scipy_lo) < 0.03 and
                       abs(mc["mc_hi"] - scipy_hi) < 0.03)
        if not ok_scipy_mod:
            print(f"      !! module vs scipy DISAGREE at n={n}")
            all_ok = False
        if not ok_mc_scipy:
            print(f"      !! independent MC vs scipy DISAGREE at n={n}")
            all_ok = False

    print(f"\nALL AGREE: {all_ok}")

    # separately verify the two headline "minimum n" claims from the abstract
    print("\n=== minimum-n claims (abstract) ===")
    for guarantee, claimed_n in [(0.85, 169), (0.88, 921)]:
        n = tv.minimum_positives(0.90, guarantee, 0.95)
        iv_at_n = tv.achieved_sensitivity_interval(n, 0.90, 0.95)
        iv_at_n_minus_1 = tv.achieved_sensitivity_interval(n - 1, 0.90, 0.95) if n > 2 else None
        lo_nm1 = iv_at_n_minus_1["lo"] if iv_at_n_minus_1 else float("nan")
        print(f"  guarantee>={guarantee}: module says n={n} (paper claims {claimed_n}) "
              f"-- lo at n={iv_at_n['lo']:.4f}, lo at n-1={lo_nm1:.4f}")
        # cross check with scipy directly
        k = order_stat_k(n, 0.90)
        a, b = n - k + 1, k
        scipy_lo_at_n = scipy_beta.ppf(0.025, a, b)
        print(f"    scipy lo at claimed n={n}: {scipy_lo_at_n:.4f} (must be >= {guarantee})")
        k_m1 = order_stat_k(n - 1, 0.90)
        a_m1, b_m1 = (n - 1) - k_m1 + 1, k_m1
        scipy_lo_at_nm1 = scipy_beta.ppf(0.025, a_m1, b_m1)
        print(f"    scipy lo at n-1={n-1}: {scipy_lo_at_nm1:.4f} (should be < {guarantee} "
              f"if n really is the MINIMUM)")

    # Table 1 reproduction check
    print("\n=== Table 1 reproduction (paper's own numbers) ===")
    paper_table1 = {
        8: (0.889, 0.631, 0.997),
        10: (0.909, 0.692, 0.997),
        20: (0.905, 0.751, 0.988),
        30: (0.903, 0.779, 0.979),
        50: (0.902, 0.808, 0.967),
        100: (0.901, 0.836, 0.951),
        200: (0.900, 0.856, 0.938),
        500: (0.900, 0.873, 0.925),
    }
    for n, (exp_claimed, lo_claimed, hi_claimed) in paper_table1.items():
        k = order_stat_k(n, 0.90)
        a, b = n - k + 1, k
        exp_scipy = scipy_beta.mean(a, b)
        lo_scipy = scipy_beta.ppf(0.025, a, b)
        hi_scipy = scipy_beta.ppf(0.975, a, b)
        match = (abs(exp_scipy - exp_claimed) < 0.002 and
                 abs(lo_scipy - lo_claimed) < 0.003 and
                 abs(hi_scipy - hi_claimed) < 0.003)
        print(f"  n={n:>4}: paper=({exp_claimed:.3f},[{lo_claimed:.3f},{hi_claimed:.3f}])  "
              f"scipy=({exp_scipy:.3f},[{lo_scipy:.3f},{hi_scipy:.3f}])  "
              f"{'MATCH' if match else 'MISMATCH!!'}")

    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
