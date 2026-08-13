# Martingale Verification under the Heston Model

Fourier pricing, quadratic-exponential simulation, and vectorized pathwise repricing.

Under a risk-neutral measure with $r = q = 0$, both the stock price and the continuation
value of every European claim must be martingales. This project verifies that numerically
for the Heston model: it prices six European-type claims at $t = 0$ by SINC Fourier
inversion, simulates the joint $(S, \nu)$ state on a monthly grid, and reprices every claim
from every simulated state to check that

$$\begin{aligned}
\mathbb{E}[S_t] &= S_0 \\
\mathbb{E}[V_O(t, S_t, \nu_t)] &= O(S_0, T)
\end{aligned}$$

holds at each monitoring date, within Monte Carlo error.

**Result:** across all seven quantities and all eleven intermediate dates, the largest
deviation from the time-0 benchmark is **1.3 standard errors**, comfortably inside the
1.96 threshold for a 5% two-sided test.

(heston_outputs/martingale_diagnostics.png)

*Each panel: green crosses are Monte Carlo means with ±1 standard error at each monthly
date; the red line is the time-0 benchmark. Flat and centred means the martingale property
holds.*

---

## What makes this non-trivial

**Repricing without nested Monte Carlo.** The obvious way to value a claim at an
intermediate state is to simulate fresh inner paths from it. With 8192 outer paths and 11
dates, even a modest $10^3$ inner paths means $\sim 9.0 \times 10^7$ trajectories, which
benchmarks at roughly 41 minutes and it contaminates the estimator with inner-sampling
noise.

Instead this implementation exploits the affine form of the Heston characteristic function.
Since

$$\log \varphi(k, \tau, \nu_t) = A(k, \tau) - \nu_t B(k, \tau)$$

the coefficients $A$ and $B$ depend only on the Fourier mode and the residual maturity,
never on the path. They are computed once per mode as complex scalars and broadcast across
all 8192 variance states at once, collapsing the SINC loop from $O(N \times M)$ scalar
evaluations to $O(M)$ array operations. The whole run finishes in **6.24 s**, about **490×**
faster than the nested alternative, and returns exact continuation values rather than noisy
estimates.

**The Feller condition is violated.** With $2\lambda\bar{\nu}/\eta^2 \approx 0.1614$, the
variance process reaches zero with non-negligible probability, and a naive discretization
would push it negative. The variance path is therefore simulated with Andersen's
quadratic-exponential scheme, and the spot with the exact conditional scheme of
Broadie-Kaya given the integrated variance.

**A rejected optimization, documented.** The number of SINC modes needed for convergence
scales as $\sim 1/\tau$ (255 terms at $\tau = 0.92$, rising to 2445 at $\tau = 0.08$), so
scaling the Fourier cutoff as $X_c\sqrt{\tau}$ looks attractive and does cut total work by
46%. It also biases the put continuation value low by 5.9% ($\approx 2.8\sigma$) at the
shortest maturity, so it is implemented but not used. Section 4.8 of the report has the
numbers.

---

## Claims priced

| Claim | Definition | Time-0 price |
|---|---|---|
| Call | $\mathbb{E}[(S_T - K)^+]$ | 0.082958 |
| Put | $\mathbb{E}[(K - S_T)^+]$ | 0.082958 |
| $P_{cn}$ | $\mathbb{E}[\mathbf{1}_{\{K > S_T\}}]$ | 0.392848 |
| $P_{an}$ | $\mathbb{E}[S_T\cdot \mathbf{1}_{\{K > S_T\}}]$ | 0.388460 |
| $C_{cn}$ | $\mathbb{E}[\mathbf{1}_{\{K < S_T\}}]$ | 0.607152 |
| $C_{an}$ | $\mathbb{E}[S_T\cdot \mathbf{1}_{\{K < S_T\}}]$ | 0.811540 |

$P$ and $C$ denote put and call-side claims; the subscripts $cn$ and $an$ denote
cash-or-nothing and asset-or-nothing digitals. Put equals Call by parity (at-the-money,
$r = q = 0$), and the digitals are asymmetric about one half because of the leverage
effect ($\rho = -0.645$).

---

## Reproducibility

Requires **Python 3.10 or newer** (the type annotations use PEP 604 union syntax).

```bash
pip install -r requirements.txt
python3 heston_martingales.py
```

Defaults reproduce every figure in the report exactly (seed 29283). Results print to the
terminal; the plot and a CSV summary are written to `./heston_outputs/`.

Useful flags:

```bash
python3 heston_martingales.py --help           # full option list
python3 heston_martingales.py -n 15            # 2^15 paths instead of 2^13
python3 heston_martingales.py -xcmode sqrt_tau # maturity-scaled Fourier cutoff
python3 heston_martingales.py -out ./results   # alternative output directory
```

Model parameters, numerical settings and their justification are in Section 2 of the report.

---

## Repository Structure

```
heston_martingales.py           main script: pricing, simulation, martingale test
heston_martingale_report.pdf    full write-up: derivations, results, discussion
CFLib/                          course library (see NOTICE.md)
heston_outputs/
 martingale_diagnostics.png     7-panel diagnostic figure
 heston_summary.csv             MC means, standard errors, benchmarks
requirements.txt
```

---

## Report

[**Full report (PDF)**](heston_martingale_report.pdf) — model and pricing identities,
methodology, monthly diagnostics for all seven quantities, computational cost analysis,
Fourier cutoff sensitivity, limitations, and references.

---

## License

MIT — see [`LICENSE`](LICENSE). This applies to everything **except** the contents of
`CFLib/`, which are the property of Prof. Pietro Rossi and are redistributed here with
attribution only; see [`CFLib/NOTICE.md`](CFLib/NOTICE.md).

---

## Attribution

`CFLib/` is the teaching library of **Prof. Pietro Rossi** (University of Bologna), included
unmodified and trimmed to the eight modules this project imports. It is **not** my work —
see [`CFLib/NOTICE.md`](CFLib/NOTICE.md). Its SINC pricing method is published in Baschetti,
Bormetti, Romagnoli & Rossi, *The SINC way: a fast and accurate approach to Fourier pricing*,
Quantitative Finance 22(3), 2022.

Everything else is my own work, produced as a coursework project for the M.Sc. in
Quantitative Finance at the University of Bologna.

---

**Mahed Ali** · [mahed.ali@hotmail.com](mailto:mahed.ali@hotmail.com)
