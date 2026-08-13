#!/usr/bin/env python3

"""
Martingale verification for the Heston stochastic volatility model.

Prices six European-type claims (call, put, cash-or-nothing and asset-or-nothing
digitals) at t=0 by SINC Fourier inversion, simulates the joint (S, nu) state on a
monthly grid with a Quadratic-Exponential CIR scheme, then reprices every claim
from each simulated state to verify numerically that

    E[S_t] = S_0
    E[V_O(t, S_t, nu_t)] = O(S_0, T)

Intermediate repricing is vectorised across paths by exploiting the affine form of
the Heston characteristic function, which avoids nested Monte Carlo entirely.

Author: Mahed Ali
Coursework project, MSc Quantitative Finance, University of Bologna.
"""

import cmath, csv, math, os, sys, time
from typing import Dict, List, Tuple
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from CFLib.config import get_input_parms
from CFLib.Heston import Heston
from CFLib.FT_opt import ft_opt
from CFLib.cir_evol import QT_cir_evol
from CFLib.heston_evol import mc_heston
from CFLib.stats import stats

# Default Parameters
DEFAULTS: Dict[str, float | int | str | bool] = {
    "So": 1.2,
    "K": 1.2,
    "T": 1.0,
    "r": 0.0,
    "q": 0.0,
    "lmbda": 1.03179746,
    "nubar": 0.07819284,
    "eta": 1.0,
    "nu_o": 0.04,
    "rho": -0.6450755,
    "nv": 13,
    "nm": 12,
    "dt": 1.0 / 730.0,
    "xc": 5.0,
    "xcmode": "fixed",
    "seed": 29283,
    "show": False,
}

OPT_KEYS = ["put", "call", "pCn", "pAn", "cCn", "cAn"]
OPT_LABELS = ["Put", "Call", "Pcn", "Pan", "Ccn", "Can"]
PLOT_ORDER = ["S", "Put", "Call", "Pcn", "Pan", "Ccn", "Can"]

# CLI Helpers and Usage
def usage() -> None:
    print("Usage: python heston_martingales.py [options]")
    print("")
    print("Contract parameters")
    print("  -So      initial spot                  (default 1.2)")
    print("  -K       strike                        (default 1.2)")
    print("  -T       maturity in years             (default 1.0)")
    print("")
    print("Heston parameters")
    print("  -lm      lambda                        (default 1.03179746)")
    print("  -th      nubar                         (default 0.07819284)")
    print("  -eta     eta                           (default 1.0)")
    print("  -nu      initial variance              (default 0.04)")
    print("  -rho     correlation                   (default -0.6450755)")
    print("")
    print("Numerical parameters")
    print("  -n       log2(number of paths)         (default 13)")
    print("  -nm      number of monitoring months   (default 12)")
    print("  -dt      variance integration step     (default 1/730)")
    print("  -xc      Fourier cutoff parameter      (default 5.0)")
    print("  -xcmode  fixed | sqrt_tau              (default fixed)")
    print("  -seed    random seed                   (default 29283)")
    print("")
    print("Output options")
    print("  -out     output directory              (default ./heston_outputs)")
    print("  --show   display plot interactively")
    print("  --help   show this help")

def parse_inputs(argv: List[str]) -> Dict[str, float | int | str | bool]:
    parms = get_input_parms(argv)

    if "help" in parms:
        usage()
        raise SystemExit(0)

    if "err" in parms:
        print(f"@ Error: {parms['err']}")
        usage()
        raise SystemExit(1)

    par = dict(DEFAULTS)
    par["So"] = float(parms.get("So", par["So"]))
    par["K"] = float(parms.get("K", par["K"]))
    par["T"] = float(parms.get("T", par["T"]))
    par["r"] = float(parms.get("r", par["r"]))
    par["q"] = float(parms.get("q", par["q"]))

    par["lmbda"] = float(parms.get("lm", par["lmbda"]))
    par["nubar"] = float(parms.get("th", par["nubar"]))
    par["eta"] = float(parms.get("eta", par["eta"]))
    par["nu_o"] = float(parms.get("nu", par["nu_o"]))
    par["rho"] = float(parms.get("rho", par["rho"]))

    par["nv"] = int(parms.get("n", par["nv"]))
    par["nm"] = int(parms.get("nm", par["nm"]))
    par["dt"] = float(parms.get("dt", par["dt"]))
    par["xc"] = float(parms.get("xc", par["xc"]))
    par["xcmode"] = str(parms.get("xcmode", par["xcmode"]))
    par["seed"] = int(parms.get("seed", par["seed"]))
    par["show"] = "show" in parms or bool(parms.get("show", False))

    if par["xcmode"] not in ("fixed", "sqrt_tau"):
        print("@ Error: -xcmode must be either 'fixed' or 'sqrt_tau'.")
        raise SystemExit(1)

    if abs(float(par["r"])) > 1.0e-14 or abs(float(par["q"])) > 1.0e-14:
        print("@ Error: this implementation assumes the driftless setting r = q = 0.")
        raise SystemExit(1)

    if "out" in parms:
        par["out_dir"] = str(parms["out"])
    else:
        base = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
        par["out_dir"] = os.path.join(base, "heston_outputs")

    return par

# Model
def build_model(par: Dict[str, float | int | str | bool]) -> Heston:
    return Heston(
        lmbda=float(par["lmbda"]),
        nubar=float(par["nubar"]),
        eta=float(par["eta"]),
        nu_o=float(par["nu_o"]),
        rho=float(par["rho"]),
    )

# Fourier cutoff
def x_cutoff(par: Dict[str, float | int | str | bool], tau: float) -> float:
    xc = float(par["xc"])
    if str(par["xcmode"]) == "sqrt_tau":
        return xc * math.sqrt(max(tau, 1.0e-16))
    return xc

# Vectorised Heston Fourier Pricing
def _cf_components(model: Heston, c_k: complex, tau: float) -> Tuple[complex, complex]:
    c_v = 1j * c_k
    c_L = 0.5 * c_v - 0.5 * c_v * c_v
    c_kappa = model.lmbda - model.rho * model.eta * c_v
    c_gmma = cmath.sqrt(c_kappa * c_kappa + 2.0 * c_L * model.eta ** 2)
    c_g = (c_gmma - c_kappa) / (c_gmma + c_kappa)
    c_zp = (c_gmma - c_kappa) / model.eta ** 2

    if model.lmbda == 0.0:
        c_A = 0.0 + 0.0j
    else:
        c_theta = (model.lmbda * model.nubar) / c_kappa
        c_c = 2.0 * cmath.log((1.0 + c_g) / (1.0 + c_g * cmath.exp(-c_gmma * tau))) / model.eta ** 2
        c_A = c_kappa * c_theta * (c_c - c_zp * tau)

    c_exp_gt = cmath.exp(-c_gmma * tau)
    c_B = c_zp * (1.0 - c_exp_gt) / (1.0 + c_g * c_exp_gt)
    return c_A, c_B

# Vectorised SINC approximation of P(X < w) for all paths.
def _pr_x_lt_w_vec(model: Heston, Xc: float, vW: np.ndarray, vNu: np.ndarray, off: complex, tau: float) -> np.ndarray:
    m = 1
    vTot = np.zeros(len(vW), dtype=float)

    while True:
        c_k = 2.0 * math.pi * (m / (2.0 * Xc) + off)
        c_A, c_B = _cf_components(model, c_k, tau)
        c_phi = np.exp(c_A - vNu * c_B)

        vTh = math.pi * m * vW / Xc
        vDelta = (np.cos(vTh) * c_phi.imag - np.sin(vTh) * c_phi.real) / m
        vTot += vDelta

        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(vTot != 0.0, np.abs(vDelta / vTot), np.inf)
        if ratio.max() < 1.0e-8:
            break
        m += 2

    return 0.5 - 2.0 * vTot / math.pi

# Prices are returned in normalised units
def ft_opt_vec(model: Heston, vKt: np.ndarray, vNu: np.ndarray, tau: float, Xc: float) -> Dict[str, np.ndarray]:
    if tau < 1.0e-12:
        pCn = np.where(vKt > 1.0, 1.0, 0.0)
        pAn = np.where(vKt > 1.0, 1.0, 0.0)
        put = np.maximum(vKt - 1.0, 0.0)
        call = np.maximum(1.0 - vKt, 0.0)
    else:
        vW = np.log(vKt)
        vNu = np.maximum(vNu, 0.0)
        pCn = _pr_x_lt_w_vec(model, Xc, vW, vNu, 0.0 + 0.0j, tau)
        pAn = _pr_x_lt_w_vec(model, Xc, vW, vNu, 0.0 - 1j / (2.0 * math.pi), tau)
        put = vKt * pCn - pAn
        call = put + (1.0 - vKt)

    cCn = 1.0 - pCn
    cAn = 1.0 - pAn
    return {"put": put, "call": call, "pCn": pCn, "pAn": pAn, "cCn": cCn, "cAn": cAn}

# Time-0 pricing and conditional repricing from intermediate states
def option_prices_t0(
    model: Heston,
    So: float,
    K: float,
    T: float,
    par: Dict[str, float | int | str | bool],
) -> Dict[str, float]:
    kT = K / So
    out = ft_opt(model, kT, T, x_cutoff(par, T))

    put = So * float(out["put"])
    call = So * float(out["call"])
    pCn = float(out["pCn"])
    pAn = So * float(out["pAn"])
    cCn = 1.0 - pCn
    cAn = So - pAn
    return {"put": put, "call": call, "pCn": pCn, "pAn": pAn, "cCn": cCn, "cAn": cAn}

# At time t_j the option value is a function of the current state (S_tj, nu_tj).
def option_prices_vec(
    model: Heston,
    Sn: np.ndarray,
    nu_n: np.ndarray,
    K: float,
    tau: float,
    par: Dict[str, float | int | str | bool],
) -> Dict[str, np.ndarray]:
    if tau < 1.0e-12:
        put = np.maximum(K - Sn, 0.0)
        call = np.maximum(Sn - K, 0.0)
        pCn = np.where(K > Sn, 1.0, 0.0)
        pAn = np.where(K > Sn, Sn, 0.0)
        cCn = 1.0 - pCn
        cAn = Sn - pAn
        return {"put": put, "call": call, "pCn": pCn, "pAn": pAn, "cCn": cCn, "cAn": cAn}

    kT = K / Sn
    out = ft_opt_vec(model, kT, nu_n, tau, x_cutoff(par, tau))

    put = Sn * out["put"]
    call = Sn * out["call"]
    pCn = out["pCn"]
    pAn = Sn * out["pAn"]
    cCn = 1.0 - pCn
    cAn = Sn - pAn
    return {"put": put, "call": call, "pCn": pCn, "pAn": pAn, "cCn": cCn, "cAn": cAn}

# Time grid and path simulation
def build_schedule(T: float, nm: int) -> np.ndarray:
    dt_monitor = T / nm
    return np.arange(0.0, nm * dt_monitor + dt_monitor / 2.0, dt_monitor)

# Simulate the monthly outer states used in the martingale test.
def simulate_outer_paths(
    model: Heston,
    par: Dict[str, float | int | str | bool],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    So = float(par["So"])
    rho = float(par["rho"])
    dt = float(par["dt"])
    n_paths = 1 << int(par["nv"])
    tf = build_schedule(float(par["T"]), int(par["nm"]))

    rand = np.random.RandomState(int(par["seed"]))
    vol, Ivol = QT_cir_evol(rand, model.cir, tf, dt, n_paths)

    S = np.zeros((len(tf), n_paths), dtype=float)
    for n in range(n_paths):
        path = mc_heston(rand, vol[:, n], Ivol[:, n], model.cir, rho, tf, 1)
        S[:, n] = So * path[:, 0]

    return tf, S, vol

# Monte Carlo standard error
def standard_error(std_dev: float, n: int) -> float:
    return std_dev / math.sqrt(n)

# Martingale test
def martingale_test(
    model: Heston,
    tf: np.ndarray,
    S: np.ndarray,
    vol: np.ndarray,
    par: Dict[str, float | int | str | bool],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    T = float(par["T"])
    K = float(par["K"])
    n_paths = S.shape[1]

    times = tf[1:-1].copy()
    Em_S = np.zeros(len(times), dtype=float)
    Err_S = np.zeros(len(times), dtype=float)
    Em_opt = {key: np.zeros(len(times), dtype=float) for key in OPT_KEYS}
    Err_opt = {key: np.zeros(len(times), dtype=float) for key in OPT_KEYS}

    for j, tj in enumerate(times, start=1):
        tau = T - tj
        Sn = S[j]
        nu_n = vol[j]

        mu_s, std_s = stats(Sn)
        Em_S[j - 1] = mu_s
        Err_S[j - 1] = standard_error(std_s, n_paths)

        values = option_prices_vec(model, Sn, nu_n, K, tau, par)
        for key in OPT_KEYS:
            mu_o, std_o = stats(values[key])
            Em_opt[key][j - 1] = mu_o
            Err_opt[key][j - 1] = standard_error(std_o, n_paths)

        print(
            f"  t_j={tj:5.4f}  tau={tau:5.4f}  E[S]={mu_s:8.6f} +/- {Err_S[j - 1]:8.2e}",
            end="\r",
            flush=True,
        )

    print()
    return times, Em_S, Err_S, Em_opt, Err_opt

# Reporting
def print_header(par: Dict[str, float | int | str | bool]) -> None:
    feller = (2.0 * float(par["lmbda"]) * float(par["nubar"])) / float(par["eta"]) ** 2
    print("@ ------------------------------------------------------------")
    print("@ Heston martingale test")
    print(f"@ So           = {float(par['So']):10.6f}")
    print(f"@ K            = {float(par['K']):10.6f}")
    print(f"@ T            = {float(par['T']):10.6f}")
    print(f"@ lambda       = {float(par['lmbda']):10.8f}")
    print(f"@ nubar        = {float(par['nubar']):10.8f}")
    print(f"@ eta          = {float(par['eta']):10.8f}")
    print(f"@ nu_o         = {float(par['nu_o']):10.8f}")
    print(f"@ rho          = {float(par['rho']):10.8f}")
    print(f"@ Feller       = {feller:10.6f}")
    print(f"@ NV           = 2^{int(par['nv'])} = {1 << int(par['nv'])}")
    print(f"@ months       = {int(par['nm'])}")
    print(f"@ dt           = {float(par['dt']):10.8f}")
    print(f"@ Xc           = {float(par['xc']):10.4f}")
    print(f"@ Xc mode      = {str(par['xcmode'])}")
    print("@ ------------------------------------------------------------")
    print()

def print_initial_prices(O0: Dict[str, float]) -> None:
    print("Time-0 prices (FT)")
    print("  %6s  %12s" % ("Option", "Price"))
    print("  " + "-" * 22)
    for label, key in zip(OPT_LABELS, OPT_KEYS):
        print("  %6s  %12.8f" % (label, O0[key]))
    print()

def print_sanity_checks(O0: Dict[str, float], So: float, K: float) -> None:
    print("@ %-16s: %8.2e" % ("Call-Put parity", O0["call"] - O0["put"] - (So - K)))
    print("@ %-16s: %8.6f" % ("Pcn + Ccn", O0["pCn"] + O0["cCn"]))
    print("@ %-16s: %8.6f" % ("Pan + Can", O0["pAn"] + O0["cAn"]))
    print()

def print_martingale_tables(
    times: np.ndarray,
    Em_S: np.ndarray,
    Err_S: np.ndarray,
    Em_opt: Dict[str, np.ndarray],
    Err_opt: Dict[str, np.ndarray],
    O0: Dict[str, float],
    So: float,
) -> None:
    print("Stock martingale check")
    print("  %8s  %14s  %12s  %12s" % ("t", "E[S_t]", "stderr", "benchmark"))
    for t, mu, se in zip(times, Em_S, Err_S):
        print("  %8.4f  %14.8f  %12.8f  %12.8f" % (t, mu, se, So))

    for label, key in zip(OPT_LABELS, OPT_KEYS):
        print()
        print(f"{label} martingale check")
        print("  %8s  %14s  %12s  %12s" % ("t", f"E[{label}]", "stderr", "benchmark"))
        for t, mu, se in zip(times, Em_opt[key], Err_opt[key]):
            print("  %8.4f  %14.8f  %12.8f  %12.8f" % (t, mu, se, O0[key]))

# Outputs
def save_summary_csv(
    out_csv: str,
    times: np.ndarray,
    Em_S: np.ndarray,
    Err_S: np.ndarray,
    Em_opt: Dict[str, np.ndarray],
    Err_opt: Dict[str, np.ndarray],
    O0: Dict[str, float],
    So: float,
) -> None:
    with open(out_csv, "w", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(["quantity", "time", "mean", "stderr", "benchmark"])
        writer.writerow(["S", 0.0, So, 0.0, So])
        for t, mu, se in zip(times, Em_S, Err_S):
            writer.writerow(["S", float(t), float(mu), float(se), float(So)])
        for label, key in zip(OPT_LABELS, OPT_KEYS):
            writer.writerow([label, 0.0, float(O0[key]), 0.0, float(O0[key])])
            for t, mu, se in zip(times, Em_opt[key], Err_opt[key]):
                writer.writerow([label, float(t), float(mu), float(se), float(O0[key])])

def plot_results(
    out_png: str,
    times: np.ndarray,
    Em_S: np.ndarray,
    Err_S: np.ndarray,
    Em_opt: Dict[str, np.ndarray],
    Err_opt: Dict[str, np.ndarray],
    O0: Dict[str, float],
    par: Dict[str, float | int | str | bool],
    show: bool = False,
) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.ravel()

    fig.suptitle(
        "Martingale Test -- Heston Model\n"
        r"$S_0$=%.2f, $K$=%.2f, $\nu_0$=%.4f, $\lambda$=%.4f, $\eta$=%.4f, $\bar{\nu}$=%.4f, $\rho$=%.4f"
        % (
            float(par["So"]),
            float(par["K"]),
            float(par["nu_o"]),
            float(par["lmbda"]),
            float(par["eta"]),
            float(par["nubar"]),
            float(par["rho"]),
        ),
        fontsize=14,
        y=0.98,
    )

    panel_data = {
        "S": (Em_S, Err_S, float(par["So"])),
        "Put": (Em_opt["put"], Err_opt["put"], O0["put"]),
        "Call": (Em_opt["call"], Err_opt["call"], O0["call"]),
        "Pcn": (Em_opt["pCn"], Err_opt["pCn"], O0["pCn"]),
        "Pan": (Em_opt["pAn"], Err_opt["pAn"], O0["pAn"]),
        "Ccn": (Em_opt["cCn"], Err_opt["cCn"], O0["cCn"]),
        "Can": (Em_opt["cAn"], Err_opt["cAn"], O0["cAn"]),
    }

    for idx, label in enumerate(PLOT_ORDER):
        ax = axes[idx]
        means, errs, benchmark = panel_data[label]
        ax.errorbar(times, means, yerr=errs, fmt="x", color="g", ecolor="g", capsize=3, label="MC stderr")
        ax.axhline(benchmark, color="r", linewidth=1.1, label="time-0 benchmark")
        ax.set_title(label)
        ax.set_xlabel("Years")
        if label == "S":
            ax.set_ylabel(r"$\mathbb{E}[S(t)]$")
        else:
            ax.set_ylabel(r"$\mathbb{E}[V(t,S(t),\nu(t))]$")
        ax.legend(loc="best", fontsize=8)
        ax.grid(alpha=0.20)

    for ax in axes[len(PLOT_ORDER):]:
        ax.axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)

# Main
def main(argv: List[str]) -> int:
    par = parse_inputs(argv)
    out_dir = str(par["out_dir"])
    os.makedirs(out_dir, exist_ok=True)

    print_header(par)
    model = build_model(par)

    t_start = time.time()

    t0 = time.time()
    O0 = option_prices_t0(model, float(par["So"]), float(par["K"]), float(par["T"]), par)
    print_initial_prices(O0)
    print_sanity_checks(O0, float(par["So"]), float(par["K"]))
    print("@ %-16s: %8.4f s" % ("t=0 pricing", time.time() - t0))
    print()

    t1 = time.time()
    tf, S, vol = simulate_outer_paths(model, par)
    print("@ %-16s: %8.4f s" % ("simulation", time.time() - t1))
    print()

    t2 = time.time()
    times, Em_S, Err_S, Em_opt, Err_opt = martingale_test(model, tf, S, vol, par)
    print("@ %-16s: %8.4f s" % ("martingale loop", time.time() - t2))
    print()

    print_martingale_tables(times, Em_S, Err_S, Em_opt, Err_opt, O0, float(par["So"]))

    out_csv = os.path.join(out_dir, "heston_summary.csv")
    out_png = os.path.join(out_dir, "martingale_diagnostics.png")
    save_summary_csv(out_csv, times, Em_S, Err_S, Em_opt, Err_opt, O0, float(par["So"]))
    plot_results(out_png, times, Em_S, Err_S, Em_opt, Err_opt, O0, par, show=bool(par["show"]))

    print()
    print(f"@ CSV summary saved to : {out_csv}")
    print(f"@ Plot saved to        : {out_png}")
    print(f"@ Total elapsed time   : {time.time() - t_start:.2f} s")
    return 0

if __name__ == "__main__":
    raise SystemExit(main(sys.argv))