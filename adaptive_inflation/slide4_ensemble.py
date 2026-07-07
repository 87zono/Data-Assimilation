"""
スライド4: アンサンブル数 m=2,3,8,20 による比較
  - 固定インフレ vs 動的インフレ（コールドスタート init=1.0）
  - 上段: RMSE時系列
  - 下段: Δ時系列
  - 数値は後半1年（収束後）の平均を使用
"""
import numpy as np
import matplotlib.pyplot as plt
from function.create_data import createdata
from function.run import run_LETKF_fixed, run_LETKF_adaptive

# --- 実験設定 ---
N, F, dt = 40, 8.0, 0.05
v_b      = 0.04 ** 2
R        = np.eye(N)

OPTIMAL = {
    2:  {"sigma": 1.0,  "fixed_inf": 0.80},
    3:  {"sigma": 2.0,  "fixed_inf": 0.30},
    8:  {"sigma": 6.0,  "fixed_inf": 0.10},
    20: {"sigma": 10.0, "fixed_inf": 0.02},
}
MEMBERS = [2, 3, 8, 20]

EXPERIMENT_YEARS = 5

print(f"データ準備中 ({EXPERIMENT_YEARS}年分)...")
truth, obs, spinup_states = createdata(N, F, dt, experiment_years=EXPERIMENT_YEARS)
initial_background = obs[0].copy()

results_fixed    = {}
results_adaptive = {}

for m in MEMBERS:
    sigma     = OPTIMAL[m]["sigma"]
    fixed_inf = OPTIMAL[m]["fixed_inf"]
    print(f"\n--- m={m} (σ={sigma}, δ_fixed={fixed_inf}) ---")
    print(f"  固定インフレ...")
    _, rmse_f, spread_f, delta_f = run_LETKF_fixed(
        truth, obs, dt, F, R, inflation=fixed_inf, m=m, sigma=sigma,
        spinup_states=spinup_states, initial_background=initial_background
    )
    results_fixed[m] = (rmse_f, spread_f, delta_f)

    print(f"  動的インフレ コールドスタート (init=1.0)...")
    _, rmse_c, spread_c, delta_c = run_LETKF_adaptive(
        truth, obs, dt, F, R, v_b=v_b, m=m, sigma=sigma,
        spinup_states=spinup_states, initial_background=initial_background,
        delta_init=1.0
    )
    results_adaptive[m] = (rmse_c, spread_c, delta_c)

print("\nグラフ描画中...")

T = len(results_fixed[MEMBERS[0]][0])
time_days = np.arange(1, T + 1) / 4.0
steps_last_year = 365 * 4

COLOR_FIXED    = "blue"
COLOR_ADAPTIVE = "red"

fig, axes = plt.subplots(2, len(MEMBERS), figsize=(7 * len(MEMBERS), 6), sharex=True)
fig.suptitle(
    f"LETKF: Fixed (optimal) vs Adaptive cold start (init=1.0) — v_b={v_b:.4f}, {EXPERIMENT_YEARS}yr",
    fontsize=13
)

for col, m in enumerate(MEMBERS):
    rmse_f, _, delta_f = results_fixed[m]
    rmse_c, _, delta_c = results_adaptive[m]
    sigma_m   = OPTIMAL[m]["sigma"]
    fixed_inf = OPTIMAL[m]["fixed_inf"]

    tail_f = np.nanmean(rmse_f[-steps_last_year:])
    tail_c = np.nanmean(rmse_c[-steps_last_year:])

    # --- 上段: RMSE ---
    ax = axes[0, col]
    ax.plot(time_days, rmse_f, color=COLOR_FIXED,    lw=1.2, alpha=0.85,
            linestyle="--", label=f"Fixed δ={fixed_inf}")
    ax.plot(time_days, rmse_c, color=COLOR_ADAPTIVE, lw=1.2, alpha=0.85,
            label="Adaptive (cold start)")
    ax.set_title(f"m={m}  (σ={sigma_m}, δ_opt={fixed_inf})", fontsize=11)
    ax.set_ylabel("Analysis RMSE" if col == 0 else "")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=8, loc="upper right")

    lf = f"{tail_f:.3f}" if np.isfinite(tail_f) else "DIV"
    lc = f"{tail_c:.3f}" if np.isfinite(tail_c) else "DIV"
    ax.text(0.02, 0.95,
            f"Fixed (last yr):    {lf}\nAdaptive (last yr): {lc}",
            transform=ax.transAxes, ha="left", va="top", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))

    # --- 下段: Δの時系列 ---
    ax = axes[1, col]
    ax.axhline(1.0 + fixed_inf, color=COLOR_FIXED, lw=1.5,
               linestyle="--", label=f"Fixed Δ={1+fixed_inf:.2f}")
    ax.plot(time_days, delta_c, color=COLOR_ADAPTIVE, lw=1.2, alpha=0.85,
            label="Adaptive Δ (cold)")
    ax.set_ylabel("Inflation factor Δ" if col == 0 else "")
    ax.set_xlabel("Time [days]")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(fontsize=8, loc="upper right")

    tail_dc = np.nanmean(delta_c[-steps_last_year:])
    ax.text(0.02, 0.95, f"Adaptive Δ (last yr): {tail_dc:.3f}",
            transform=ax.transAxes, ha="left", va="top", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))

plt.tight_layout()
plt.savefig("slide4_ensemble.png", dpi=150, bbox_inches="tight")
plt.show()

# --- 比較表（後半1年） ---
print("\n" + "=" * 66)
print(f"{'m':>4} {'手法':<26} {'後半1年RMSE':>12} {'後半1年 平均Δ':>14}")
print("-" * 66)
for m in MEMBERS:
    rmse_f, _, delta_f = results_fixed[m]
    rmse_c, _, delta_c = results_adaptive[m]
    fixed_inf = OPTIMAL[m]["fixed_inf"]
    sigma_m   = OPTIMAL[m]["sigma"]
    tail_f  = np.nanmean(rmse_f[-steps_last_year:])
    tail_c  = np.nanmean(rmse_c[-steps_last_year:])
    tail_df = np.nanmean(delta_f[-steps_last_year:])
    tail_dc = np.nanmean(delta_c[-steps_last_year:])
    lf = f"{tail_f:.4f}" if np.isfinite(tail_f) else "DIV"
    lc = f"{tail_c:.4f}" if np.isfinite(tail_c) else "DIV"
    print(f"{m:>4} {('Fixed δ='+str(fixed_inf)+', σ='+str(sigma_m)):<26} {lf:>12} {tail_df:>14.3f}")
    print(f"{m:>4} {'Adaptive (cold start)':<26} {lc:>12} {tail_dc:>14.3f}")
    print("-" * 66)
print("=" * 66)
