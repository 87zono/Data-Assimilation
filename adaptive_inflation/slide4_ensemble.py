"""
スライド4: アンサンブル数 m=3, 8, 20 による比較
  - 固定インフレ vs 動的インフレ を同じ軸に重ねてプロット
  - 上段: RMSE時系列（固定・動的を同軸）
  - 下段: Δ（インフレーション係数）時系列
"""
import numpy as np
import matplotlib.pyplot as plt
from function.create_data import createdata
from function.run import run_LETKF_fixed, run_LETKF_adaptive

# --- 実験設定 ---
N, F, dt = 40, 8.0, 0.05
v_b      = 0.04 ** 2
R        = np.eye(N)

# m別の最適パラメータ（ヒートマップで確認済み）
# delta_init: 動的inflationの初期Δ（固定最適値付近から始める）
OPTIMAL = {
    3:  {"sigma": 2.0,  "fixed_inf": 0.30, "delta_init": 1.30},  # m=3: RMSE=0.344 (ヒートマップ確認済み)
    8:  {"sigma": 6.0,  "fixed_inf": 0.10, "delta_init": 1.10},  # m=8: RMSE=0.193 (ヒートマップ確認済み)
    20: {"sigma": 10.0, "fixed_inf": 0.02, "delta_init": 1.02},  # m=20: RMSE=0.160 (ヒートマップ確認済み)
}
MEMBERS = [3, 8, 20]

print("データ準備中...")
truth, obs, spinup_states = createdata(N, F, dt)
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

    print(f"  動的インフレ (Miyoshi 2011)...")
    _, rmse_a, spread_a, delta_a = run_LETKF_adaptive(
        truth, obs, dt, F, R, v_b=v_b, m=m, sigma=sigma,
        spinup_states=spinup_states, initial_background=initial_background,
        delta_init=OPTIMAL[m]["delta_init"]
    )
    results_adaptive[m] = (rmse_a, spread_a, delta_a)

print("\nグラフ描画中...")

T = len(results_fixed[MEMBERS[0]][0])
time_days = np.arange(1, T + 1) / 4.0

COLOR_FIXED    = "blue"
COLOR_ADAPTIVE = "green"

fig, axes = plt.subplots(2, 3, figsize=(20, 6), sharex=True)
fig.suptitle(
    f"LETKF: Fixed (m別最適値) vs Adaptive (Miyoshi 2011) — v_b={v_b:.4f}",
    fontsize=13
)

for col, m in enumerate(MEMBERS):
    rmse_f, _, delta_f = results_fixed[m]
    rmse_a, _, delta_a = results_adaptive[m]
    sigma_m   = OPTIMAL[m]["sigma"]
    fixed_inf = OPTIMAL[m]["fixed_inf"]

    # --- 上段: RMSE（固定・動的を同じ軸に重ねる）---
    ax = axes[0, col]
    ax.plot(time_days, rmse_f, color=COLOR_FIXED,    lw=1.2, alpha=0.85,
            label=f"Fixed δ={fixed_inf}")
    ax.plot(time_days, rmse_a, color=COLOR_ADAPTIVE, lw=1.5, alpha=0.85,
            label="Adaptive")
    ax.set_title(f"m={m}  (σ={sigma_m}, δ_opt={fixed_inf})", fontsize=11)
    ax.set_ylabel("Analysis RMSE" if col == 0 else "")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=8, loc="upper right")

    mean_f = np.nanmean(rmse_f)
    mean_a = np.nanmean(rmse_a)
    label_f = f"{mean_f:.3f}" if np.isfinite(mean_f) else "DIV"
    ax.text(0.02, 0.95,
            f"Fixed mean: {label_f}\nAdapt mean: {mean_a:.3f}",
            transform=ax.transAxes, ha="left", va="top", fontsize=8,
            color="black",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))

    # --- 下段: Δの時系列（固定・動的を同じ軸に重ねる）---
    ax = axes[1, col]
    ax.axhline(1.0 + fixed_inf, color=COLOR_FIXED,    lw=1.5,
               linestyle="--", label=f"Fixed Δ={1+fixed_inf:.2f}")
    ax.plot(time_days, delta_a, color=COLOR_ADAPTIVE, lw=1.2, alpha=0.85,
            label="Adaptive Δ (grid mean)")
    ax.set_ylabel("Inflation factor Δ" if col == 0 else "")
    ax.set_xlabel("Time [days]")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(fontsize=8, loc="upper right")

    mean_da = np.nanmean(delta_a)
    ax.text(0.02, 0.95, f"Adapt Δ mean: {mean_da:.3f}",
            transform=ax.transAxes, ha="left", va="top", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))

plt.tight_layout()
plt.savefig("slide4_ensemble.png", dpi=150, bbox_inches="tight")
plt.show()

# --- 比較表 ---
print("\n" + "=" * 62)
print(f"{'m':>4} {'手法':<22} {'平均RMSE':>10} {'平均Δ':>10}")
print("-" * 62)
for m in MEMBERS:
    rmse_f, _, delta_f = results_fixed[m]
    rmse_a, _, delta_a = results_adaptive[m]
    fixed_inf = OPTIMAL[m]["fixed_inf"]
    sigma_m   = OPTIMAL[m]["sigma"]
    mean_f = np.nanmean(rmse_f)
    mean_a = np.nanmean(rmse_a)
    mean_df = np.nanmean(delta_f)
    mean_da = np.nanmean(delta_a)
    label_f = f"{mean_f:.4f}" if np.isfinite(mean_f) else "DIVERGED"
    print(f"{m:>4} {('Fixed δ='+str(fixed_inf)+', σ='+str(sigma_m)):<22} {label_f:>10} {mean_df:>10.3f}")
    print(f"{m:>4} {'Adaptive':<22} {mean_a:>10.4f} {mean_da:>10.4f}")
    print("-" * 62)
print("=" * 62)
