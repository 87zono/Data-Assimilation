"""
動的インフレーションの初期Δ比較
  - delta_init = 固定最適値（ズルあり）vs Δ=1.0（何も知らない状態）
  - m=3,8,20 で比較
"""
import numpy as np
import matplotlib.pyplot as plt
from function.create_data import createdata
from function.run import run_LETKF_fixed, run_LETKF_adaptive

N, F, dt = 40, 8.0, 0.05
R        = np.eye(N)
v_b      = 0.04 ** 2

OPTIMAL = {
    3:  {"sigma": 2.0,  "fixed_inf": 0.30, "delta_init": 1.30},
    8:  {"sigma": 6.0,  "fixed_inf": 0.10, "delta_init": 1.10},
    20: {"sigma": 10.0, "fixed_inf": 0.02, "delta_init": 1.02},
}
MEMBERS = [3, 8, 20]

print("データ準備中...")
truth, obs, spinup_states = createdata(N, F, dt)
initial_background = obs[0].copy()

results = {}
for m in MEMBERS:
    sigma     = OPTIMAL[m]["sigma"]
    fixed_inf = OPTIMAL[m]["fixed_inf"]
    delta_opt = OPTIMAL[m]["delta_init"]

    print(f"\n--- m={m} ---")
    print(f"  固定インフレ (ベースライン)...")
    _, rmse_f, _, delta_f = run_LETKF_fixed(
        truth, obs, dt, F, R, inflation=fixed_inf, m=m, sigma=sigma,
        spinup_states=spinup_states, initial_background=initial_background
    )

    print(f"  動的 (delta_init=固定最適値 {delta_opt})...")
    _, rmse_opt, _, delta_opt_hist = run_LETKF_adaptive(
        truth, obs, dt, F, R, v_b=v_b, m=m, sigma=sigma,
        spinup_states=spinup_states, initial_background=initial_background,
        delta_init=delta_opt
    )

    print(f"  動的 (delta_init=1.0 何も知らない)...")
    _, rmse_1, _, delta_1_hist = run_LETKF_adaptive(
        truth, obs, dt, F, R, v_b=v_b, m=m, sigma=sigma,
        spinup_states=spinup_states, initial_background=initial_background,
        delta_init=1.0
    )

    results[m] = (rmse_f, delta_f, rmse_opt, delta_opt_hist, rmse_1, delta_1_hist)

T = len(results[MEMBERS[0]][0])
time_days = np.arange(1, T + 1) / 4.0

fig, axes = plt.subplots(2, len(MEMBERS), figsize=(7 * len(MEMBERS), 6), sharex=True)
fig.suptitle("動的インフレーション: delta_init=固定最適値 vs delta_init=1.0", fontsize=13)

for col, m in enumerate(MEMBERS):
    rmse_f, delta_f, rmse_opt, delta_opt_hist, rmse_1, delta_1_hist = results[m]
    sigma     = OPTIMAL[m]["sigma"]
    fixed_inf = OPTIMAL[m]["fixed_inf"]
    delta_opt = OPTIMAL[m]["delta_init"]

    # RMSE
    ax = axes[0, col]
    ax.plot(time_days, rmse_f,   color="blue",   lw=1.5, linestyle="--", label=f"Fixed δ={fixed_inf}")
    ax.plot(time_days, rmse_opt, color="green",  lw=1.2, alpha=0.85, label=f"Adaptive (init={delta_opt})")
    ax.plot(time_days, rmse_1,   color="red",    lw=1.2, alpha=0.85, label="Adaptive (init=1.0)")
    ax.set_title(f"m={m}  (σ={sigma})", fontsize=11)
    ax.set_ylabel("Analysis RMSE" if col == 0 else "")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=8, loc="upper right")
    ax.text(0.02, 0.95,
            f"Fixed:      {np.nanmean(rmse_f):.3f}\n"
            f"Init=opt:  {np.nanmean(rmse_opt):.3f}\n"
            f"Init=1.0:  {np.nanmean(rmse_1):.3f}",
            transform=ax.transAxes, ha="left", va="top", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))

    # Δ
    ax = axes[1, col]
    ax.axhline(1.0 + fixed_inf, color="blue",  lw=1.5, linestyle="--", label=f"Fixed Δ={1+fixed_inf:.2f}")
    ax.plot(time_days, delta_opt_hist, color="green", lw=1.2, alpha=0.85, label=f"Adaptive (init={delta_opt})")
    ax.plot(time_days, delta_1_hist,   color="red",   lw=1.2, alpha=0.85, label="Adaptive (init=1.0)")
    ax.set_ylabel("Inflation factor Δ" if col == 0 else "")
    ax.set_xlabel("Time [days]")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(fontsize=8, loc="upper right")

plt.tight_layout()
plt.savefig("slide_delta_init.png", dpi=150, bbox_inches="tight")
plt.show()

print("\n" + "=" * 62)
print(f"{'m':>4} {'手法':<28} {'平均RMSE':>10} {'平均Δ':>10}")
print("-" * 62)
for m in MEMBERS:
    rmse_f, delta_f, rmse_opt, delta_opt_hist, rmse_1, delta_1_hist = results[m]
    fixed_inf = OPTIMAL[m]["fixed_inf"]
    delta_opt = OPTIMAL[m]["delta_init"]
    print(f"{m:>4} {'Fixed δ='+str(fixed_inf):<28} {np.nanmean(rmse_f):>10.4f} {1+fixed_inf:>10.3f}")
    print(f"{m:>4} {'Adaptive (init='+str(delta_opt)+')':<28} {np.nanmean(rmse_opt):>10.4f} {np.nanmean(delta_opt_hist):>10.4f}")
    print(f"{m:>4} {'Adaptive (init=1.0)':<28} {np.nanmean(rmse_1):>10.4f} {np.nanmean(delta_1_hist):>10.4f}")
    print("-" * 62)
print("=" * 62)
