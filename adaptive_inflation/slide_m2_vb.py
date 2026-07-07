"""
m=2: v_b を変えたときの動的インフレーションの挙動比較
  固定最適（δ=0.80, σ=1.0）をベースラインとして比較
"""
import numpy as np
import matplotlib.pyplot as plt
from function.create_data import createdata
from function.run import run_LETKF_fixed, run_LETKF_adaptive

N, F, dt = 40, 8.0, 0.05
m        = 2
sigma    = 1.0
R        = np.eye(N)
FIXED_INF  = 0.80
DELTA_INIT = 1.80

VB_LIST = [0.04**2, 0.02**2, 0.01**2, 0.005**2]
LABELS  = ["v_b=0.04² (現在)", "v_b=0.02²", "v_b=0.01²", "v_b=0.005²"]
COLORS  = ["green", "orange", "purple", "brown"]

print("データ準備中...")
truth, obs, spinup_states = createdata(N, F, dt)
initial_background = obs[0].copy()

print("固定インフレ (ベースライン)...")
_, rmse_f, spread_f, delta_f = run_LETKF_fixed(
    truth, obs, dt, F, R, inflation=FIXED_INF, m=m, sigma=sigma,
    spinup_states=spinup_states, initial_background=initial_background
)

results = []
for v_b, label in zip(VB_LIST, LABELS):
    print(f"動的インフレ ({label}) ...")
    _, rmse_a, spread_a, delta_a = run_LETKF_adaptive(
        truth, obs, dt, F, R, v_b=v_b, m=m, sigma=sigma,
        spinup_states=spinup_states, initial_background=initial_background,
        delta_init=DELTA_INIT
    )
    results.append((rmse_a, spread_a, delta_a))

T = len(rmse_f)
time_days = np.arange(1, T + 1) / 4.0

fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
fig.suptitle(f"m={m}, σ={sigma}  |  Fixed δ={FIXED_INF} vs Adaptive (v_b 比較)", fontsize=13)

# RMSE
ax = axes[0]
ax.plot(time_days, rmse_f, color="blue", lw=1.5, linestyle="--", label=f"Fixed δ={FIXED_INF} (optimal)")
for (rmse_a, _, _), label, color in zip(results, LABELS, COLORS):
    ax.plot(time_days, rmse_a, color=color, lw=1.2, alpha=0.85, label=label)
ax.set_ylabel("Analysis RMSE")
ax.legend(loc="upper right", fontsize=9)
ax.grid(True, linestyle="--", alpha=0.4)

# Δ
ax = axes[1]
ax.axhline(1.0 + FIXED_INF, color="blue", linestyle="--", lw=1.5, label=f"Fixed Δ={1+FIXED_INF:.2f}")
for (_, _, delta_a), label, color in zip(results, LABELS, COLORS):
    ax.plot(time_days, delta_a, color=color, lw=1.2, alpha=0.85, label=label)
ax.set_ylabel("Inflation factor Δ")
ax.legend(loc="upper right", fontsize=9)
ax.grid(True, linestyle="--", alpha=0.4)

# Spread
ax = axes[2]
ax.plot(time_days, spread_f, color="blue", lw=1.5, linestyle="--", label=f"Fixed δ={FIXED_INF}")
for (_, spread_a, _), label, color in zip(results, LABELS, COLORS):
    ax.plot(time_days, spread_a, color=color, lw=1.2, alpha=0.85, label=label)
ax.set_xlabel("Time [days]")
ax.set_ylabel("Ensemble Spread (mean)")
ax.legend(loc="upper right", fontsize=9)
ax.grid(True, linestyle="--", alpha=0.4)

plt.tight_layout()
plt.savefig("slide_m2_vb.png", dpi=150, bbox_inches="tight")
plt.show()

print("\n" + "=" * 58)
print(f"{'手法':<28} {'平均RMSE':>10} {'平均Δ':>10}")
print("-" * 58)
print(f"{'Fixed δ='+str(FIXED_INF):<28} {np.nanmean(rmse_f):>10.4f} {1+FIXED_INF:>10.3f}")
for (rmse_a, _, delta_a), label in zip(results, LABELS):
    print(f"{label:<28} {np.nanmean(rmse_a):>10.4f} {np.nanmean(delta_a):>10.4f}")
print("=" * 58)
