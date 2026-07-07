"""
スライド3: 固定LETKF vs 動的LETKF の比較
  - 比較: ①インフレなし ②固定δ=0.02 (最適値) ③動的(Miyoshi 2011)
  - 出力: RMSE / Δ / Spread の時系列グラフ + 平均RMSE表
"""
import numpy as np
import matplotlib.pyplot as plt
from function.create_data import createdata
from function.run import run_LETKF_fixed, run_LETKF_adaptive

# --- 実験設定 ---
N, F, dt = 40, 8.0, 0.05
m    = 20
sigma = 10.0
v_b  = 0.04 ** 2       # Δの事前分散
H    = np.eye(N)
R    = np.eye(N)
FIXED_INF = 0.02       # 最適固定値: RMSE=0.1706 (delta=0.02, sigma=10.0)

print("データ準備中...")
truth, obs, spinup_states = createdata(N, F, dt)
initial_background = obs[0].copy()

# --- 各手法の実行 ---
print("(1/3) インフレなし (δ=0) ...")
_, rmse_0, spread_0, delta_0 = run_LETKF_fixed(
    truth, obs, dt, F, R, inflation=0.0, m=m, sigma=sigma,
    spinup_states=spinup_states, initial_background=initial_background
)

print(f"(2/3) 固定インフレ (δ={FIXED_INF}) ...")
_, rmse_f, spread_f, delta_f = run_LETKF_fixed(
    truth, obs, dt, F, R, inflation=FIXED_INF, m=m, sigma=sigma,
    spinup_states=spinup_states, initial_background=initial_background
)

print("(3/3) 動的インフレ (Miyoshi 2011) ...")
_, rmse_a, spread_a, delta_a = run_LETKF_adaptive(
    truth, obs, dt, F, R, v_b=v_b, m=m, sigma=sigma,
    spinup_states=spinup_states, initial_background=initial_background
)

print("グラフ描画中...")

T = len(rmse_0)
time_days = np.arange(1, T + 1) / 4.0

fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
fig.suptitle(
    f"LETKF Adaptive Inflation: m={m}, σ={sigma}, v_b={v_b:.4f}\n"
    f"No Inflation vs Fixed δ={FIXED_INF} (optimal, RMSE=0.1706) vs Miyoshi (2011)",
    fontsize=13
)

# --- Panel 1: RMSE ---
ax = axes[0]
ax.plot(time_days, rmse_0, label="No inflation", color="red",   alpha=0.7, lw=1.2)
ax.plot(time_days, rmse_f, label=f"Fixed δ={FIXED_INF}", color="blue",  alpha=0.7, lw=1.2)
ax.plot(time_days, rmse_a, label="Adaptive (Miyoshi 2011)", color="green", alpha=0.85, lw=1.5)
ax.set_ylabel("Analysis RMSE")
ax.legend(loc="upper right")
ax.grid(True, linestyle="--", alpha=0.4)

# --- Panel 2: Δ (inflation factor) ---
ax = axes[1]
ax.axhline(1.0,          color="red",  linestyle="--", lw=1.5, label="No inflation (Δ=1.0)")
ax.axhline(1.0+FIXED_INF, color="blue", linestyle="--", lw=1.5, label=f"Fixed (Δ={1+FIXED_INF:.2f})")
ax.plot(time_days, delta_a, color="green", alpha=0.8, lw=1.2, label="Adaptive Δ (grid mean)")
ax.set_ylabel("Inflation factor Δ")
ax.legend(loc="upper right")
ax.grid(True, linestyle="--", alpha=0.4)

# --- Panel 3: Spread ---
ax = axes[2]
ax.plot(time_days, spread_0, label="No inflation", color="red",   alpha=0.7, lw=1.2)
ax.plot(time_days, spread_f, label=f"Fixed δ={FIXED_INF}", color="blue",  alpha=0.7, lw=1.2)
ax.plot(time_days, spread_a, label="Adaptive",     color="green", alpha=0.85, lw=1.5)
ax.set_xlabel("Time [days]")
ax.set_ylabel("Ensemble Spread (mean)")
ax.legend(loc="upper right")
ax.grid(True, linestyle="--", alpha=0.4)

plt.tight_layout()
plt.savefig("slide3_compare_fixed.png", dpi=150, bbox_inches="tight")
plt.show()

# --- 平均RMSE表 ---
print("\n" + "=" * 52)
print(f"{'手法':<22} {'平均RMSE':>10} {'平均Δ':>10}")
print("-" * 52)
print(f"{'No inflation':<22} {np.nanmean(rmse_0):>10.4f} {'1.000':>10}")
print(f"{'Fixed δ=' + str(FIXED_INF):<22} {np.nanmean(rmse_f):>10.4f} {1+FIXED_INF:>10.3f}")
print(f"{'Adaptive (Miyoshi 2011)':<22} {np.nanmean(rmse_a):>10.4f} {np.nanmean(delta_a):>10.4f}")
print("=" * 52)
