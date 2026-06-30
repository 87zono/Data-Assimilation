import numpy as np
from function.create_data import createdata
from function.run import run_LETKF
from function.plot_heatmap import plot_rmse_heatmap_slide

# --- 1. 基本実験設定 ---
N = 40       # 変数の数
F = 8.0      # 強制項
dt = 0.05    # タイムステップ
m = 20       # アンサンブルメンバー数 (m=10)
H = np.eye(N)
R = np.eye(N)

# heatmap_LETKH.py のパラメータ設定をこれに変更してみてください
inflation_values = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1])
sigma_values = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])

# --- 3. データの作成と準備 (共通データを1回だけ作成) ---
truth, obs, spinup_states = createdata(N, F, dt)

# 同化開始時点の初期予測値（ここは今まで通り）
initial_background = obs[0].copy()

# --- 4. 結果を格納する2次元の箱（グリッド）を用意 ---
rmse_map_m10_p40 = np.zeros((len(inflation_values), len(sigma_values)))

print(f"--- LETKF パラメータ総当たり計算を開始します (m={m:}) ---")

# --- 5. 2重ループで総当たり計算 ---
# heatmap_LETKH.py のループ部分を書き換え
for i, inf in enumerate(inflation_values):
    for j, sig in enumerate(sigma_values):
        print(f"計算中... [Inflation (δ): {inf:.4f} | Sigma (σ): {sig:.4f}]")
        
        try:
            _, _, _, letkf_rmse_a, _ = run_LETKF(
                truth, obs, dt, F, H, R, inf, m, sig,
                spinup_states, initial_background
            )
            rmse_map_m10_p40[i, j] = np.mean(letkf_rmse_a[200:])
            
        except np.linalg.LinAlgError:
            print("\n!!! [エラー発生] 行列が特異になりました !!!")
            print("LETKFの内部計算で数値的な問題が発生しています。")
            print("このパラメータの組み合わせは NaN としてスキップします。\n")
            rmse_map_m10_p40[i, j] = np.nan  # エラー時は NaN を入れて次のループへ強制進行

print("--- すべての計算が完了しました。ヒートマップを描画します ---")

# --- 6. お持ちの関数を呼び出して描画を実行 ---
plot_rmse_heatmap_slide(
    rmse_map=rmse_map_m10_p40,
    inflation_values=inflation_values,
    sigma_values=sigma_values,
    title="Local Ensemble Transform Kalman Filter (LETKF) RMSE: m=20, p=40",
    vmin=0.20,
    vmax=0.35,
    cmap_name="viridis",
    show_values=True
)