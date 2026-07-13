import time
import numpy as np
import matplotlib.pyplot as plt

from function.create_data import createdata
from function.create_ensemble_data import create_localization_matrix

# run関数が保存されているファイル名に合わせる
from function.compare_all_normal import (
    run_PO_EnKF,
    run_Serial_EnSRF,
    run_LETKF,
)

# ============================================================
# 共通設定
# ============================================================

N = 40
F = 8.0
dt = 0.05

# 全地点観測
H = np.eye(N)
R = np.eye(N)

# 初期過渡期間を平均計算から除外
burn_in = 100

# 真値・観測値・スピンアップデータ
truth, obs, spinup_states = createdata(
    N=N,
    F=F,
    dt=dt
)

# 全手法で同じ初期背景値
initial_background = obs[0].copy()

methods = [
    "PO-EnKF",
    "Serial EnSRF",
    "LETKF"
]

def run_ensemble_methods(
    m,
    inflation,
    localization_radius
):
    """
    指定したm、inflation、局所化半径で
    PO-EnKF、Serial EnSRF、LETKFを実行する。
    """

    localization_matrix = create_localization_matrix(
        N=N,
        localization_radius=localization_radius
    )

    results = {}

    # ========================================================
    # PO-EnKF
    # ========================================================

    start_time = time.perf_counter()

    po_result = run_PO_EnKF(
        truth=truth,
        obs=obs,
        dt=dt,
        F=F,
        H=H,
        R=R,
        inflation=inflation,
        localization_matrix=localization_matrix,
        m=m,
        spinup_states=spinup_states,
        initial_background=initial_background
    )

    po_time = time.perf_counter() - start_time

    results["PO-EnKF"] = {
        "mean_rmse": np.mean(
            po_result["rmse_a"][burn_in:]
        ),
        "mean_spread": np.mean(
            po_result["spread_a"][burn_in:]
        ),
        "runtime": po_time
    }

    # ========================================================
    # Serial EnSRF
    # ========================================================

    start_time = time.perf_counter()

    ensrf_result = run_Serial_EnSRF(
        truth=truth,
        obs=obs,
        dt=dt,
        F=F,
        H=H,
        R=R,
        inflation=inflation,
        localization_matrix=localization_matrix,
        m=m,
        spinup_states=spinup_states,
        initial_background=initial_background
    )

    ensrf_time = time.perf_counter() - start_time

    results["Serial EnSRF"] = {
        "mean_rmse": np.mean(
            ensrf_result["rmse_a"][burn_in:]
        ),
        "mean_spread": np.mean(
            ensrf_result["spread_a"][burn_in:]
        ),
        "runtime": ensrf_time
    }

    # ========================================================
    # LETKF
    # ========================================================

    start_time = time.perf_counter()

    letkf_result = run_LETKF(
        truth=truth,
        obs=obs,
        dt=dt,
        F=F,
        R=R,
        inflation=inflation,
        localization_radius_sigma=localization_radius,
        m=m,
        spinup_states=spinup_states,
        initial_background=initial_background
    )

    letkf_time = time.perf_counter() - start_time

    results["LETKF"] = {
        "mean_rmse": np.mean(
            letkf_result["rmse_a"][burn_in:]
        ),
        "mean_spread": np.mean(
            letkf_result["spread_a"][burn_in:]
        ),
        "runtime": letkf_time
    }

    return results

heatmap_m_values = [
    5,
    10,
    20,
    40
]

heatmap_sigma_values = [
    1.0,
    2.0,
    3.0,
    4.0,
    5.0,
    6.0,
    7.0,
    8.0,
    9.0, 10.0
]

fixed_inflation = 0.04

heatmap_results = {
    method: np.zeros(
        (
            len(heatmap_m_values),
            len(heatmap_sigma_values)
        )
    )
    for method in methods
}

for m_index, m_value in enumerate(
    heatmap_m_values
):
    for sigma_index, sigma_value in enumerate(
        heatmap_sigma_values
    ):

        print(
            f"\nRunning m={m_value}, "
            f"sigma={sigma_value}"
        )

        current_results = run_ensemble_methods(
            m=m_value,
            inflation=fixed_inflation,
            localization_radius=sigma_value
        )

        for method in methods:
            heatmap_results[method][
                m_index,
                sigma_index
            ] = current_results[method][
                "mean_rmse"
            ]

for method in methods:

    rmse_matrix = heatmap_results[method]

    plt.figure(figsize=(9, 6))

    image = plt.imshow(
        rmse_matrix,
        aspect="auto",
        origin="lower"
    )

    plt.colorbar(
        image,
        label="Mean analysis RMSE"
    )

    plt.xticks(
        np.arange(len(heatmap_sigma_values)),
        heatmap_sigma_values
    )

    plt.yticks(
        np.arange(len(heatmap_m_values)),
        heatmap_m_values
    )

    plt.xlabel("Localization radius σ")
    plt.ylabel("Ensemble size m")

    plt.title(
        f"{method}: RMSE for m and σ\n"
        f"Inflation={fixed_inflation}"
    )

    # 各セルに数値を表示
    for i in range(len(heatmap_m_values)):
        for j in range(
            len(heatmap_sigma_values)
        ):
            plt.text(
                j,
                i,
                f"{rmse_matrix[i, j]:.3f}",
                ha="center",
                va="center"
            )

    plt.tight_layout()
    plt.show()