import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# 自作関数のimport
# ============================================================
from function.create_data import createdata

from function.create_ensemble_data import (
    make_initial_ensemble_from_spinup,
    create_localization_matrix
)

#from function.Lorenz96 import rk4
from function.EKF_function import EKF_cycle
from function.PO_EnKF_function import PO_EnKF_cycle
from function.Serial_EnKF_function import Serial_EnSRF_cycle
from function.ThreeD_var import threedvar_cycle_gradient_descent
from function.LETKF_function import LETKF_cycle
from function.create_data import calculate_global_spread, calculate_rmse

def run_EKF(
    truth,
    obs,
    dt,
    F,
    H,
    R,
    inflation,
    initial_background
):
    N = truth.shape[1]

    ekf_xa = initial_background.copy()
    ekf_Pa = 10.0 * np.eye(N)

    rmse_a_save = []
    spread_a_save = []

    for t in range(1, len(truth)):

        (
            ekf_xa,
            ekf_Pa,
            ekf_xb,
            ekf_Pb,
            ekf_K
        ) = EKF_cycle(
            ekf_xa,
            ekf_Pa,
            obs[t],
            dt,
            F,
            H,
            R,
            inflation
        )

        rmse_a = calculate_rmse(
            ekf_xa,
            truth[t]
        )

        spread_a = np.sqrt(
            np.trace(ekf_Pa) / N
        )

        rmse_a_save.append(rmse_a)
        spread_a_save.append(spread_a)

    return {
        "rmse_a": np.asarray(rmse_a_save),
        "spread_a": np.asarray(spread_a_save)
    }

def run_3DVar(
    truth,
    obs,
    dt,
    F,
    H,
    B_inv,
    R_inv,
    initial_background
):
    var_xa = initial_background.copy()

    rmse_a_save = []
    spread_a_save = []

    A = np.linalg.inv( B_inv + H.T @ R_inv @ H)
    analysis_spread = np.sqrt( np.trace(A) / N)

    for t in range(1, len(truth)):

        var_xa, var_xb = (
            threedvar_cycle_gradient_descent(
                var_xa,
                obs[t],
                dt,
                F,
                H,
                B_inv,
                R_inv
            )
        )

        rmse_a = calculate_rmse(
            var_xa,
            truth[t]
        )

        rmse_a_save.append(rmse_a)
        spread_a_save.append(analysis_spread  )

    return {
        "rmse_a": np.asarray(rmse_a_save),
        "spread_a": np.asarray(spread_a_save)
    }

def run_PO_EnKF(
    truth,
    obs,
    dt,
    F,
    H,
    R,
    inflation,
    localization_matrix,
    m,
    spinup_states,
    initial_background
):
    po_ensemble_a = make_initial_ensemble_from_spinup(
        spinup_states=spinup_states,
        initial_background=initial_background,
        m=m,
        seed=42
    )

    rmse_a_save = []
    spread_a_save = []

    for t in range(1, len(truth)):

        (
            po_ensemble_a,
            po_xa_mean,
            po_Pa_approx,
            po_spread_a,
            po_spread_by_state
        ) = PO_EnKF_cycle(
            po_ensemble_a,
            obs[t],
            dt,
            F,
            H,
            R,
            inflation,
            localization_matrix
        )

        rmse_a = calculate_rmse(
            po_xa_mean,
            truth[t]
        )

        rmse_a_save.append(rmse_a)
        spread_a_save.append(po_spread_a)

    return {
        "rmse_a": np.asarray(rmse_a_save),
        "spread_a": np.asarray(spread_a_save)
    }

def run_Serial_EnSRF(
    truth,
    obs,
    dt,
    F,
    H,
    R,
    inflation,
    localization_matrix,
    m,
    spinup_states,
    initial_background
):
    ensrf_ensemble_a = make_initial_ensemble_from_spinup(
        spinup_states=spinup_states,
        initial_background=initial_background,
        m=m,
        seed=42
    )

    rmse_a_save = []
    spread_a_save = []

    for t in range(1, len(truth)):

        (
            ensrf_ensemble_a,
            ensrf_ensemble_b,
            ensrf_xb_mean,
            ensrf_xa_mean,
            ensrf_Pa_approx,
            ensrf_spread_a
        ) = Serial_EnSRF_cycle(
            ensrf_ensemble_a,
            obs[t],
            dt,
            F,
            H,
            R,
            inflation,
            localization_matrix
        )

        rmse_a = calculate_rmse(
            ensrf_xa_mean,
            truth[t]
        )

        rmse_a_save.append(rmse_a)
        spread_a_save.append(ensrf_spread_a)

    return {
        "rmse_a": np.asarray(rmse_a_save),
        "spread_a": np.asarray(spread_a_save)
    }

def run_LETKF(
    truth,
    obs,
    dt,
    F,
    R,
    inflation,
    localization_radius_sigma,
    m,
    spinup_states,
    initial_background
):
    letkf_ensemble_a = make_initial_ensemble_from_spinup(
        spinup_states=spinup_states,
        initial_background=initial_background,
        m=m,
        seed=42
    )

    rmse_a_save = []
    spread_a_save = []

    for t in range(1, len(truth)):

        (
            letkf_ensemble_a,
            letkf_ensemble_b,
            letkf_xb_mean,
            letkf_xa_mean,
            letkf_spread_by_state
        ) = LETKF_cycle(
            letkf_ensemble_a,
            obs[t],
            dt,
            F,
            R,
            inflation,
            localization_radius_sigma
        )

        rmse_a = calculate_rmse(
            letkf_xa_mean,
            truth[t]
        )

        global_spread = np.sqrt(
            np.mean(letkf_spread_by_state**2)
        )

        rmse_a_save.append(rmse_a)
        spread_a_save.append(global_spread)

    return {
        "rmse_a": np.asarray(rmse_a_save),
        "spread_a": np.asarray(spread_a_save)
    }


# ============================================================
# 実験設定
# ============================================================

N = 40
F = 8.0
dt = 0.05

m = 20
inflation = 0.05
localization_radius = 5.0

# 全地点観測
H = np.eye(N)

# 観測誤差共分散
R = np.eye(N)
R_inv = np.linalg.inv(R)

# 3D-Varの固定背景誤差共分散
# 最初は単純に単位行列で確認
B = 0.25 * np.eye(N)
B_inv = np.linalg.inv(B)

truth, obs, spinup_states = createdata(
    N=N,
    F=F,
    dt=dt
)

# 全手法で同じ初期背景値
initial_background = obs[0].copy()

# Gaspari–Cohn局所化行列
localization_matrix = (
    create_localization_matrix(
        N=N,
        localization_radius=localization_radius
    )
)

print("Running EKF...")

ekf_result = run_EKF(
    truth=truth,
    obs=obs,
    dt=dt,
    F=F,
    H=H,
    R=R,
    inflation=inflation,
    initial_background=initial_background
)


print("Running 3D-Var...")

var_result = run_3DVar(
    truth=truth,
    obs=obs,
    dt=dt,
    F=F,
    H=H,
    B_inv=B_inv,
    R_inv=R_inv,
    initial_background=initial_background
)


print("Running PO-EnKF...")

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


print("Running Serial EnSRF...")

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


print("Running LETKF...")

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

print("All methods finished.")

time_axis = np.arange(
    1,
    len(truth)
)

plt.figure(figsize=(12, 6))

plt.plot(
    time_axis,
    ekf_result["rmse_a"],
    label="EKF"
)

plt.plot(
    time_axis,
    var_result["rmse_a"],
    label="3D-Var"
)

plt.plot(
    time_axis,
    po_result["rmse_a"],
    label="PO-EnKF"
)

plt.plot(
    time_axis,
    ensrf_result["rmse_a"],
    label="Serial EnSRF"
)

plt.plot(
    time_axis,
    letkf_result["rmse_a"],
    label="LETKF"
)

plt.xlabel("Assimilation step")
plt.ylabel("Analysis RMSE")
plt.title(
    "Analysis RMSE comparison\n"
    f"m={m}, inflation={inflation}, "
    f"localization radius={localization_radius}"
)

plt.grid()
plt.legend()
plt.tight_layout()
plt.show()

burn_in = 100

print("\n===== Summary table =====")

print(
    f"{'Method':<15}"
    f"{'Mean RMSE':>12}"
    f"{'Mean Spread':>15}"
)

print("-" * 42)

print(
    f"{'EKF':<15}"
    f"{np.mean(ekf_result['rmse_a'][burn_in:]):>12.4f}"
    f"{np.mean(ekf_result['spread_a'][burn_in:]):>15.4f}"
)

print(
    f"{'3D-Var':<15}"
    f"{np.mean(var_result['rmse_a'][burn_in:]):>12.4f}"
    f"{np.mean(var_result['spread_a'][burn_in:]):>15.4f}"
)

print(
    f"{'PO-EnKF':<15}"
    f"{np.mean(po_result['rmse_a'][burn_in:]):>12.4f}"
    f"{np.mean(po_result['spread_a'][burn_in:]):>15.4f}"
)

print(
    f"{'Serial EnSRF':<15}"
    f"{np.mean(ensrf_result['rmse_a'][burn_in:]):>12.4f}"
    f"{np.mean(ensrf_result['spread_a'][burn_in:]):>15.4f}"
)

print(
    f"{'LETKF':<15}"
    f"{np.mean(letkf_result['rmse_a'][burn_in:]):>12.4f}"
    f"{np.mean(letkf_result['spread_a'][burn_in:]):>15.4f}"
)

plt.figure(figsize=(12, 6))

plt.plot(
    time_axis,
    ekf_result["spread_a"],
    label="EKF"
)
plt.plot(
    time_axis,
    po_result["spread_a"],
    label="3DVAR"
)
plt.plot(
    time_axis,
    po_result["spread_a"],
    label="PO-EnKF"
)

plt.plot(
    time_axis,
    ensrf_result["spread_a"],
    label="Serial EnSRF"
)

plt.plot(
    time_axis,
    letkf_result["spread_a"],
    label="LETKF"
)

plt.xlabel("Assimilation step")
plt.ylabel("Analysis spread")

plt.title(
    "Analysis spread comparison\n"
    f"m={m}, inflation={inflation}, "
    f"localization radius={localization_radius}"
)

plt.grid()
plt.legend()
plt.tight_layout()
plt.show()

methods_with_spread = [
    "EKF",
    "3D-Var",
    "PO-EnKF",
    "Serial EnSRF",
    "LETKF"
]

mean_rmse = [
    np.mean(ekf_result["rmse_a"][burn_in:]),
    np.mean(var_result["rmse_a"][burn_in:]),
    np.mean(po_result["rmse_a"][burn_in:]),
    np.mean(ensrf_result["rmse_a"][burn_in:]),
    np.mean(letkf_result["rmse_a"][burn_in:])
]

mean_spread = [
    np.mean(ekf_result["spread_a"][burn_in:]),
    np.mean(var_result["spread_a"][burn_in:]),
    np.mean(po_result["spread_a"][burn_in:]),
    np.mean(ensrf_result["spread_a"][burn_in:]),
    np.mean(letkf_result["spread_a"][burn_in:])
]

x = np.arange(len(methods_with_spread))
width = 0.35

plt.figure(figsize=(10, 6))

plt.bar(
    x - width / 2,
    mean_rmse,
    width,
    label="Mean RMSE"
)

plt.bar(
    x + width / 2,
    mean_spread,
    width,
    label="Mean Spread"
)

plt.xticks(
    x,
    methods_with_spread
)

plt.ylabel("Value")

plt.title(
    "Mean RMSE and Mean Spread\n"
    f"m={m}, inflation={inflation}, "
    f"localization radius={localization_radius}"
)

plt.grid(axis="y")
plt.legend()
plt.tight_layout()
plt.show()