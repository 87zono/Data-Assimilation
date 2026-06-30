# function/run_bias.py
import numpy as np
from function.EKF_function import EKF_cycle
from function.Serial_EnKF_function import Serial_EnSRF_cycle, create_localization_matrix
from function.LETKF_function import LETKF_cycle
from function.PO_EnKF_function import PO_EnKF_cycle

def run_bias_experiment_all_methods(truth, obs_full, dt, F, m, inflation, sigma, bias_indices):
    """
    観測の偏り（データ欠損領域）の条件下で、EKF, LETKF, Serial_EnSRF, PO法を一斉に走らせる関数
    """
    N = truth.shape[1]
    num_steps = truth.shape[0]
    num_obs = len(bias_indices)

    # 1. 偏り環境用の正しい H 行列 (20 x 40) と R 行列 (20 x 20) を作成
    H_bias = np.zeros((num_obs, N))
    for r, idx in enumerate(bias_indices):
        H_bias[r, idx] = 1.0
    R_bias = np.eye(num_obs)  # 観測がある20点分だけの誤差行列 (すべて1.0)

    # 2. LETKF用の設定
    # LETKFは内部で全変数観測(H=I)を前提としているロジックがあるため、
    # 観測のある20点に対応する対角成分だけを1.0、ない場所は1e4（1万：爆発しない安全な上限値）にします
    R_letkf_diag = np.ones(N) * 10000.0  
    R_letkf_diag[bias_indices] = 1.0
    R_letkf = np.diag(R_letkf_diag)

    # 共通の初期値アンサンブルを作成
    rng = np.random.default_rng(42)
    init_ens = np.zeros((m, N))
    for i in range(m):
        init_ens[i] = truth[0] + rng.normal(0.0, 1.0, N)

    # 各手法の保存用リスト
    ekf_xa = obs_full[0].copy()
    ekf_Pa = 10.0 * np.eye(N)
    ekf_rmse_list = []
    
    letkf_ens = init_ens.copy()
    letkf_rmse_list = []

    serial_ens = init_ens.copy()
    serial_rmse_list = []
    loc_matrix_serial = create_localization_matrix(N, sigma)

    po_ens = init_ens.copy()
    po_rmse_list = []

    # 時間ループ
    for t in range(1, num_steps):
        # 実際に観測が存在する20点だけのデータに切り出す
        y_bias = obs_full[t, bias_indices]
        y_full = obs_full[t]  # LETKF用

        # --- ① EKF ---
        ekf_xa, ekf_Pa, *_ = EKF_cycle(ekf_xa, ekf_Pa, y_bias, dt, F, H_bias, R_bias, inflation)
        ekf_rmse = np.sqrt(np.mean((ekf_xa - truth[t]) ** 2))
        ekf_rmse_list.append(ekf_rmse)

        # --- ② LETKF ---
        # LETKFは全空間の配列（y_full, R_letkf）を渡すことで、内部の weight > 0.001 フィルターにより
        # 自動的に誤差1万の地点を安全にスキップさせます
        letkf_results = LETKF_cycle(letkf_ens, y_full, dt, F, R_letkf, inflation, sigma)
        letkf_ens = letkf_results[0]
        letkf_xa = letkf_results[3]
        letkf_rmse = np.sqrt(np.mean((letkf_xa - truth[t]) ** 2))
        letkf_rmse_list.append(letkf_rmse)

        # --- ③ Serial EnSRF ---
        # 観測のある20点だけの y_bias, H_bias, R_bias を渡す
        serial_results = Serial_EnSRF_cycle(serial_ens, y_bias, dt, F, H_bias, R_bias, inflation, loc_matrix_serial)
        serial_ens = serial_results[0]
        serial_xa = serial_results[1]
        serial_rmse = np.sqrt(np.mean((serial_xa - truth[t]) ** 2))
        serial_rmse_list.append(serial_rmse)

        # --- ④ PO法 ---
        # 観測のある20点だけの y_bias, H_bias, R_bias を渡す
        po_ens, po_xa = PO_EnKF_cycle(po_ens, y_bias, dt, F, H_bias, R_bias, inflation, loc_matrix_serial)
        po_rmse = np.sqrt(np.mean((po_xa - truth[t]) ** 2))
        po_rmse_list.append(po_rmse)

    return {
        "EKF": np.array(ekf_rmse_list),
        "LETKF": np.array(letkf_rmse_list),
        "Serial_EnSRF": np.array(serial_rmse_list),
        "PO_EnKF": np.array(po_rmse_list)
    }