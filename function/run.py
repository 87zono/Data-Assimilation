import numpy as np
from function.EKF_function import EKF_cycle  # EKF_cycleが別ファイルにある場合はインポート
from function.Serial_EnKF_function import Serial_EnSRF_cycle
from create_ensemble_data import make_initial_ensemble_from_spinup

def run_EKF(truth, obs, dt, F, H, R, inflation):
    """
    EKFの全タイムステップのシミュレーションを実行する関数
    """
    N = truth.shape[1]  # 変数の数（例: 40）
    
    # --- 1. 初期値の設定（ループの前に1度だけ行う） ---
    ekf_xa = obs[0].copy()       # 初期解析値（最初の観測値を仮に使用）
    ekf_Pa = 10.0 * np.eye(N)    # 初期誤差共分散行列
    
    # データを保存するリストの準備
    ekf_xb_save = []
    ekf_xa_save = []
    ekf_rmse_b_list = []
    ekf_rmse_a_list = []
    ekf_spread_a_list = []

    # --- 2. 時間のループ処理 ---
    for t in range(1, truth.shape[0]):
        # EKFの予測・修正サイクル
        ekf_xa, ekf_Pa, ekf_xb, ekf_Pb, ekf_K = EKF_cycle(
            ekf_xa, ekf_Pa, obs[t], dt, F, H, R, inflation
        )

        # 各指標の計算
        ekf_rmse_b = np.sqrt(np.mean((ekf_xb - truth[t]) ** 2))
        ekf_rmse_a = np.sqrt(np.mean((ekf_xa - truth[t]) ** 2))
        ekf_spread_a = np.sqrt(np.trace(ekf_Pa) / N)
        
        # リストに保存
        ekf_xb_save.append(ekf_xb.copy())
        ekf_xa_save.append(ekf_xa.copy())
        ekf_rmse_b_list.append(ekf_rmse_b)
        ekf_rmse_a_list.append(ekf_rmse_a)
        ekf_spread_a_list.append(ekf_spread_a)

    # --- 3. メインファイルに計算結果を返す ---
    # あとで扱いやすいように np.array (NumPy配列) に変換して返すのがおすすめです
    return (
        np.array(ekf_xb_save),
        np.array(ekf_xa_save),
        np.array(ekf_rmse_b_list),
        np.array(ekf_rmse_a_list),
        np.array(ekf_spread_a_list)
    )

# run.py（あるいは関数を定義したファイル）の中身を修正

def run_Serial_EnSRF(truth, obs, dt, F, H, R, inflation, localization_matrix, m, spinup_states, initial_background):
    """
    Serial EnSRF の全タイムステップのシミュレーションを実行する関数
    """
    N = truth.shape[1]
    
    # ★ ここを実際の定義に合わせて正しく呼び出す！
    # seed=42 を指定して呼び出す例
    ensrf_ensemble_a = make_initial_ensemble_from_spinup(
        spinup_states=spinup_states,
        initial_background=initial_background,
        m=m,
        seed=42
    )
    
    ensrf_xb_mean_save = []
    ensrf_xa_mean_save = []
    ensrf_rmse_b_list = []
    ensrf_rmse_a_list = []
    ensrf_spread_a_list = []

    for t in range(1, truth.shape[0]):
        (ensrf_ensemble_a, 
         ensrf_ensemble_b, 
         ensrf_xb_mean, 
         ensrf_xa_mean, 
         ensrf_Pa_approx, 
         ensrf_spread_a) = Serial_EnSRF_cycle(
            ensrf_ensemble_a, obs[t], dt, F, H, R, inflation, localization_matrix
        )

        ensrf_rmse_b = np.sqrt(np.mean((ensrf_xb_mean - truth[t]) ** 2))
        ensrf_rmse_a = np.sqrt(np.mean((ensrf_xa_mean - truth[t]) ** 2))
        
        ensrf_xb_mean_save.append(ensrf_xb_mean.copy())
        ensrf_xa_mean_save.append(ensrf_xa_mean.copy())
        ensrf_rmse_b_list.append(ensrf_rmse_b)
        ensrf_rmse_a_list.append(ensrf_rmse_a)
        ensrf_spread_a_list.append(ensrf_spread_a)

    return (
        np.array(ensrf_xb_mean_save),
        np.array(ensrf_xa_mean_save),
        np.array(ensrf_rmse_b_list),
        np.array(ensrf_rmse_a_list),
        np.array(ensrf_spread_a_list)
    )


# src/run.py の一番下に書き足す

def run_LETKF(truth, obs, dt, F, H, R, inflation, m, localization_radius_sigma, spinup_states, initial_background):
    """
    LETKF の全タイムステップのシミュレーションを実行する関数
    """
    #インポート
    from function.Serial_EnKF_function import make_initial_ensemble_from_spinup
    from function.LETKF_function import LETKF_cycle
    
    # 初期アンサンブルの設定
    letkf_ensemble_a = make_initial_ensemble_from_spinup(
        spinup_states=spinup_states,
        initial_background=initial_background,
        m=m,
        seed=42
    )
    
    letkf_xb_mean_save = []
    letkf_xa_mean_save = []
    letkf_rmse_b_list = []
    letkf_rmse_a_list = []
    letkf_spread_a_list = []

    # 時間ループ（t=0の初期値の次はt=1から開始）
    for t in range(1, truth.shape[0]):
        # LETKFの1サイクルを実行
        (letkf_ensemble_a, 
         letkf_ensemble_b, 
         letkf_xb_mean, 
         letkf_xa_mean, 
         letkf_spread_a) = LETKF_cycle(
            letkf_ensemble_a, obs[t], dt, F, R, inflation, localization_radius_sigma
        )

        # 各指標の計算
        letkf_rmse_b = np.sqrt(np.mean((letkf_xb_mean - truth[t]) ** 2))
        letkf_rmse_a = np.sqrt(np.mean((letkf_xa_mean - truth[t]) ** 2))
        
        # リストに保存
        letkf_xb_mean_save.append(letkf_xb_mean.copy())
        letkf_xa_mean_save.append(letkf_xa_mean.copy())
        letkf_rmse_b_list.append(letkf_rmse_b)
        letkf_rmse_a_list.append(letkf_rmse_a)
        letkf_spread_a_list.append(letkf_spread_a)

    return (
        np.array(letkf_xb_mean_save),
        np.array(letkf_xa_mean_save),
        np.array(letkf_rmse_b_list),
        np.array(letkf_rmse_a_list),
        np.array(letkf_spread_a_list)
    )