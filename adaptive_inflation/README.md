# 動的共分散膨張（Adaptive Covariance Inflation）

Miyoshi (2011) に基づく動的インフレーション推定アルゴリズムの実装・検証。
Lorenz96 モデル上で LETKF（局所アンサンブル変換カルマンフィルタ）と組み合わせ、
観測イノベーション統計からインフレーション係数 Δ を毎ステップ自動推定する。

---

## 背景

アンサンブルカルマンフィルタ（EnKF）では、有限メンバー数による**背景誤差共分散の過小評価**を補正するために共分散膨張（Covariance Inflation）が必要となる。

$$\delta X^b_\mathrm{inf} = \sqrt{\Delta} \cdot \delta X^b \quad (\Delta = 1 + \delta,\ \delta > 0)$$

従来の**固定インフレーション**では最適な δ をヒートマップで手動探索する必要があり、アンサンブル数 m や局所化半径 σ が変わるたびに再探索が必要だった。

本実装では、Miyoshi (2011) のガウスアプローチにより**Δ を毎サイクル自動推定**する。

---

## アルゴリズム概要（Miyoshi 2011）

### ステップ 1：観測からΔを推定

$$\Delta_t^{o} = \frac{\mathrm{tr}(\mathbf{d}^{o-b}(\mathbf{d}^{o-b})^T \circ \mathbf{R}^{-1}) - p_\mathrm{eff}}{\mathrm{tr}(\mathbf{H}\mathbf{B}\mathbf{H}^T \circ \mathbf{R}^{-1})}$$

- $\mathbf{d}^{o-b}$：イノベーション（観測 − 予報平均）
- $p_\mathrm{eff} = \sum_j w_j$：局所化重みの和（実効観測数）
- $\circ$：要素積（アダマール積）

### ステップ 2：ベイズ的加重平均で更新

$$\Delta_t^{a} = \frac{\Delta_t^{b} \cdot v^{o} + \Delta_t^{o} \cdot v^{b}}{v^{o} + v^{b}}$$

$$v^{o} = \frac{2}{p_\mathrm{eff}} \left(\frac{\Delta_t^{b} \cdot \mathrm{tr}(\mathbf{H}\mathbf{B}\mathbf{H}^T \circ \mathbf{R}^{-1}) + p_\mathrm{eff}}{\mathrm{tr}(\mathbf{H}\mathbf{B}\mathbf{H}^T \circ \mathbf{R}^{-1})}\right)^2$$

- $v^b = 0.04^2$：Δ の事前分散（Miyoshi 2011 推奨値）
- $\Delta_t^{a}$ → 次サイクルの事前値 $\Delta_{t+1}^{b}$ として引き継ぐ
- 最終的に Δ は [1.0, 2.0] にクリップ

---

## ディレクトリ構成

```
adaptive_inflation/
├── function/
│   ├── adaptive_inflation.py     # Δ推定コア関数
│   ├── LETKF_adaptive.py         # 固定・動的LETKF サイクル
│   ├── run.py                    # 全タイムステップ実行ラッパー
│   ├── Lorenz96.py               # Lorenz96 モデル（RK4積分）
│   ├── create_data.py            # 真値・観測データ生成
│   └── Serial_EnKF_function.py   # アンサンブル初期化ユーティリティ
├── slide2_settings.py            # ヒートマップ探索（最適固定値の確認）
├── slide3_compare_fixed.py       # 固定 vs 動的 比較（m=20）
├── slide3_m10.py                 # 固定 vs 動的 比較（m=10）
├── slide4_ensemble.py            # アンサンブル数 m=3,8,20 比較
├── presentation.md               # 発表スライド（Marp形式）
└── README.md
```

---

## ファイル詳細

### `function/adaptive_inflation.py`

Δ推定のコアロジック。`estimate_local_delta()` 関数が各格子点の局所領域に対して Miyoshi (2011) の式を適用する。

| 引数 | 説明 |
|---|---|
| `d_local` | イノベーション (p_local,) |
| `Yb_raw_local` | インフレ前の予報摂動（観測空間）(m, p_local) |
| `R_diag_local` | 局所化済み観測誤差分散 (p_local,) |
| `R_diag_raw` | 局所化前の観測誤差分散 (p_local,) |
| `delta_prior` | 前ステップのΔ（事前値） |
| `v_b` | Δの事前分散（チューニングパラメータ） |

### `function/LETKF_adaptive.py`

- `LETKF_fixed_cycle()` — 固定インフレーション δ を使った通常の LETKF
- `LETKF_adaptive_cycle()` — 動的インフレーション。各格子点で Δ を推定し、局所的にインフレを適用する

局所化はガウス重み（カットオフ = 3.6σ）で実装。

### `function/run.py`

全タイムステップのループ処理と発散検出（RMSE > 100）を担当。

- `run_LETKF_fixed()` — 固定 δ で実行、RMSE / スプレッド / Δ履歴を返す
- `run_LETKF_adaptive()` — Miyoshi 2011 動的 inflation で実行、格子点平均Δを返す

---

## 実験設定

| パラメータ | 値 |
|---|---|
| モデル | Lorenz96（N=40, F=8.0） |
| 時間刻み | dt = 0.05（6時間相当） |
| スピンアップ | 1年分（365×4 ステップ） |
| 実験期間 | 1年分（365×4 ステップ） |
| 観測演算子 | H = I₄₀（全変数観測） |
| 観測誤差 | R = I₄₀ |
| 事前分散 | v_b = 0.04²（Miyoshi 2011 推奨値） |

---

## 実験スクリプトの実行方法

実験スクリプトは `adaptive_inflation/` ディレクトリ直下から実行する。

```bash
# ヒートマップ（最適固定値の確認）
python3 slide2_settings.py

# 固定 vs 動的 比較（m=20）
python3 slide3_compare_fixed.py

# 固定 vs 動的 比較（m=10）
python3 slide3_m10.py

# アンサンブル数 m=3,8,20 の比較
python3 slide4_ensemble.py
```

### 依存パッケージ

```
numpy
matplotlib
```

---

## 実験結果

### 各アンサンブル数の最適固定パラメータ（ヒートマップ探索）

| m | 最適σ | 最適δ | 固定 RMSE |
|---|---|---|---|
| 20 | 10.0 | 0.02 | 0.160 |
| 10 | 7.0 | 0.07 | 0.189 |
| 8 | 6.0 | 0.10 | 0.193 |
| 3 | 2.0 | 0.30 | 0.344 |

m が小さいほど大きな δ・小さな σ が必要となる。

### 固定 vs 動的の比較（m=10, σ=7.0）

| 手法 | 平均RMSE | 平均Δ |
|---|---|---|
| インフレなし | 3.85（発散） | 1.000 |
| 固定 δ=0.07（最適） | 0.202 | 1.070 |
| 動的（Miyoshi 2011） | 0.207 | 1.071 |

動的はヒートマップ探索なしで、固定最適値（Δ=1.070）に対してΔ=1.071 を自動推定。RMSE 差は 2.3%。

### アンサンブル数 m による比較

| m | σ | 固定 Δ | 動的の平均Δ | 固定 RMSE | 動的 RMSE |
|---|---|---|---|---|---|
| 3 | 2.0 | 1.30 | 1.309 | 0.395 | **0.372** |
| 8 | 6.0 | 1.10 | 1.103 | 0.231 | **0.222** |
| 20 | 10.0 | 1.02 | 1.016 | 0.177 | 0.179 |

m が小さいほど動的インフレーションの優位性が大きくなる。

---

## 実装上のポイント

**`p_eff` の扱い**：局所化ありの場合、単純な観測点数ではなく局所化重みの和 `p_eff = Σ(R_raw / R_local)` を使う。これを使わないとΔが系統的に過小評価される。

**Δ のクリップ**：数値安定性のため Δ ∈ [1.0, 2.0] に制限している（`adaptive_inflation.py:48`）。

**局所性**：Δ推定は各格子点の局所観測のみを使い、格子点ごとに独立した Δ を保持する。

---

## 参考文献

- Miyoshi T. (2011): The Gaussian Approach to Adaptive Covariance Inflation and Its Implementation with the LETKF. *Mon. Wea. Rev.*, 139, 1519–1535.
- Kotsuki et al. (2017): Adaptive covariance relaxation methods for ensemble data assimilation. *Q. J. R. Meteorol. Soc.*, 143, 2001–2015.
