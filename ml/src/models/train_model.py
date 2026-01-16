"""
模型訓練腳本
使用多種迴歸模型訓練租金預測模型並評估效能
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score

# 新增src目錄到Python路徑
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / 'src'))

from preprocessing.data_loader import DataLoader
from preprocessing.feature_engineering import FeatureEngineer


def prepare_model_data(df):
    """
    準備模型訓練資料

    Args:
        df: 特徵工程後的DataFrame

    Returns:
        X, y: 特徵和目標變數
    """
    # 使用 FeatureEngineer 進行編碼
    df_encoded = FeatureEngineer.encode_features(df)

    # 選擇數值欄位
    df_numeric = df_encoded.select_dtypes(include=['number'])

    # 移除不需要的欄位
    drop_cols = ['總額元', '每坪租金', '單價元平方公尺', '車位總額元',
                 '租賃年月日', '建築完成年月', '編號', '租賃筆棟數']
    cols_to_drop = [c for c in drop_cols if c in df_numeric.columns]

    X = df_numeric.drop(columns=cols_to_drop)
    X = X.fillna(X.mean())      # 填補 NaN
    X = X.loc[:, X.var() > 0]   # 移除常數特徵

    # 目標變數取 Log（改善模型效能）
    # log1p = log(1 + x)，避免 x=0 時出錯
    y = np.log1p(df_numeric['總額元'])

    return X, y


def train_and_evaluate_models(X, y):
    """
    訓練並評估多個模型

    Args:
        X: 特徵DataFrame
        y: 目標變數（log空間）
    """
    print(f"\n特徵數: {X.shape[1]}, 資料筆數: {X.shape[0]}")

    # 切分資料
    X_train, X_test, y_train_log, y_test_log = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 標準化（Ridge/Lasso 需要）
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 定義模型
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge (L2)": Ridge(alpha=1.0),
        "Lasso (L1)": Lasso(alpha=0.001)
    }

    print("\n" + "="*60)
    print("模型訓練與評估結果")
    print("="*60)

    for name, model in models.items():
        # 訓練
        model.fit(X_train_scaled, y_train_log)

        # 預測（log空間）
        y_pred_log = model.predict(X_test_scaled)

        # 還原回真實金額
        y_pred_real = np.expm1(y_pred_log)
        y_test_real = np.expm1(y_test_log)

        # 計算評估指標
        r2 = r2_score(y_test_real, y_pred_real)
        rmse = np.sqrt(mean_squared_error(y_test_real, y_pred_real))

        print(f"\n[{name}]")
        print(f"  R²: {r2:.4f}")
        print(f"  RMSE: {rmse:,.0f} 元")

        # Lasso特徵篩選資訊
        if name == "Lasso (L1)":
            n_zero = np.sum(model.coef_ == 0)
            print(f"  → Lasso 將 {n_zero} 個特徵的係數變為 0 (特徵篩選)")


def main():
    """主流程"""
    print("\n" + "="*60)
    print("租金預測模型訓練")
    print("="*60)

    # 1. 載入特徵工程後的資料
    print("\n[步驟 1/2] 載入特徵工程後的資料...")
    loader = DataLoader(data_dir='ml/data')
    df = loader.load_featured_data()

    # 2. 準備模型資料
    print("\n[步驟 2/2] 準備模型訓練資料...")
    X, y = prepare_model_data(df)

    # 3. 訓練並評估模型
    train_and_evaluate_models(X, y)

    print("\n" + "="*60)
    print("✓ 模型訓練完成！")
    print("="*60)
    print()


if __name__ == '__main__':
    main()
