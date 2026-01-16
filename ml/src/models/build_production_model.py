"""
生產環境模型建置腳本
使用完整資料集訓練最終模型並儲存
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

# 新增src目錄到Python路徑
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / 'src'))

from preprocessing.data_loader import DataLoader
from preprocessing.feature_engineering import FeatureEngineer


def prepare_production_data(df):
    """
    準備生產環境訓練資料

    Args:
        df: 特徵工程後的DataFrame

    Returns:
        X, y, feature_names: 特徵、目標變數、特徵名稱列表
    """
    # 使用統一的 FeatureEngineer 進行編碼
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

    # 目標變數取 Log
    y = np.log1p(df_numeric['總額元'])

    return X, y, X.columns.tolist()


def build_production_model(X, y):
    """
    訓練生產環境模型

    Args:
        X: 特徵DataFrame
        y: 目標變數

    Returns:
        model, scaler: 訓練好的模型和標準化器
    """
    print("\n正在訓練生產環境模型 (Ridge)...")
    print(f"使用 100% 資料: {len(X)} 筆")

    # 標準化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 訓練 Ridge 模型
    model = Ridge(alpha=1.0)
    model.fit(X_scaled, y)

    print("✓ 模型訓練完成")

    return model, scaler


def save_production_artifacts(model, scaler, features, save_path):
    """
    儲存生產環境所需的所有物件

    Args:
        model: 訓練好的模型
        scaler: 標準化器
        features: 特徵名稱列表
        save_path: 儲存路徑
    """
    artifacts = {
        'model': model,
        'scaler': scaler,
        'features': features,
        'metadata': {
            'model_type': 'Ridge',
            'alpha': model.alpha,
            'n_features': len(features),
            'preprocessor': 'FeatureEngineer'
        }
    }

    # 確保目錄存在
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # 儲存
    joblib.dump(artifacts, save_path)
    print(f"\n✓ 模型已儲存至: {save_path}")

    # 顯示模型資訊
    print("\n=== 模型資訊 ===")
    print(f"模型類型: {artifacts['metadata']['model_type']}")
    print(f"正則化參數: α = {artifacts['metadata']['alpha']}")
    print(f"特徵數量: {artifacts['metadata']['n_features']}")
    print(f"前10個特徵: {features[:10]}")


def main():
    """主流程"""
    print("\n" + "="*60)
    print("生產環境模型建置")
    print("="*60)

    # 1. 載入資料
    print("\n[步驟 1/3] 載入特徵工程後的資料...")
    loader = DataLoader(data_dir='ml/data')
    df = loader.load_featured_data()

    # 2. 準備資料
    print("\n[步驟 2/3] 準備訓練資料...")
    X, y, features = prepare_production_data(df)

    # 3. 訓練模型
    print("\n[步驟 3/3] 訓練生產環境模型...")
    model, scaler = build_production_model(X, y)

    # 4. 儲存模型
    save_path = 'ml/src/models/rent_prediction_model.pkl'
    save_production_artifacts(model, scaler, features, save_path)

    print("\n" + "="*60)
    print("✓ 生產環境模型建置完成！")
    print("="*60)
    print(f"\n可在 Django 中載入此模型進行預測")
    print()


if __name__ == '__main__':
    main()
