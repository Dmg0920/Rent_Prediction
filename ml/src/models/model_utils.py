"""
模型工具模組
提供統一的模型載入、驗證和預測介面
"""
import joblib
import numpy as np
import pandas as pd
from pathlib import Path


class ModelManager:
    """
    模型管理器
    統一處理模型載入、驗證和預測
    """

    def __init__(self, model_path='ml/src/models/rent_prediction_model.pkl'):
        """
        初始化模型管理器

        Args:
            model_path: 模型檔案路徑
        """
        self.model_path = Path(model_path)
        self.artifacts = None
        self.model = None
        self.scaler = None
        self.features = None
        self.metadata = None

    def load_model(self):
        """
        載入模型和相關物件

        Returns:
            bool: 是否成功載入
        """
        try:
            if not self.model_path.exists():
                print(f"錯誤: 找不到模型檔案 {self.model_path}")
                return False

            self.artifacts = joblib.load(self.model_path)

            # 提取元件
            self.model = self.artifacts['model']
            self.scaler = self.artifacts['scaler']
            self.features = self.artifacts['features']
            self.metadata = self.artifacts.get('metadata', {})

            print(f"✓ 成功載入模型: {self.model_path}")
            print(f"  模型類型: {self.metadata.get('model_type', 'Unknown')}")
            print(f"  特徵數量: {len(self.features)}")

            return True

        except Exception as e:
            print(f"載入模型時發生錯誤: {e}")
            return False

    def validate_features(self, input_features):
        """
        驗證輸入特徵是否符合模型要求

        Args:
            input_features: 輸入特徵列表或DataFrame欄位

        Returns:
            bool, str: 是否通過驗證, 錯誤訊息
        """
        if self.features is None:
            return False, "模型尚未載入"

        input_set = set(input_features)
        model_set = set(self.features)

        # 檢查缺少的特徵
        missing = model_set - input_set
        if missing:
            return False, f"缺少特徵: {list(missing)[:5]}"

        # 檢查多餘的特徵
        extra = input_set - model_set
        if extra:
            return False, f"多餘特徵: {list(extra)[:5]}"

        return True, "驗證通過"

    def predict(self, X):
        """
        進行預測

        Args:
            X: 特徵DataFrame或numpy array

        Returns:
            predictions: 預測結果（原始金額）
        """
        if self.model is None or self.scaler is None:
            raise ValueError("模型尚未載入，請先呼叫 load_model()")

        # 確保特徵順序正確
        if isinstance(X, pd.DataFrame):
            # 驗證特徵
            valid, msg = self.validate_features(X.columns)
            if not valid:
                raise ValueError(f"特徵驗證失敗: {msg}")

            X = X[self.features].values

        # 標準化
        X_scaled = self.scaler.transform(X)

        # 預測（log空間）
        y_pred_log = self.model.predict(X_scaled)

        # 還原到原始金額
        y_pred = np.expm1(y_pred_log)

        return y_pred

    def predict_single(self, feature_dict):
        """
        對單筆資料進行預測

        Args:
            feature_dict: 特徵字典

        Returns:
            float: 預測租金
        """
        # 轉換為DataFrame
        df = pd.DataFrame([feature_dict])

        # 確保所有特徵都存在
        for feat in self.features:
            if feat not in df.columns:
                df[feat] = 0  # 缺少的特徵填0

        # 預測
        prediction = self.predict(df)

        return float(prediction[0])

    def get_model_info(self):
        """
        取得模型資訊

        Returns:
            dict: 模型資訊
        """
        if self.artifacts is None:
            return {"status": "未載入"}

        return {
            "model_type": self.metadata.get('model_type', 'Unknown'),
            "n_features": len(self.features),
            "features": self.features[:10],  # 前10個特徵
            "alpha": self.metadata.get('alpha', None),
            "preprocessor": self.metadata.get('preprocessor', 'Unknown')
        }


# 便利函式
def load_production_model(model_path='ml/src/models/rent_prediction_model.pkl'):
    """
    快速載入生產環境模型

    Args:
        model_path: 模型路徑

    Returns:
        ModelManager: 已載入的模型管理器
    """
    manager = ModelManager(model_path)
    if manager.load_model():
        return manager
    else:
        raise FileNotFoundError(f"無法載入模型: {model_path}")


def predict_rent(features, model_path='ml/src/models/rent_prediction_model.pkl'):
    """
    快速預測租金

    Args:
        features: 特徵DataFrame或字典
        model_path: 模型路徑

    Returns:
        predictions: 預測結果
    """
    manager = load_production_model(model_path)

    if isinstance(features, dict):
        return manager.predict_single(features)
    else:
        return manager.predict(features)
