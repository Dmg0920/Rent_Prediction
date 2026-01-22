"""
集成學習模型模組
提供進階機器學習模型，包括：
- Random Forest
- Gradient Boosting
- XGBoost
- LightGBM
- 模型堆疊 (Stacking)
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# 嘗試導入XGBoost和LightGBM（如果已安裝）
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False


class EnsembleModelTrainer:
    """集成學習模型訓練器"""

    def __init__(self, random_state=42):
        """
        初始化訓練器

        Args:
            random_state: 隨機種子
        """
        self.random_state = random_state
        self.models = {}
        self.results = {}

    def get_models(self):
        """
        獲取所有可用的模型

        Returns:
            dict: 模型名稱到模型對象的映射
        """
        models = {
            'Random Forest': RandomForestRegressor(
                n_estimators=100,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=self.random_state,
                n_jobs=-1
            ),
            'Gradient Boosting': GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=self.random_state
            )
        }

        if XGBOOST_AVAILABLE:
            models['XGBoost'] = xgb.XGBRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                min_child_weight=3,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.random_state,
                n_jobs=-1
            )

        if LIGHTGBM_AVAILABLE:
            models['LightGBM'] = lgb.LGBMRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                num_leaves=31,
                min_child_samples=20,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.random_state,
                n_jobs=-1,
                verbose=-1
            )

        return models

    def train_single_model(self, name, model, X_train, y_train, X_test, y_test,
                          use_log=True):
        """
        訓練單一模型並評估

        Args:
            name: 模型名稱
            model: 模型對象
            X_train: 訓練特徵
            y_train: 訓練目標
            X_test: 測試特徵
            y_test: 測試目標
            use_log: 是否使用對數轉換目標變數

        Returns:
            dict: 包含模型和評估結果
        """
        # 訓練模型
        if use_log:
            y_train_transformed = np.log1p(y_train)
            model.fit(X_train, y_train_transformed)
            y_pred_transformed = model.predict(X_test)
            y_pred = np.expm1(y_pred_transformed)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

        # 計算評估指標
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

        return {
            'model': model,
            'predictions': y_pred,
            'metrics': {
                'R²': r2,
                'RMSE': rmse,
                'MAE': mae,
                'MAPE': mape
            }
        }

    def train_all_models(self, X_train, y_train, X_test, y_test, use_log=True):
        """
        訓練所有可用模型

        Args:
            X_train: 訓練特徵
            y_train: 訓練目標
            X_test: 測試特徵
            y_test: 測試目標
            use_log: 是否使用對數轉換

        Returns:
            dict: 所有模型的訓練結果
        """
        print(f"\n{'='*70}")
        print("集成學習模型訓練")
        print(f"{'='*70}")
        print(f"\n資料集大小：")
        print(f"  訓練集: {X_train.shape[0]} 筆, {X_train.shape[1]} 個特徵")
        print(f"  測試集: {X_test.shape[0]} 筆")
        print(f"  目標轉換: {'Log(1+y)' if use_log else '原始值'}")

        models = self.get_models()
        results = {}

        for name, model in models.items():
            print(f"\n{'─'*70}")
            print(f"訓練模型: {name}")
            print(f"{'─'*70}")

            result = self.train_single_model(
                name, model, X_train, y_train, X_test, y_test, use_log
            )

            results[name] = result
            self.models[name] = result['model']

            # 顯示結果
            metrics = result['metrics']
            print(f"\n評估指標：")
            print(f"  R² Score:       {metrics['R²']:.4f}")
            print(f"  RMSE:           {metrics['RMSE']:,.0f} 元")
            print(f"  MAE:            {metrics['MAE']:,.0f} 元")
            print(f"  MAPE:           {metrics['MAPE']:.2f}%")

        self.results = results
        return results

    def get_feature_importance(self, model_name, feature_names, top_n=20):
        """
        獲取特徵重要性

        Args:
            model_name: 模型名稱
            feature_names: 特徵名稱列表
            top_n: 顯示前N個重要特徵

        Returns:
            DataFrame: 特徵重要性
        """
        if model_name not in self.models:
            print(f"模型 {model_name} 不存在")
            return None

        model = self.models[model_name]

        # 檢查模型是否有feature_importances_屬性
        if not hasattr(model, 'feature_importances_'):
            print(f"模型 {model_name} 不支援特徵重要性分析")
            return None

        # 獲取特徵重要性
        importance = model.feature_importances_
        feature_importance = pd.DataFrame({
            '特徵': feature_names,
            '重要性': importance
        }).sort_values('重要性', ascending=False)

        print(f"\n{'='*60}")
        print(f"{model_name} - 特徵重要性分析")
        print(f"{'='*60}")
        print(f"\n前 {top_n} 個重要特徵：\n")

        # 標準化重要性到0-100
        max_importance = feature_importance['重要性'].max()
        feature_importance['重要性(%)'] = (
            feature_importance['重要性'] / max_importance * 100
        )

        for i, row in feature_importance.head(top_n).iterrows():
            bars = '█' * int(row['重要性(%)'] / 5)
            print(f"{row['特徵']:30s} {row['重要性(%)']:6.2f}% {bars}")

        return feature_importance

    def cross_validate_model(self, name, model, X, y, cv=5, use_log=True):
        """
        交叉驗證評估模型

        Args:
            name: 模型名稱
            model: 模型對象
            X: 特徵
            y: 目標
            cv: 交叉驗證折數
            use_log: 是否使用對數轉換

        Returns:
            dict: 交叉驗證結果
        """
        print(f"\n{'─'*70}")
        print(f"{cv}-Fold 交叉驗證: {name}")
        print(f"{'─'*70}")

        if use_log:
            y_transformed = np.log1p(y)
        else:
            y_transformed = y

        # 執行交叉驗證
        kfold = KFold(n_splits=cv, shuffle=True, random_state=self.random_state)

        # 使用負MSE作為評分標準（sklearn的慣例）
        cv_scores = cross_val_score(
            model, X, y_transformed,
            cv=kfold,
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )

        # 轉換為RMSE
        cv_rmse = np.sqrt(-cv_scores)

        # 如果使用log轉換，需要轉換回原始尺度來理解RMSE
        # 這裡的RMSE是在log空間的，實際使用時需要注意

        print(f"\n交叉驗證結果：")
        print(f"  平均 RMSE:      {cv_rmse.mean():,.4f}")
        print(f"  標準差:         {cv_rmse.std():,.4f}")
        print(f"  95% 信賴區間:   [{cv_rmse.mean() - 1.96*cv_rmse.std():,.4f}, "
              f"{cv_rmse.mean() + 1.96*cv_rmse.std():,.4f}]")

        return {
            'scores': cv_rmse,
            'mean': cv_rmse.mean(),
            'std': cv_rmse.std()
        }

    def compare_models(self):
        """
        比較所有訓練過的模型

        Returns:
            DataFrame: 模型比較結果
        """
        if not self.results:
            print("尚未訓練任何模型")
            return None

        print(f"\n{'='*70}")
        print("模型效能比較")
        print(f"{'='*70}\n")

        comparison = []
        for name, result in self.results.items():
            metrics = result['metrics']
            comparison.append({
                '模型': name,
                'R²': metrics['R²'],
                'RMSE': metrics['RMSE'],
                'MAE': metrics['MAE'],
                'MAPE (%)': metrics['MAPE']
            })

        comparison_df = pd.DataFrame(comparison).sort_values('R²', ascending=False)

        # 格式化輸出
        print(f"{'模型':<20s} {'R²':>8s} {'RMSE':>12s} {'MAE':>12s} {'MAPE':>10s}")
        print("─" * 70)

        for _, row in comparison_df.iterrows():
            print(f"{row['模型']:<20s} "
                  f"{row['R²']:8.4f} "
                  f"{row['RMSE']:12,.0f} "
                  f"{row['MAE']:12,.0f} "
                  f"{row['MAPE (%)']:9.2f}%")

        # 標註最佳模型
        best_model = comparison_df.iloc[0]['模型']
        print(f"\n✓ 最佳模型（按R²）: {best_model}")

        return comparison_df

    def ensemble_predict(self, X, method='average', weights=None):
        """
        集成預測 - 結合多個模型的預測

        Args:
            X: 特徵
            method: 'average'（平均）或 'weighted'（加權平均）
            weights: 各模型權重（當method='weighted'時使用）

        Returns:
            array: 集成預測結果
        """
        if not self.models:
            raise ValueError("尚未訓練任何模型")

        predictions = []
        model_names = []

        for name, model in self.models.items():
            pred = model.predict(X)
            predictions.append(pred)
            model_names.append(name)

        predictions = np.array(predictions)

        if method == 'average':
            ensemble_pred = np.mean(predictions, axis=0)
        elif method == 'weighted':
            if weights is None:
                # 使用R²作為權重
                weights = [self.results[name]['metrics']['R²']
                          for name in model_names]
            weights = np.array(weights) / np.sum(weights)
            ensemble_pred = np.average(predictions, axis=0, weights=weights)
        else:
            raise ValueError("method 必須是 'average' 或 'weighted'")

        return ensemble_pred
