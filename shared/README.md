# 共用模組 (Shared)

ML 訓練環境和 Web 應用都會使用的共用程式碼。

## 📦 模組內容

### FeatureEngineer - 特徵工程器

統一的特徵工程邏輯，確保訓練和預測時使用完全相同的特徵轉換。

```python
from shared.feature_engineering import FeatureEngineer

# 計算屋齡
df = FeatureEngineer.calculate_house_age(df)

# 提取樓層特徵
df = FeatureEngineer.extract_floor_feature(df)

# 特徵編碼（One-Hot Encoding）
df_encoded = FeatureEngineer.encode_features(df)
```

## 🎯 設計目的

### 問題：為什麼需要 shared 模組？

在重構前，專案有以下問題：
1. `ml/src/models/train_model.py` 有特徵工程程式碼
2. `ml/src/models/build_production_model.py` 也有特徵工程程式碼
3. 兩份程式碼可能不一致，導致訓練和預測結果不同

### 解決方案：單一真相來源 (Single Source of Truth)

將特徵工程邏輯統一放在 `shared/feature_engineering.py`：
- ML 訓練時使用這份程式碼
- Web 預測時也使用這份程式碼
- 確保完全一致，避免訓練/預測不匹配的問題

## 📝 使用方式

### 在 ML 訓練中使用

```python
# ml/src/models/train_model.py
from preprocessing.feature_engineering import FeatureEngineer

df_encoded = FeatureEngineer.encode_features(df)
```

### 在 Web 應用中使用

```python
# webapp/predictor/views.py
import sys
from pathlib import Path

# 新增 shared 到路徑
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from shared.feature_engineering import FeatureEngineer

df_encoded = FeatureEngineer.encode_features(df)
```

## 🔧 維護指南

⚠️ **重要**:
- 如果需要修改特徵工程邏輯，**只能修改 `shared/feature_engineering.py`**
- 修改後需要重新訓練模型
- 不要在其他地方重複實作特徵工程邏輯

## 📂 檔案同步

`shared/feature_engineering.py` 是從 `ml/src/preprocessing/feature_engineering.py` 複製的。

如果更新了 `ml/src/preprocessing/feature_engineering.py`，記得同步更新 `shared/feature_engineering.py`。
