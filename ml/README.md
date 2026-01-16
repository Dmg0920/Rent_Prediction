# ML 訓練環境

機器學習模型開發、訓練和實驗的完整環境。

## 📁 目錄結構

```
ml/
├── data/                    # 資料檔案
│   ├── a_lvr_land_c.csv    # 台北市原始資料
│   ├── f_lvr_land_c.csv    # 新北市原始資料
│   ├── taipei_newtaipei_cleaned.csv    # 清洗後資料
│   └── taipei_newtaipei_featured.csv   # 特徵工程後資料
│
├── src/                     # 原始碼
│   ├── preprocessing/       # 資料預處理模組
│   │   ├── data_loader.py          # 資料載入
│   │   ├── data_cleaner.py         # 資料清洗
│   │   ├── feature_engineering.py  # 特徵工程
│   │   └── visualizer.py           # 視覺化
│   └── models/             # 模型訓練
│       ├── train_model.py           # 模型訓練腳本
│       ├── build_production_model.py # 生產模型建置
│       ├── model_utils.py           # 模型工具
│       └── rent_prediction_model.pkl # 生產模型（訓練後產生）
│
├── scripts/                # 執行腳本
│   └── data_pipeline.py    # 資料處理主流程
│
└── output/                 # 輸出結果
    └── visualizations/     # 視覺化圖表
```

## 🚀 快速開始

### 1. 執行完整資料處理流程

```bash
# 從專案根目錄執行
python ml/scripts/data_pipeline.py
```

這會執行：
- 載入台北和新北市原始資料
- 計算基礎特徵（坪數、每坪租金）
- 資料清洗（移除非住宅、異常值）
- 特徵工程（屋齡、樓層）
- 特徵編碼（One-Hot Encoding）
- 產生視覺化圖表

### 2. 訓練模型（開發用）

```bash
# 從專案根目錄執行
python ml/src/models/train_model.py
```

會訓練並評估多個模型：
- Linear Regression
- Ridge (L2)
- Lasso (L1)

### 3. 建置生產模型

```bash
# 從專案根目錄執行
python ml/src/models/build_production_model.py
```

使用 100% 資料訓練最終模型，並儲存至 `ml/src/models/rent_prediction_model.pkl`。

## 📦 模組說明

### DataLoader - 資料載入器
```python
from ml.src.preprocessing.data_loader import DataLoader

loader = DataLoader(data_dir='ml/data')
df = loader.load_raw_data()
df = loader.add_basic_features(df)
```

### DataCleaner - 資料清洗器
```python
from ml.src.preprocessing.data_cleaner import DataCleaner

cleaner = DataCleaner()
df_clean = cleaner.clean_pipeline(df, remove_outliers=True)
```

### FeatureEngineer - 特徵工程器
```python
from ml.src.preprocessing.feature_engineering import FeatureEngineer

df = FeatureEngineer.calculate_house_age(df)
df = FeatureEngineer.extract_floor_feature(df)
df_encoded = FeatureEngineer.encode_features(df)
```

### ModelManager - 模型管理器
```python
from ml.src.models.model_utils import ModelManager

manager = ModelManager('ml/src/models/rent_prediction_model.pkl')
manager.load_model()
predictions = manager.predict(X)
```

## 📊 輸出檔案

### 資料檔案
- `ml/data/taipei_newtaipei_cleaned.csv` - 清洗後的資料
- `ml/data/taipei_newtaipei_featured.csv` - 特徵工程後的資料

### 視覺化圖表
- `01_raw_rent_distribution.png` - 原始租金分佈
- `02_cleaned_rent_distribution.png` - 清洗後租金分佈
- `03_correlation_matrix.png` - 相關性矩陣
- `04_age_vs_rent.png` - 屋齡與租金關係
- `05_floor_vs_rent.png` - 樓層與租金關係

### 模型檔案
- `ml/src/models/rent_prediction_model.pkl` - 生產環境模型

## 🔗 與 Web 應用整合

生產模型會被 Django Web 應用使用。Web 應用透過 `shared/feature_engineering.py` 使用相同的特徵工程邏輯，確保訓練和預測的一致性。

## 📝 注意事項

1. **執行路徑**: 所有腳本都應從專案根目錄執行
2. **資料依賴**: 訓練模型前必須先執行 `data_pipeline.py`
3. **特徵一致性**: 使用統一的 `FeatureEngineer` 確保特徵工程邏輯一致
