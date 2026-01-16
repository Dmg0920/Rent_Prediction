# 使用指南

## 快速開始

### 1. 執行完整資料處理流程

```bash
# 啟動虛擬環境
source .venv/bin/activate

# 執行資料處理流程
python scripts/data_pipeline.py
```

這將執行以下步驟：
1. 載入台北和新北市的原始資料
2. 計算基礎特徵（坪數、每坪租金）
3. 資料清洗（移除非住宅、異常值）
4. 特徵工程（屋齡、樓層）
5. 特徵編碼（One-Hot Encoding）
6. 產生視覺化圖表

### 2. 作為模組使用

```python
from src.preprocessing import DataLoader, DataCleaner, FeatureEngineer, DataVisualizer

# 載入資料
loader = DataLoader(data_dir='data')
df = loader.load_raw_data()
df = loader.add_basic_features(df)

# 清洗資料
cleaner = DataCleaner()
df_clean = cleaner.clean_pipeline(df, remove_outliers=True)

# 特徵工程
engineer = FeatureEngineer()
df_clean = engineer.calculate_house_age(df_clean)
df_clean = engineer.extract_floor_feature(df_clean)
df_encoded = engineer.encode_features(df_clean)

# 視覺化
visualizer = DataVisualizer(output_dir='output/visualizations')
visualizer.plot_rent_distribution(df_clean)
visualizer.plot_correlation_matrix(df_encoded)
```

### 3. 訓練模型

```bash
# 執行模型訓練腳本
python src/models/train_model.py
```

## 模組說明

### DataLoader - 資料載入器
負責從CSV檔案載入和初步處理資料

**主要方法：**
- `load_raw_data()` - 載入原始資料
- `add_basic_features()` - 新增坪數和每坪租金
- `load_cleaned_data()` - 載入已清洗的資料
- `load_featured_data()` - 載入特徵工程後的資料

### DataCleaner - 資料清洗器
負責資料清洗和異常值處理

**主要方法：**
- `remove_non_residential()` - 移除非住宅類型
- `detect_outliers_iqr()` - 檢測異常值
- `remove_outliers_iqr()` - 移除異常值
- `clean_pipeline()` - 完整清洗流程

### FeatureEngineer - 特徵工程器
負責特徵建立和轉換

**主要方法：**
- `calculate_house_age()` - 計算房屋年齡
- `extract_floor_feature()` - 提取樓層特徵
- `encode_features()` - 特徵編碼

### DataVisualizer - 資料視覺化器
負責產生各種分析圖表

**主要方法：**
- `plot_rent_distribution()` - 租金分佈圖
- `plot_correlation_matrix()` - 相關性矩陣
- `plot_feature_importance()` - 特徵重要性圖
- `plot_scatter()` - 散點圖

## 輸出檔案

### 資料檔案
- `data/taipei_newtaipei_cleaned.csv` - 清洗後的資料
- `data/taipei_newtaipei_featured.csv` - 特徵工程後的資料

### 視覺化圖表
- `01_raw_rent_distribution.png` - 原始租金分佈
- `02_cleaned_rent_distribution.png` - 清洗後租金分佈
- `03_correlation_matrix.png` - 相關性矩陣熱圖
- `04_age_vs_rent.png` - 屋齡與租金關係
- `05_floor_vs_rent.png` - 樓層與租金關係

## 自訂設定

### 修改異常值檢測參數

```python
# 使用更嚴格的異常值檢測（1.0倍IQR）
df_clean = DataCleaner.remove_outliers_iqr(df, multiplier=1.0)

# 使用更寬鬆的異常值檢測（2.0倍IQR）
df_clean = DataCleaner.remove_outliers_iqr(df, multiplier=2.0)
```

### 自訂視覺化

```python
# 顯示更多相關特徵
visualizer.plot_correlation_matrix(df, top_k=20)

# 儲存到自訂位置
visualizer = DataVisualizer(output_dir='custom/path')
```

## 常見問題

### Q: 如何只執行部分流程？
A: 直接匯入需要的模組並呼叫相應方法即可，不需要執行完整pipeline。

### Q: 如何修改資料路徑？
A: 在初始化DataLoader時指定data_dir參數：
```python
loader = DataLoader(data_dir='your/custom/path')
```

### Q: 如何除錯某個步驟？
A: 每個模組都是獨立的，可以單獨匯入測試：
```python
from src.preprocessing.data_cleaner import DataCleaner
# 測試清洗功能
```

### Q: 為什麼模型訓練要先執行 data_pipeline.py？
A: 因為模型訓練需要特徵工程後的資料 (`taipei_newtaipei_featured.csv`)，這個檔案由 data_pipeline.py 產生。

## 完整工作流程範例

```python
from src.preprocessing import (
    DataLoader,
    DataCleaner,
    FeatureEngineer,
    DataVisualizer
)

# 1. 載入資料
loader = DataLoader()
df = loader.load_raw_data()
df = loader.add_basic_features(df)

# 2. 清洗資料
cleaner = DataCleaner()
df = cleaner.clean_pipeline(df, remove_outliers=True)

# 3. 特徵工程
engineer = FeatureEngineer()
df = engineer.calculate_house_age(df)
df = engineer.extract_floor_feature(df)
df = engineer.encode_features(df)

# 4. 視覺化
viz = DataVisualizer()
viz.plot_rent_distribution(df)
viz.plot_correlation_matrix(df)
```

## 部分特徵工程範例

```python
# 只計算屋齡
df = FeatureEngineer.calculate_house_age(df)

# 只提取樓層
df = FeatureEngineer.extract_floor_feature(df)

# 不編碼，只簡化建築類型
df['建物型態_簡化'] = df['建物型態'].apply(
    FeatureEngineer.simplify_building_type
)
```
