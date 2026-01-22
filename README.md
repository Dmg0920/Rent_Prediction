# 租金預測專案 (Rent Prediction Web)

基於台北和新北市房屋資料的租金預測機器學習專案，包含完整的資料處理 pipeline 和 Django Web 應用。

## 專案特色

- **清晰架構** - ML 開發和 Web 部署完全分離
- **集成學習** - 支援 Random Forest、XGBoost、LightGBM 等模型
- **完整 Pipeline** - 從原始資料到生產模型的完整流程
- **即時預測** - Django Web 介面，輸入房屋資訊即可預測租金

## 專案結構

```
Rent_Prediction_Web/
├── ml/                          # ML 開發環境
│   ├── data/                    # 原始和處理後的資料
│   ├── src/
│   │   ├── preprocessing/       # 資料預處理模組
│   │   ├── models/              # 模型訓練
│   │   └── analysis/            # 統計分析
│   └── scripts/                 # 執行腳本
├── webapp/                      # Django Web 應用
│   ├── predictor/               # 預測應用
│   └── templates/               # HTML 模板
├── shared/                      # 共用模組
│   └── feature_engineering.py   # 特徵工程
└── docs/                        # 文件
```

## 快速開始

```bash
# 1. 資料處理
python ml/scripts/data_pipeline.py

# 2. 訓練模型
python ml/src/models/train_model.py

# 3. 啟動 Web 應用
cd webapp && python manage.py runserver
```

開啟瀏覽器訪問 http://127.0.0.1:8000

## 技術架構

### 資料流程

```
原始資料 (CSV)
    ↓
DataLoader (載入、計算坪數)
    ↓
DataCleaner (移除非住宅、異常值)
    ↓
FeatureEngineer (屋齡、樓層、編碼)
    ↓
模型訓練 (Random Forest)
    ↓
生產模型 (.pkl)
    ↓
Django Web 預測
```

### 模型

| 模型 | R² | RMSE | 說明 |
|------|-----|------|------|
| **Random Forest** | 0.67 | 392 元 | 生產環境使用 |
| Gradient Boosting | 0.67 | 393 元 | |
| XGBoost | 0.66 | 394 元 | |
| LightGBM | 0.64 | 405 元 | |

### 特徵 (15 個)

| 類別 | 特徵 |
|------|------|
| 面積 | 坪數、建物總面積、土地面積、車位面積 |
| 樓層 | 所在樓層、總樓層數 |
| 格局 | 房數、廳數、衛數 |
| 屋況 | 屋齡 |
| 設施 | 電梯、管理組織、附傢俱 |
| 其他 | 非都市土地使用分區/編定 |

### 技術棧

- **ML**: scikit-learn, XGBoost, LightGBM, statsmodels
- **資料處理**: pandas, numpy
- **Web**: Django 4.2
- **序列化**: joblib

## Web 應用

### 輸入欄位

- **基本資訊**: 坪數、屋齡、樓層、總樓層數
- **格局**: 房/廳/衛
- **設施**: 電梯、管理組織、附傢俱
- **車位**: 車位面積 (平方公尺)

### 輸出

預測每坪租金 (元/坪/月)

## 重要指令

```bash
# 資料處理
python ml/scripts/data_pipeline.py

# 訓練基礎模型 (Ridge, Lasso)
python ml/src/models/train_model.py

# 進階統計分析
python ml/scripts/advanced_analysis.py

# 啟動 Web
cd webapp && python manage.py runserver
```

## 注意事項

1. 所有腳本從專案根目錄執行
2. 模型使用 log 轉換訓練，預測時自動還原
3. 修改特徵工程後需重新訓練模型

## 部署

生產環境需要：
- `webapp/` - Django 應用
- `shared/` - 特徵工程模組
- `ml/src/models/rent_prediction_model.pkl` - 模型檔案

## 資料來源

台北市、新北市政府實價登錄開放資料
