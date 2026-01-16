# 專案目錄結構

重構後的完整目錄結構（2026-01-16）

## 根目錄
```
Rent_Prediction_Web/
├── README.md                    # 專案總覽
├── RESTRUCTURE_PLAN.md          # 重構計畫文件
├── PROJECT_SUMMARY.md           # 專案摘要
├── DIRECTORY_STRUCTURE.md       # 本檔案
├── .gitignore                   # Git 忽略設定
├── .venv/                       # Python 虛擬環境
│
├── ml/                          # ML 開發環境
├── webapp/                      # Django Web 應用
├── shared/                      # 共用模組
└── docs/                        # 文件
```

## ml/ - ML 開發環境
```
ml/
├── README.md                    # ML 環境說明
│
├── data/                        # 資料檔案
│   ├── a_lvr_land_c.csv        # 台北市原始資料
│   ├── f_lvr_land_c.csv        # 新北市原始資料
│   ├── taipei_newtaipei_cleaned.csv    # 清洗後資料
│   └── taipei_newtaipei_featured.csv   # 特徵工程後資料
│
├── src/                         # ML 原始碼
│   ├── preprocessing/           # 資料預處理
│   │   ├── __init__.py
│   │   ├── data_loader.py      # 資料載入器
│   │   ├── data_cleaner.py     # 資料清洗器
│   │   ├── feature_engineering.py  # 特徵工程器
│   │   └── visualizer.py       # 視覺化工具
│   │
│   └── models/                  # 模型訓練
│       ├── train_model.py       # 模型訓練腳本
│       ├── build_production_model.py  # 生產模型建置
│       ├── model_utils.py       # 模型管理工具
│       └── rent_prediction_model.pkl  # 生產模型（訓練後產生）
│
├── scripts/                     # 執行腳本
│   └── data_pipeline.py        # 完整資料處理流程
│
└── output/                      # 輸出結果
    └── visualizations/          # 視覺化圖表
        ├── 01_raw_rent_distribution.png
        ├── 02_cleaned_rent_distribution.png
        ├── 03_correlation_matrix.png
        ├── 04_age_vs_rent.png
        └── 05_floor_vs_rent.png
```

## webapp/ - Django Web 應用
```
webapp/
├── README.md                    # Web 應用說明
├── manage.py                    # Django 管理工具
│
└── rent_project/                # Django 專案
    ├── __init__.py
    ├── settings.py              # 專案設定
    ├── urls.py                  # URL 路由
    ├── asgi.py
    └── wsgi.py
```

## shared/ - 共用模組
```
shared/
├── README.md                    # 共用模組說明
├── __init__.py
└── feature_engineering.py       # 統一的特徵工程邏輯
```

## docs/ - 文件
```
docs/
└── USAGE.md                     # 詳細使用指南
```

## 執行路徑說明

所有腳本都應從專案根目錄執行：

```bash
# 正確 ✓
python ml/scripts/data_pipeline.py
python ml/src/models/train_model.py

# 錯誤 ✗
cd ml/scripts && python data_pipeline.py
```

## 路徑設計原則

1. **資料路徑**: 都以 `ml/data/` 開頭
2. **輸出路徑**: 都以 `ml/output/` 開頭
3. **模型路徑**: 都以 `ml/src/models/` 開頭
4. **Import 路徑**: ML 模組使用相對 import（從 `ml/src`）

## 重要檔案位置

| 檔案類型 | 位置 |
|---------|------|
| 原始資料 | `ml/data/*.csv` |
| 處理後資料 | `ml/data/taipei_newtaipei_*.csv` |
| 生產模型 | `ml/src/models/rent_prediction_model.pkl` |
| 視覺化圖表 | `ml/output/visualizations/*.png` |
| 特徵工程 | `shared/feature_engineering.py` |

## 清理記錄

已刪除的舊檔案/目錄：
- `src/` - 舊的原始碼目錄（已移至 `ml/src/`）
- `data/` - 舊的資料目錄（已移至 `ml/data/`）
- `output/` - 舊的輸出目錄（已移至 `ml/output/`）
- `scripts/` - 舊的腳本目錄（已移至 `ml/scripts/`）
- `ml/notebooks/` - 空目錄
- `ml/output/models/` - 空目錄
