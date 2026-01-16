# 專案重整計畫

## 🎯 目標
將混合的 ML 訓練和 Web 應用分離，建立清晰的專案架構

## 📋 現況問題

1. **混亂的目錄結構**
   - ML 程式碼 (`src/`, `scripts/`) 和 Django (`rent_project/`) 混在一起
   - 不清楚哪些是訓練用，哪些是部署用

2. **重複的程式碼**
   - `build_production_model.py` 又重複了特徵工程邏輯
   - 沒有統一使用 `FeatureEngineer`

3. **模型管理不清**
   - 模型檔案位置不明確
   - 訓練和部署使用不同的程式碼

## 🏗️ 建議架構

### 方案 A：完全分離（推薦）
```
Rent_Prediction_Web/
├── ml/                    # ML 開發環境
│   ├── data/
│   ├── src/
│   ├── scripts/
│   ├── output/
│   └── notebooks/
│
├── webapp/                # Django 生產環境
│   ├── rent_project/
│   ├── predictor/
│   ├── models/           # 只放生產模型
│   └── manage.py
│
└── shared/               # 共用程式碼
    └── preprocessing/
```

**優點：**
- 清晰分離開發和生產
- 可獨立部署 Web
- ML 實驗不影響 Web

**缺點：**
- 需要移動檔案
- 需要更新 import 路徑

### 方案 B：簡化整合
```
Rent_Prediction_Web/
├── ml_pipeline/          # 所有 ML 相關
│   ├── preprocessing/
│   ├── training/
│   ├── data/
│   └── output/
│
├── rent_project/        # Django
├── manage.py
└── models/              # 生產模型（頂層）
```

**優點：**
- 改動較小
- 保持整合性

**缺點：**
- 仍然有點混亂
- 部署時需要排除 ML 程式碼

## 🔧 立即可做的改善（不移動檔案）

1. **統一特徵工程**
   - 修改 `build_production_model.py` 使用 `FeatureEngineer`
   - 確保訓練和部署用同一套邏輯

2. **建立模型管理工具**
   - 建立 `src/models/model_manager.py`
   - 統一模型載入、儲存、驗證

3. **整理文件**
   - 清楚標示各目錄用途
   - 建立使用流程圖

## ❓ 你想要哪種方案？

我可以幫你：
1. **方案 A** - 完全重構（最乾淨，但改動大）
2. **方案 B** - 簡化整合（改動中等）
3. **立即改善** - 不移動檔案，只修正程式碼（改動最小）

請告訴我你的選擇！
