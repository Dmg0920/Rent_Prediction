# Django Web 應用

租金預測的生產環境 Web 應用。

## 📁 目錄結構

```
webapp/
├── rent_project/       # Django 專案設定
│   ├── settings.py
│   ├── urls.py
│   └── wsgi.py
│
├── predictor/          # 預測應用（如果有）
│
└── manage.py           # Django 管理工具
```

## 🚀 執行開發伺服器

```bash
# 從 webapp/ 目錄執行
cd webapp
python manage.py runserver
```

## 🔗 使用訓練好的模型

### 方法 1: 使用 shared 模組

```python
# 在 Django views 中
import sys
from pathlib import Path

# 新增 shared 到 Python 路徑
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from shared.feature_engineering import FeatureEngineer

# 使用統一的特徵工程
df_encoded = FeatureEngineer.encode_features(df)
```

### 方法 2: 使用 ModelManager

```python
import sys
from pathlib import Path

# 新增 ml 模組到路徑
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ml.src.models.model_utils import ModelManager

# 載入生產模型
manager = ModelManager('ml/src/models/rent_prediction_model.pkl')
manager.load_model()

# 進行預測
predictions = manager.predict(X)
```

## 📝 重要提醒

1. **特徵一致性**: 必須使用 `shared/feature_engineering.py` 中的 `FeatureEngineer`，確保與訓練時的特徵工程邏輯完全一致

2. **模型路徑**: 生產模型位於 `ml/src/models/rent_prediction_model.pkl`

3. **資料預處理**: 在預測前，輸入資料必須經過與訓練時相同的預處理步驟

## 🔧 部署建議

部署到生產環境時：
- 只需要部署 `webapp/` 目錄
- 複製 `shared/` 目錄到部署環境
- 複製訓練好的模型檔案 `ml/src/models/rent_prediction_model.pkl`
- 不需要部署 `ml/` 的其他檔案（資料、訓練腳本等）
