import os
import sys
import joblib
import pandas as pd
import numpy as np
from django.shortcuts import render
from django.conf import settings

# --- 1. 設定路徑以匯入 shared ---
# BASE_DIR 是 webapp 資料夾，需要將項目根目錄加入路徑以導入 shared 包
PROJECT_ROOT = os.path.abspath(os.path.join(str(settings.BASE_DIR), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from shared import FeatureEngineer

# --- 2. 載入模型 (全域載入，避免每次請求都重讀) ---
# 模型檔案位於專案根目錄的 ml/src/models/rent_prediction_model.pkl
MODEL_PATH = os.path.join(PROJECT_ROOT, 'ml', 'src', 'models', 'rent_prediction_model.pkl')

model_artifacts = {}
try:
    print(f"正在載入模型: {MODEL_PATH}")
    if os.path.exists(MODEL_PATH):
        model_artifacts = joblib.load(MODEL_PATH)
        print("✅ 模型載入成功！")
    else:
        print("❌ 找不到模型檔案，請確認路徑。")
except Exception as e:
    print(f"❌ 模型載入發生錯誤: {e}")

# --- 3. View 函數 ---
def home(request):
    prediction = None
    error_msg = None
    
    # 取得模型需要的特徵列表 (從訓練時的 artifacts 取得)
    feature_names = model_artifacts.get('features', [])
    model = model_artifacts.get('model')
    scaler = model_artifacts.get('scaler')

    if request.method == 'POST':
        if not model:
            error_msg = "系統錯誤：模型未載入，無法預測。"
        else:
            try:
                # A. 接收表單資料
                # 注意：表單的 name 屬性必須與這裡對應
                input_data = {
                    '城市': request.POST.get('city'),
                    '鄉鎮市區': request.POST.get('district'),
                    '建物型態': request.POST.get('building_type'),
                    
                    # 數值型直接轉 float
                    '坪數': float(request.POST.get('area')),
                    '屋齡': float(request.POST.get('age')),
                    '樓層': float(request.POST.get('floor')),
                    
                    # 二元變數 (表單回傳 '1' 或 '0'，如果沒勾選則為 None，預設為 0)
                    '有無電梯': int(request.POST.get('has_elevator') or 0),
                    '有無管理組織': int(request.POST.get('has_manager') or 0),
                    '有無附傢俱': int(request.POST.get('has_furniture') or 0),
                }

                # 轉成 DataFrame (單列)
                df = pd.DataFrame([input_data])

                # B. 特徵工程 (使用 Shared Logic)
                # 1. 簡化建物型態 (呼叫共用的邏輯)
                df['建物型態_簡化'] = df['建物型態'].apply(FeatureEngineer.simplify_building_type)

                # 2. One-Hot Encoding
                # 注意：必須與訓練時的編碼方式一致（drop_first=True）
                df_encoded = pd.get_dummies(
                    df, 
                    columns=['城市', '鄉鎮市區', '建物型態_簡化'], 
                    prefix=['城市', '鄉鎮市區', '建物型態_簡化'],
                    drop_first=True
                )

                # C. 關鍵步驟：特徵對齊 (Reindex)
                # 這是最容易出錯的地方！強迫 DataFrame 欄位與訓練時完全一致
                # 缺少的欄位補 0，多餘的欄位丟掉
                df_aligned = df_encoded.reindex(columns=feature_names, fill_value=0)

                # D. 標準化 & 預測
                X_scaled = scaler.transform(df_aligned)
                
                # 模型預測出來的是 Log(Price)
                log_price = model.predict(X_scaled)[0]
                
                # E. 還原價格 (Exponential)
                final_price = np.expm1(log_price)
                prediction = int(round(final_price))

            except Exception as e:
                print(f"預測錯誤: {e}")
                # 為了 Debug，可以把錯誤印在網頁上 (正式上線建議隱藏)
                error_msg = f"輸入資料有誤或處理失敗: {str(e)}"

    return render(request, 'home.html', {'prediction': prediction, 'error': error_msg})