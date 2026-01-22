import os
import sys
import joblib
import pandas as pd
import numpy as np
from django.shortcuts import render
from django.conf import settings

# --- 1. 設定路徑以匯入 shared ---
PROJECT_ROOT = os.path.abspath(os.path.join(str(settings.BASE_DIR), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# --- 2. 載入模型 (全域載入，避免每次請求都重讀) ---
MODEL_PATH = os.path.join(PROJECT_ROOT, 'ml', 'src', 'models', 'rent_prediction_model.pkl')

model_artifacts = {}
try:
    print(f"正在載入模型: {MODEL_PATH}")
    if os.path.exists(MODEL_PATH):
        model_artifacts = joblib.load(MODEL_PATH)
        print("✅ 模型載入成功！")
        print(f"   模型類型: {model_artifacts.get('metadata', {}).get('model_type', 'Unknown')}")
        print(f"   特徵數量: {len(model_artifacts.get('features', []))}")
    else:
        print("❌ 找不到模型檔案，請確認路徑。")
except Exception as e:
    print(f"❌ 模型載入發生錯誤: {e}")


# --- 3. View 函數 ---
def home(request):
    prediction = None
    error_msg = None

    feature_names = model_artifacts.get('features', [])
    model = model_artifacts.get('model')
    scaler = model_artifacts.get('scaler')

    if request.method == 'POST':
        if not model:
            error_msg = "系統錯誤：模型未載入，無法預測。"
        else:
            try:
                # A. 接收表單資料
                ping = float(request.POST.get('area'))  # 坪數
                age = float(request.POST.get('age'))  # 屋齡
                floor = float(request.POST.get('floor'))  # 樓層
                total_floor = float(request.POST.get('total_floor'))  # 總樓層數
                rooms = int(request.POST.get('rooms'))  # 房
                living_rooms = int(request.POST.get('living_rooms'))  # 廳
                bathrooms = int(request.POST.get('bathrooms'))  # 衛
                parking_area = float(request.POST.get('parking_area') or 0)  # 車位面積
                has_elevator = int(request.POST.get('has_elevator') or 0)
                has_manager = int(request.POST.get('has_manager') or 0)
                has_furniture = int(request.POST.get('has_furniture') or 0)

                # B. 建立特徵字典（對應模型訓練時的特徵）
                # 1 坪 = 3.30579 平方公尺
                building_area_m2 = ping * 3.30579

                input_data = {
                    '土地面積平方公尺': 0,  # 通常使用者不知道，設為 0
                    '非都市土地使用分區': 0,
                    '非都市土地使用編定': 0,
                    '總樓層數': total_floor,
                    '建物總面積平方公尺': building_area_m2,
                    '建物現況格局-房': rooms,
                    '建物現況格局-廳': living_rooms,
                    '建物現況格局-衛': bathrooms,
                    '有無管理組織': has_manager,
                    '有無附傢俱': has_furniture,
                    '車位面積平方公尺': parking_area,
                    '有無電梯': has_elevator,
                    '坪數': ping,
                    '屋齡': age,
                    '樓層': floor,
                }

                # C. 轉成 DataFrame
                df = pd.DataFrame([input_data])

                # D. 特徵對齊 - 確保欄位順序和訓練時一致
                df_aligned = df.reindex(columns=feature_names, fill_value=0)

                # E. 標準化 & 預測
                X_scaled = scaler.transform(df_aligned)

                # 模型預測 (log 空間)
                log_price = model.predict(X_scaled)[0]

                # F. 還原價格
                final_price = np.expm1(log_price)
                prediction = int(round(final_price))

            except ValueError as e:
                error_msg = f"請確認所有欄位都已正確填寫: {str(e)}"
            except Exception as e:
                print(f"預測錯誤: {e}")
                error_msg = f"預測過程發生錯誤: {str(e)}"

    return render(request, 'home.html', {'prediction': prediction, 'error': error_msg})
