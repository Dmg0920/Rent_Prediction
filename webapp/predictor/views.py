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

# --- 2. 載入模型和市場統計 ---
MODEL_PATH = os.path.join(PROJECT_ROOT, 'ml', 'src', 'models', 'rent_prediction_model.pkl')
STATS_PATH = os.path.join(PROJECT_ROOT, 'ml', 'src', 'models', 'market_stats.pkl')

model_artifacts = {}
market_stats = {}

try:
    print(f"正在載入模型: {MODEL_PATH}")
    if os.path.exists(MODEL_PATH):
        model_artifacts = joblib.load(MODEL_PATH)
        print("✅ 模型載入成功！")

    if os.path.exists(STATS_PATH):
        market_stats = joblib.load(STATS_PATH)
        print("✅ 市場統計載入成功！")
except Exception as e:
    print(f"❌ 載入發生錯誤: {e}")


def get_area_range(ping):
    """根據坪數返回區間標籤"""
    if ping < 10:
        return '小於10坪'
    elif ping < 20:
        return '10-20坪'
    elif ping < 30:
        return '20-30坪'
    elif ping < 40:
        return '30-40坪'
    else:
        return '40坪以上'


def get_age_range(age):
    """根據屋齡返回區間標籤"""
    if age < 5:
        return '新屋(0-5年)'
    elif age < 15:
        return '中古(5-15年)'
    elif age < 30:
        return '老屋(15-30年)'
    else:
        return '超老屋(30年+)'


def calculate_percentile(value, percentiles):
    """計算數值在分佈中的百分位"""
    if value <= percentiles[10]:
        return 10
    elif value <= percentiles[25]:
        return 10 + 15 * (value - percentiles[10]) / (percentiles[25] - percentiles[10])
    elif value <= percentiles[50]:
        return 25 + 25 * (value - percentiles[25]) / (percentiles[50] - percentiles[25])
    elif value <= percentiles[75]:
        return 50 + 25 * (value - percentiles[50]) / (percentiles[75] - percentiles[50])
    elif value <= percentiles[90]:
        return 75 + 15 * (value - percentiles[75]) / (percentiles[90] - percentiles[75])
    else:
        return 90 + 10 * min(1, (value - percentiles[90]) / (percentiles[90] * 0.5))


def get_price_evaluation(rent_per_ping, area_avg, overall_avg):
    """評估租金水平"""
    # 與類似坪數比較
    if area_avg > 0:
        ratio = rent_per_ping / area_avg
        if ratio < 0.85:
            return {'level': 'low', 'text': '便宜', 'color': '#27ae60', 'description': '低於同坪數平均'}
        elif ratio < 1.0:
            return {'level': 'below_avg', 'text': '略低', 'color': '#2ecc71', 'description': '略低於同坪數平均'}
        elif ratio < 1.15:
            return {'level': 'normal', 'text': '合理', 'color': '#3498db', 'description': '接近同坪數平均'}
        elif ratio < 1.3:
            return {'level': 'above_avg', 'text': '略高', 'color': '#f39c12', 'description': '略高於同坪數平均'}
        else:
            return {'level': 'high', 'text': '偏高', 'color': '#e74c3c', 'description': '高於同坪數平均'}
    return {'level': 'unknown', 'text': '無資料', 'color': '#95a5a6', 'description': ''}


def get_market_comparison(rent_per_ping, ping, rooms, age):
    """獲取市場比較資料"""
    if not market_stats:
        return None

    overall = market_stats.get('overall', {})
    by_area = market_stats.get('by_area', {})
    by_rooms = market_stats.get('by_rooms', {})
    by_age = market_stats.get('by_age', {})

    # 1. 計算百分位排名
    percentiles = overall.get('percentiles', {})
    if percentiles:
        percentile = calculate_percentile(rent_per_ping, percentiles)
    else:
        percentile = 50

    # 2. 同坪數區間比較
    area_range = get_area_range(ping)
    area_data = by_area.get(area_range, {})
    area_avg = area_data.get('mean', 0)
    area_count = area_data.get('count', 0)

    # 3. 同房數比較 (限制在合理範圍)
    room_key = min(rooms, 5)  # 超過5房統一用5房
    room_data = by_rooms.get(room_key, by_rooms.get(float(room_key), {}))
    room_avg = room_data.get('mean', 0)

    # 4. 同屋齡比較
    age_range = get_age_range(age)
    age_data = by_age.get(age_range, {})
    age_avg = age_data.get('mean', 0)

    # 5. 租金評價
    evaluation = get_price_evaluation(rent_per_ping, area_avg, overall.get('mean', 0))

    # 6. 計算與各類平均的差異
    overall_avg = overall.get('mean', 0)

    return {
        'percentile': int(percentile),
        'percentile_text': f"比 {100 - int(percentile)}% 的物件便宜" if percentile < 50 else f"比 {int(percentile)}% 的物件貴",

        'overall_avg': int(overall_avg),
        'overall_diff': int(rent_per_ping - overall_avg),
        'overall_diff_pct': int((rent_per_ping - overall_avg) / overall_avg * 100) if overall_avg > 0 else 0,

        'area_range': area_range,
        'area_avg': int(area_avg),
        'area_diff': int(rent_per_ping - area_avg) if area_avg > 0 else 0,
        'area_diff_pct': int((rent_per_ping - area_avg) / area_avg * 100) if area_avg > 0 else 0,
        'area_count': int(area_count),

        'room_avg': int(room_avg),
        'room_diff': int(rent_per_ping - room_avg) if room_avg > 0 else 0,

        'age_range': age_range,
        'age_avg': int(age_avg),

        'evaluation': evaluation,
        'total_samples': market_stats.get('total_samples', 0),
    }


# --- 3. View 函數 ---
def home(request):
    result = None
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
                ping = float(request.POST.get('area'))
                age = float(request.POST.get('age'))
                floor = float(request.POST.get('floor'))
                total_floor = float(request.POST.get('total_floor'))
                rooms = int(request.POST.get('rooms'))
                living_rooms = int(request.POST.get('living_rooms'))
                bathrooms = int(request.POST.get('bathrooms'))
                parking_area = float(request.POST.get('parking_area') or 0)
                has_elevator = int(request.POST.get('has_elevator') or 0)
                has_manager = int(request.POST.get('has_manager') or 0)
                has_furniture = int(request.POST.get('has_furniture') or 0)

                # B. 建立特徵字典
                building_area_m2 = ping * 3.30579

                input_data = {
                    '土地面積平方公尺': 0,
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

                # C. 轉成 DataFrame 並對齊特徵
                df = pd.DataFrame([input_data])
                df_aligned = df.reindex(columns=feature_names, fill_value=0)

                # D. 標準化 & 預測
                X_scaled = scaler.transform(df_aligned)
                log_price = model.predict(X_scaled)[0]

                # E. 還原價格
                rent_per_ping = np.expm1(log_price)
                total_rent = rent_per_ping * ping

                # F. 取得市場比較
                market_comparison = get_market_comparison(rent_per_ping, ping, rooms, age)

                # G. 組合結果
                result = {
                    'rent_per_ping': int(round(rent_per_ping)),
                    'total_rent': int(round(total_rent)),
                    'area': ping,
                    'rooms': rooms,
                    'living_rooms': living_rooms,
                    'bathrooms': bathrooms,
                    'floor': int(floor),
                    'total_floor': int(total_floor),
                    'age': int(age),
                    'has_elevator': has_elevator,
                    'has_manager': has_manager,
                    'has_furniture': has_furniture,
                    'market': market_comparison,
                }

            except ValueError as e:
                error_msg = "請確認所有欄位都已正確填寫"
            except Exception as e:
                print(f"預測錯誤: {e}")
                error_msg = f"預測過程發生錯誤: {str(e)}"

    return render(request, 'home.html', {'result': result, 'error': error_msg})
