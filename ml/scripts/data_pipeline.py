#!/usr/bin/env python
"""
資料處理主流程腳本
整合所有資料預處理步驟
"""
import sys
from pathlib import Path

# 新增src目錄到Python路徑
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

from preprocessing.data_loader import DataLoader
from preprocessing.data_cleaner import DataCleaner
from preprocessing.feature_engineering import FeatureEngineer
from preprocessing.visualizer import DataVisualizer


def main():
    """主流程"""
    print("\n" + "="*60)
    print("租金預測專案 - 資料處理流程")
    print("="*60)

    # 初始化元件
    loader = DataLoader(data_dir='ml/data')
    visualizer = DataVisualizer(output_dir='ml/output/visualizations')

    # ========== 步驟1: 載入原始資料 ==========
    print("\n[步驟 1/7] 載入原始資料...")
    df = loader.load_raw_data()

    # ========== 步驟2: 新增基礎特徵 ==========
    print("\n[步驟 2/7] 計算基礎特徵（坪數、每坪租金）...")
    df = loader.add_basic_features(df)

    # ========== 步驟3: 視覺化原始資料 ==========
    print("\n[步驟 3/7] 視覺化原始資料分佈...")
    visualizer.plot_rent_distribution(
        df,
        filename='01_raw_rent_distribution.png'
    )

    # ========== 步驟4: 資料清洗 ==========
    print("\n[步驟 4/7] 資料清洗（移除非住宅、異常值）...")
    df_clean = DataCleaner.clean_pipeline(
        df,
        remove_outliers=True,
        save_path='ml/data/taipei_newtaipei_cleaned.csv'
    )

    # ========== 步驟5: 特徵工程 ==========
    print("\n[步驟 5/7] 特徵工程（屋齡、樓層）...")

    # 計算屋齡
    df_clean = FeatureEngineer.calculate_house_age(df_clean)

    # 提取樓層特徵
    df_clean = FeatureEngineer.extract_floor_feature(df_clean)

    # 儲存特徵工程後的資料
    df_clean.to_csv(
        'ml/data/taipei_newtaipei_featured.csv',
        index=False,
        encoding='utf-8-sig'
    )
    print("\n特徵工程資料已儲存至: ml/data/taipei_newtaipei_featured.csv")

    # ========== 步驟6: 特徵編碼 ==========
    print("\n[步驟 6/7] 特徵編碼（One-Hot Encoding）...")
    df_encoded = FeatureEngineer.encode_features(df_clean)

    # ========== 步驟7: 產生分析圖表 ==========
    print("\n[步驟 7/7] 產生分析圖表...")

    # 清洗後的租金分佈
    visualizer.plot_rent_distribution(
        df_clean,
        filename='02_cleaned_rent_distribution.png'
    )

    # 相關性矩陣
    visualizer.plot_correlation_matrix(
        df_encoded,
        target_column='總額元',
        top_k=15,
        filename='03_correlation_matrix.png'
    )

    # 屋齡 vs 租金散點圖
    visualizer.plot_scatter(
        df_clean,
        x_col='屋齡',
        y_col='每坪租金',
        hue_col='城市',
        filename='04_age_vs_rent.png'
    )

    # 樓層 vs 租金散點圖
    visualizer.plot_scatter(
        df_clean,
        x_col='樓層',
        y_col='每坪租金',
        hue_col='城市',
        filename='05_floor_vs_rent.png'
    )

    # ========== 完成 ==========
    print("\n" + "="*60)
    print("✓ 資料處理流程完成！")
    print("="*60)
    print(f"\n最終資料統計:")
    print(f"  - 總筆數: {len(df_clean)}")
    print(f"  - 特徵數: {len(df_clean.columns)}")
    print(f"  - 編碼後特徵數: {len(df_encoded.columns)}")
    print(f"\n輸出檔案:")
    print(f"  - ml/data/taipei_newtaipei_cleaned.csv")
    print(f"  - ml/data/taipei_newtaipei_featured.csv")
    print(f"  - ml/output/visualizations/ (5張圖表)")
    print()


if __name__ == '__main__':
    main()
