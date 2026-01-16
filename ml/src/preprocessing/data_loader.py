"""
資料載入模組
負責從CSV檔案載入原始資料並進行初步處理
"""
import pandas as pd
from pathlib import Path


class DataLoader:
    """資料載入器"""

    def __init__(self, data_dir='data'):
        """
        初始化資料載入器

        Args:
            data_dir: 資料目錄路徑
        """
        self.data_dir = Path(data_dir)

    def load_raw_data(self, taipei_file='a_lvr_land_c.csv',
                      newtaipei_file='f_lvr_land_c.csv'):
        """
        載入台北市和新北市的原始資料

        Args:
            taipei_file: 台北市資料檔案名
            newtaipei_file: 新北市資料檔案名

        Returns:
            合併後的DataFrame
        """
        # 載入台北市資料
        df_taipei = pd.read_csv(
            self.data_dir / taipei_file,
            header=0,
            skiprows=[1]
        )
        df_taipei['城市'] = '台北市'
        print(f"台北市: {len(df_taipei)} 筆資料")

        # 載入新北市資料
        df_newtaipei = pd.read_csv(
            self.data_dir / newtaipei_file,
            header=0,
            skiprows=[1]
        )
        df_newtaipei['城市'] = '新北市'
        print(f"新北市: {len(df_newtaipei)} 筆資料")

        # 合併資料
        df_combined = pd.concat([df_taipei, df_newtaipei],
                               axis=0,
                               ignore_index=True)
        print(f"雙北合併: {len(df_combined)} 筆資料")

        return df_combined

    def add_basic_features(self, df):
        """
        新增基礎特徵：坪數和每坪租金

        Args:
            df: 原始DataFrame

        Returns:
            新增特徵後的DataFrame
        """
        df = df.copy()

        # 計算坪數
        df['坪數'] = df['建物總面積平方公尺'] * 0.3025

        # 計算每坪租金
        df['每坪租金'] = df['總額元'] / df['坪數']

        # 移除坪數為0的異常資料
        df = df[df['坪數'] > 0].copy()

        print("\n=== 每坪租金統計 ===")
        print(df.groupby('城市')['每坪租金'].describe())

        return df

    def load_cleaned_data(self, filename='taipei_newtaipei_cleaned.csv'):
        """
        載入已清洗的資料

        Args:
            filename: 清洗後的資料檔案名

        Returns:
            清洗後的DataFrame
        """
        filepath = self.data_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"找不到清洗後的資料檔案: {filepath}")

        df = pd.read_csv(filepath)
        print(f"載入已清洗資料: {len(df)} 筆")
        return df

    def load_featured_data(self, filename='taipei_newtaipei_featured.csv'):
        """
        載入特徵工程後的資料

        Args:
            filename: 特徵工程後的資料檔案名

        Returns:
            特徵工程後的DataFrame
        """
        filepath = self.data_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"找不到特徵工程資料檔案: {filepath}")

        df = pd.read_csv(filepath)
        print(f"載入特徵工程資料: {len(df)} 筆")
        return df
