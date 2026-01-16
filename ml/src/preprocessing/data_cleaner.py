"""
資料清洗模組
負責異常值檢測、資料過濾等清洗工作
"""
import pandas as pd


class DataCleaner:
    """資料清洗器"""

    # 定義非住宅類型
    NON_RESIDENTIAL_TYPES = ['店面(店鋪)', '辦公商業大樓', '廠辦', '工廠', '倉庫']

    @staticmethod
    def remove_non_residential(df):
        """
        移除非住宅類型的資料

        Args:
            df: 原始DataFrame

        Returns:
            過濾後的DataFrame
        """
        df = df.copy()

        # 過濾非住宅
        df = df[~df['建物型態'].isin(DataCleaner.NON_RESIDENTIAL_TYPES)].copy()

        print(f"\n=== 移除非住宅類型 ===")
        print(f"剩餘筆數: {len(df)}")

        return df

    @staticmethod
    def detect_outliers_iqr(df, column='每坪租金', city_column='城市',
                           multiplier=1.5):
        """
        使用IQR方法檢測異常值

        Args:
            df: DataFrame
            column: 要檢測異常值的欄位
            city_column: 城市欄位（分城市計算）
            multiplier: IQR倍數（預設1.5）

        Returns:
            異常值的DataFrame
        """
        outliers_list = []

        for city in df[city_column].unique():
            city_data = df[df[city_column] == city]

            # 計算IQR
            Q1 = city_data[column].quantile(0.25)
            Q3 = city_data[column].quantile(0.75)
            IQR = Q3 - Q1

            # 定義邊界
            upper_bound = Q3 + multiplier * IQR
            lower_bound = Q1 - multiplier * IQR

            # 找出異常值
            city_outliers = city_data[
                (city_data[column] > upper_bound) |
                (city_data[column] < lower_bound)
            ]

            outliers_list.append(city_outliers)

            print(f"\n{city}:")
            print(f"  Q1={Q1:.1f}, Q3={Q3:.1f}, IQR={IQR:.1f}")
            print(f"  異常值範圍: < {lower_bound:.1f} 或 > {upper_bound:.1f}")
            print(f"  異常筆數: {len(city_outliers)} "
                  f"({len(city_outliers)/len(city_data)*100:.1f}%)")

        return pd.concat(outliers_list) if outliers_list else pd.DataFrame()

    @staticmethod
    def remove_outliers_iqr(df, column='每坪租金', city_column='城市',
                           multiplier=1.5):
        """
        使用IQR方法移除異常值

        Args:
            df: DataFrame
            column: 要檢測異常值的欄位
            city_column: 城市欄位（分城市計算）
            multiplier: IQR倍數（預設1.5）

        Returns:
            移除異常值後的DataFrame
        """
        df = df.copy()
        clean_data_list = []

        print(f"\n=== 使用IQR方法移除異常值 (倍數={multiplier}) ===")

        for city in df[city_column].unique():
            city_data = df[df[city_column] == city]

            # 計算IQR
            Q1 = city_data[column].quantile(0.25)
            Q3 = city_data[column].quantile(0.75)
            IQR = Q3 - Q1

            # 定義邊界
            upper_bound = Q3 + multiplier * IQR
            lower_bound = Q1 - multiplier * IQR

            # 保留正常值
            clean_city_data = city_data[
                (city_data[column] >= lower_bound) &
                (city_data[column] <= upper_bound)
            ]

            clean_data_list.append(clean_city_data)

            print(f"\n{city}:")
            print(f"  原始筆數: {len(city_data)}")
            print(f"  移除筆數: {len(city_data) - len(clean_city_data)}")
            print(f"  剩餘筆數: {len(clean_city_data)}")

        result = pd.concat(clean_data_list, ignore_index=True)
        print(f"\n總計剩餘: {len(result)} 筆")

        return result

    @staticmethod
    def clean_pipeline(df, remove_outliers=True, save_path=None):
        """
        完整的資料清洗流程

        Args:
            df: 原始DataFrame
            remove_outliers: 是否移除異常值
            save_path: 儲存路徑（可選）

        Returns:
            清洗後的DataFrame
        """
        print("\n" + "="*50)
        print("開始資料清洗流程")
        print("="*50)

        # 1. 移除非住宅
        df = DataCleaner.remove_non_residential(df)

        # 2. 移除異常值（可選）
        if remove_outliers:
            df = DataCleaner.remove_outliers_iqr(df)

        # 3. 儲存（可選）
        if save_path:
            df.to_csv(save_path, index=False, encoding='utf-8-sig')
            print(f"\n清洗後資料已儲存至: {save_path}")

        print("\n" + "="*50)
        print(f"資料清洗完成！最終筆數: {len(df)}")
        print("="*50)

        return df
