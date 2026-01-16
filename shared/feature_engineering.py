"""
特徵工程模組
負責建立和轉換特徵
"""
import pandas as pd


class FeatureEngineer:
    """特徵工程處理器"""

    @staticmethod
    def calculate_house_age(df):
        """
        計算房屋年齡

        Args:
            df: 包含交易日期和建築完成日期的DataFrame

        Returns:
            新增屋齡特徵的DataFrame
        """
        df = df.copy()

        # 提取交易年份和建築年份
        df['交易年'] = df['租賃年月日'] // 10000
        df['建築年'] = df['建築完成年月'] // 10000

        # 計算屋齡
        df['屋齡'] = df['交易年'] - df['建築年']

        # 處理負數屋齡（預售屋或資料錯誤）
        df.loc[df['屋齡'] < 0, '屋齡'] = 0

        print(f"\n=== 屋齡統計 ===")
        print(df['屋齡'].describe())

        return df

    @staticmethod
    def convert_floor(floor_str):
        """
        將中文樓層描述轉換為數字

        Args:
            floor_str: 樓層字串（如："五層"）

        Returns:
            樓層數字
        """
        if pd.isna(floor_str):
            return None

        # 中文數字對照表
        chinese_nums = {
            '一': 1, '二': 2, '三': 3, '四': 4, '五': 5,
            '六': 6, '七': 7, '八': 8, '九': 9, '十': 10
        }

        # 特殊情況處理
        if '全' in floor_str:
            return 1
        if '見其他' in floor_str:
            return None
        if '地下' in floor_str:
            return -1

        try:
            # 移除"層"字
            clean_str = floor_str.replace('層', '')

            # 單一數字（一層～九層）
            if clean_str in chinese_nums:
                return chinese_nums[clean_str]

            # 十幾（十一～十九）
            if clean_str.startswith('十') and len(clean_str) > 1:
                return 10 + chinese_nums.get(clean_str[1], 0)

            # 其他複雜情況返回None
            return None

        except Exception:
            return None

    @staticmethod
    def extract_floor_feature(df):
        """
        提取樓層特徵

        Args:
            df: 包含租賃層次欄位的DataFrame

        Returns:
            新增樓層數字特徵的DataFrame
        """
        df = df.copy()

        # 應用樓層轉換
        df['樓層'] = df['租賃層次'].apply(FeatureEngineer.convert_floor)

        # 統計缺失值
        missing_count = df['樓層'].isna().sum()
        print(f"\n樓層轉換後缺失值: {missing_count} 筆 "
              f"({missing_count/len(df)*100:.1f}%)")

        # 移除無法解析樓層的資料
        df = df.dropna(subset=['樓層'])
        print(f"移除缺失值後剩餘: {len(df)} 筆")

        return df

    @staticmethod
    def simplify_building_type(type_str):
        """
        簡化建築類型

        Args:
            type_str: 建築類型字串

        Returns:
            簡化後的類型
        """
        if pd.isna(type_str):
            return '其他'
        if '公寓' in type_str:
            return '公寓'
        if '華廈' in type_str:
            return '華廈'
        if '住宅大樓' in type_str:
            return '大樓'
        if '透天' in type_str:
            return '透天'
        return '其他'

    @staticmethod
    def encode_features(df):
        """
        對分類特徵進行編碼

        Args:
            df: 待編碼的DataFrame

        Returns:
            編碼後的DataFrame
        """
        df = df.copy()

        # 二元變數編碼
        binary_map = {'有': 1, '無': 0}
        binary_cols = ['有無電梯', '有無管理組織', '有無附傢俱']

        for col in binary_cols:
            if col in df.columns:
                df[col] = df[col].map(binary_map).fillna(0)

        # 簡化建築類型
        if '建物型態' in df.columns:
            df['建物型態_簡化'] = df['建物型態'].apply(
                FeatureEngineer.simplify_building_type
            )

            # One-Hot編碼
            df = pd.get_dummies(
                df,
                columns=['城市', '鄉鎮市區', '建物型態_簡化'],
                drop_first=True
            )

        print("\n=== 特徵編碼完成 ===")
        print(f"編碼後特徵數: {len(df.columns)}")

        return df
