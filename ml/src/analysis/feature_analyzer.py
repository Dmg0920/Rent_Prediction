"""
特徵分析模組
提供統計驅動的特徵分析功能，包括：
- 相關性分析（Pearson, Spearman）
- 多重共線性診斷（VIF）
- 互信息分析
- 卡方檢驗
"""
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import chi2_contingency
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt
import seaborn as sns


class FeatureAnalyzer:
    """特徵統計分析器"""

    def __init__(self, df, target_col='總額元'):
        """
        初始化特徵分析器

        Args:
            df: DataFrame，包含特徵和目標變數
            target_col: 目標變數欄位名稱
        """
        self.df = df.copy()
        self.target_col = target_col

    def correlation_analysis(self, method='pearson', top_n=20):
        """
        相關性分析

        Args:
            method: 'pearson'（線性相關）或 'spearman'（秩相關）
            top_n: 顯示前N個最相關的特徵

        Returns:
            DataFrame: 與目標變數的相關係數，按絕對值排序
        """
        print(f"\n{'='*60}")
        print(f"相關性分析 ({method.capitalize()} Correlation)")
        print(f"{'='*60}")

        # 只選擇數值型欄位
        numeric_df = self.df.select_dtypes(include=[np.number])

        if self.target_col not in numeric_df.columns:
            print(f"警告：目標變數 {self.target_col} 不在數值欄位中")
            return None

        # 計算相關係數
        if method == 'pearson':
            correlations = numeric_df.corr()[self.target_col]
        elif method == 'spearman':
            correlations = numeric_df.corr(method='spearman')[self.target_col]
        else:
            raise ValueError("method 必須是 'pearson' 或 'spearman'")

        # 移除目標變數本身
        correlations = correlations.drop(self.target_col)

        # 按絕對值排序
        correlations_sorted = correlations.reindex(
            correlations.abs().sort_values(ascending=False).index
        )

        # 顯示前N個
        print(f"\n與 {self.target_col} 相關性最高的前 {top_n} 個特徵：\n")
        for i, (feature, corr) in enumerate(correlations_sorted.head(top_n).items(), 1):
            significance = self._get_significance_stars(abs(corr))
            print(f"{i:2d}. {feature:30s}: {corr:7.4f} {significance}")

        print(f"\n說明：")
        print(f"  *** |r| > 0.5  (強相關)")
        print(f"  **  |r| > 0.3  (中等相關)")
        print(f"  *   |r| > 0.1  (弱相關)")

        return correlations_sorted

    def _get_significance_stars(self, abs_corr):
        """根據相關係數強度返回星號標記"""
        if abs_corr > 0.5:
            return "***"
        elif abs_corr > 0.3:
            return "**"
        elif abs_corr > 0.1:
            return "*"
        return ""

    def vif_analysis(self, features=None, threshold=10):
        """
        多重共線性診斷 - 變異數膨脹因子 (Variance Inflation Factor)

        VIF > 10 表示嚴重共線性
        VIF > 5  表示中等共線性

        Args:
            features: 要檢查的特徵列表，None表示使用所有數值特徵
            threshold: VIF閾值，超過此值將標記為高共線性

        Returns:
            DataFrame: 包含特徵名稱和VIF值
        """
        print(f"\n{'='*60}")
        print("多重共線性診斷 (VIF Analysis)")
        print(f"{'='*60}")

        # 選擇數值型特徵
        numeric_df = self.df.select_dtypes(include=[np.number])

        # 移除目標變數和其他不需要的欄位
        exclude_cols = [self.target_col, '每坪租金', '單價元平方公尺', '車位總額元',
                       '租賃年月日', '建築完成年月', '編號', '租賃筆棟數']
        cols_to_use = [c for c in numeric_df.columns if c not in exclude_cols]

        if features is not None:
            cols_to_use = [c for c in features if c in cols_to_use]

        X = numeric_df[cols_to_use].fillna(numeric_df[cols_to_use].mean())

        # 移除常數特徵（變異數為0）
        X = X.loc[:, X.var() > 0]

        # 計算VIF
        vif_data = pd.DataFrame()
        vif_data["特徵"] = X.columns
        vif_data["VIF"] = [variance_inflation_factor(X.values, i)
                          for i in range(X.shape[1])]

        # 排序
        vif_data = vif_data.sort_values('VIF', ascending=False)

        # 顯示結果
        print(f"\nVIF 分析結果：\n")
        print(f"{'特徵':<30s} {'VIF':>10s} {'診斷':<15s}")
        print("-" * 60)

        for _, row in vif_data.iterrows():
            feature = row['特徵']
            vif = row['VIF']

            if vif > threshold:
                diagnosis = "⚠️ 嚴重共線性"
            elif vif > 5:
                diagnosis = "⚡ 中等共線性"
            else:
                diagnosis = "✓ 正常"

            print(f"{feature:<30s} {vif:10.2f} {diagnosis:<15s}")

        print(f"\n說明：")
        print(f"  VIF > {threshold}  → 嚴重多重共線性，考慮移除")
        print(f"  VIF > 5   → 中等共線性，需注意")
        print(f"  VIF < 5   → 共線性可接受")

        return vif_data

    def mutual_information_analysis(self, top_n=20):
        """
        互信息分析 - 檢測非線性關係

        互信息可以捕捉變數間的非線性相依性

        Args:
            top_n: 顯示前N個最重要的特徵

        Returns:
            Series: 特徵的互信息分數
        """
        print(f"\n{'='*60}")
        print("互信息分析 (Mutual Information)")
        print(f"{'='*60}")

        # 準備資料
        numeric_df = self.df.select_dtypes(include=[np.number])

        exclude_cols = [self.target_col, '每坪租金', '單價元平方公尺', '車位總額元',
                       '租賃年月日', '建築完成年月', '編號', '租賃筆棟數']
        feature_cols = [c for c in numeric_df.columns if c not in exclude_cols]

        X = numeric_df[feature_cols].fillna(numeric_df[feature_cols].mean())
        y = numeric_df[self.target_col]

        # 移除常數特徵
        X = X.loc[:, X.var() > 0]

        # 計算互信息
        mi_scores = mutual_info_regression(X, y, random_state=42)
        mi_scores = pd.Series(mi_scores, index=X.columns)
        mi_scores = mi_scores.sort_values(ascending=False)

        # 顯示結果
        print(f"\n互信息分數最高的前 {top_n} 個特徵：\n")
        print(f"{'排名':<5s} {'特徵':<30s} {'MI分數':>10s} {'標準化分數':>12s}")
        print("-" * 60)

        # 標準化到0-1
        mi_normalized = mi_scores / mi_scores.max() if mi_scores.max() > 0 else mi_scores

        for i, (feature, score) in enumerate(mi_scores.head(top_n).items(), 1):
            norm_score = mi_normalized[feature]
            bars = '█' * int(norm_score * 20)
            print(f"{i:<5d} {feature:<30s} {score:10.4f} {bars}")

        print(f"\n說明：")
        print(f"  互信息可以檢測非線性關係")
        print(f"  分數越高表示該特徵對目標變數的資訊量越大")

        return mi_scores

    def chi_square_test(self, categorical_features=None, alpha=0.05):
        """
        卡方獨立性檢驗 - 類別變數與目標的關聯性

        Args:
            categorical_features: 要檢驗的類別變數列表
            alpha: 顯著水準

        Returns:
            DataFrame: 卡方檢驗結果
        """
        print(f"\n{'='*60}")
        print(f"卡方獨立性檢驗 (Chi-Square Test, α={alpha})")
        print(f"{'='*60}")

        # 如果沒有指定，自動選擇object類型的欄位
        if categorical_features is None:
            categorical_features = self.df.select_dtypes(include=['object']).columns.tolist()

        if len(categorical_features) == 0:
            print("沒有找到類別變數")
            return None

        # 將目標變數分箱（如果是連續變數）
        if self.df[self.target_col].dtype in [np.float64, np.int64]:
            target_binned = pd.qcut(self.df[self.target_col], q=4,
                                   labels=['低', '中低', '中高', '高'],
                                   duplicates='drop')
        else:
            target_binned = self.df[self.target_col]

        results = []

        for feature in categorical_features:
            # 建立列聯表
            contingency_table = pd.crosstab(self.df[feature], target_binned)

            # 卡方檢驗
            chi2, p_value, dof, expected = chi2_contingency(contingency_table)

            # 判斷顯著性
            is_significant = p_value < alpha

            results.append({
                '特徵': feature,
                '卡方統計量': chi2,
                'p值': p_value,
                '自由度': dof,
                '顯著性': '✓ 顯著' if is_significant else '✗ 不顯著'
            })

        results_df = pd.DataFrame(results).sort_values('p值')

        # 顯示結果
        print(f"\n檢驗結果：\n")
        for _, row in results_df.iterrows():
            print(f"特徵: {row['特徵']}")
            print(f"  卡方統計量 = {row['卡方統計量']:.4f}")
            print(f"  p值 = {row['p值']:.4e}")
            print(f"  {row['顯著性']} (α={alpha})")
            print()

        return results_df

    def plot_correlation_heatmap(self, top_n=20, figsize=(12, 10)):
        """
        繪製相關性熱力圖

        Args:
            top_n: 顯示前N個與目標最相關的特徵
            figsize: 圖形大小
        """
        # 計算相關性
        correlations = self.correlation_analysis(method='pearson', top_n=top_n)

        if correlations is None:
            return

        # 選擇前N個特徵
        top_features = correlations.head(top_n).index.tolist()
        top_features.append(self.target_col)

        # 計算相關性矩陣
        numeric_df = self.df.select_dtypes(include=[np.number])
        corr_matrix = numeric_df[top_features].corr()

        # 繪製熱力圖
        plt.figure(figsize=figsize)
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                   center=0, square=True, linewidths=0.5,
                   cbar_kws={'label': 'Correlation Coefficient'})
        plt.title(f'前{top_n}個重要特徵的相關性矩陣', fontsize=14, pad=20)
        plt.tight_layout()

        return plt.gcf()
