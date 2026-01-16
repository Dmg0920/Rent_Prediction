"""
資料視覺化模組
負責產生各種資料分析圖表
"""
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


class DataVisualizer:
    """資料視覺化器"""

    def __init__(self, output_dir='output/visualizations'):
        """
        初始化視覺化器

        Args:
            output_dir: 圖表輸出目錄
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 設定中文字型
        plt.rcParams['font.sans-serif'] = [
            'Arial Unicode MS',
            'PingFang TC',
            'Heiti TC'
        ]
        plt.rcParams['axes.unicode_minus'] = False

    def plot_rent_distribution(self, df, column='每坪租金',
                               city_column='城市',
                               filename='rent_distribution.png'):
        """
        繪製租金分佈圖（直方圖+箱型圖）

        Args:
            df: DataFrame
            column: 租金欄位名稱
            city_column: 城市欄位名稱
            filename: 儲存檔案名稱
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # 直方圖
        sns.histplot(
            data=df,
            x=column,
            hue=city_column,
            kde=True,
            element="step",
            ax=axes[0]
        )
        axes[0].set_title('租金分佈 (直方圖)')
        axes[0].set_xlabel('每坪租金 (元)')
        axes[0].set_ylabel('數量')

        # 箱型圖
        sns.boxplot(
            data=df,
            x=city_column,
            y=column,
            ax=axes[1]
        )
        axes[1].set_title('租金比較 (箱型圖)')
        axes[1].set_xlabel('城市')
        axes[1].set_ylabel('每坪租金 (元)')

        plt.tight_layout()

        # 儲存圖表
        save_path = self.output_dir / filename
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"圖表已儲存: {save_path}")

        plt.close()

    def plot_correlation_matrix(self, df, target_column='總額元',
                                top_k=15,
                                filename='correlation_matrix.png'):
        """
        繪製相關性矩陣熱圖

        Args:
            df: DataFrame（需要是編碼後的數值資料）
            target_column: 目標欄位
            top_k: 顯示前K個最相關的特徵
            filename: 儲存檔案名稱
        """
        # 只選擇數值欄位
        numeric_df = df.select_dtypes(include=['number'])

        if target_column not in numeric_df.columns:
            print(f"警告: 目標欄位 '{target_column}' 不在數值欄位中")
            return

        # 計算相關性矩陣
        corr_matrix = numeric_df.corr()

        # 取得與目標欄位最相關的前K個特徵
        top_features = corr_matrix.nlargest(
            top_k,
            target_column
        )[target_column].index

        # 繪製熱圖
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            numeric_df[top_features].corr(),
            annot=True,
            square=True,
            fmt='.2f',
            cmap='coolwarm',
            center=0
        )
        plt.title(f'Top {top_k} 特徵與 {target_column} 的相關性')
        plt.tight_layout()

        # 儲存圖表
        save_path = self.output_dir / filename
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"相關性矩陣已儲存: {save_path}")

        # 列印統計資訊
        print(f"\n與「{target_column}」正相關最強的前5名：")
        print(corr_matrix[target_column].sort_values(ascending=False).head(6))

        print(f"\n與「{target_column}」負相關最強的前5名：")
        print(corr_matrix[target_column].sort_values(ascending=True).head(5))

        plt.close()

    def plot_feature_importance(self, feature_names, importance_values,
                                top_k=20,
                                filename='feature_importance.png'):
        """
        繪製特徵重要性圖

        Args:
            feature_names: 特徵名稱列表
            importance_values: 重要性數值列表
            top_k: 顯示前K個重要特徵
            filename: 儲存檔案名稱
        """
        import pandas as pd

        # 建立DataFrame並排序
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance_values
        }).sort_values('importance', ascending=False).head(top_k)

        # 繪製條形圖
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(importance_df)), importance_df['importance'])
        plt.yticks(range(len(importance_df)), importance_df['feature'])
        plt.xlabel('重要性')
        plt.title(f'Top {top_k} 特徵重要性')
        plt.gca().invert_yaxis()
        plt.tight_layout()

        # 儲存圖表
        save_path = self.output_dir / filename
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"特徵重要性圖已儲存: {save_path}")

        plt.close()

    def plot_scatter(self, df, x_col, y_col, hue_col=None,
                    filename='scatter_plot.png'):
        """
        繪製散點圖

        Args:
            df: DataFrame
            x_col: X軸欄位名稱
            y_col: Y軸欄位名稱
            hue_col: 顏色分組欄位名稱（可選）
            filename: 儲存檔案名稱
        """
        plt.figure(figsize=(10, 6))

        if hue_col:
            sns.scatterplot(data=df, x=x_col, y=y_col, hue=hue_col, alpha=0.6)
        else:
            sns.scatterplot(data=df, x=x_col, y=y_col, alpha=0.6)

        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.title(f'{y_col} vs {x_col}')
        plt.tight_layout()

        # 儲存圖表
        save_path = self.output_dir / filename
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"散點圖已儲存: {save_path}")

        plt.close()
