"""
模型評估與統計檢驗模組
提供嚴謹的統計檢驗來比較模型效能
"""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import cross_val_score, KFold
import matplotlib.pyplot as plt
import seaborn as sns


class ModelEvaluator:
    """模型統計評估器"""

    def __init__(self, random_state=42):
        """
        初始化評估器

        Args:
            random_state: 隨機種子
        """
        self.random_state = random_state

    def paired_t_test(self, predictions1, predictions2, y_true, alpha=0.05):
        """
        配對 t 檢驗 - 比較兩個模型的預測誤差

        H0: 兩模型的平均誤差相同
        H1: 兩模型的平均誤差不同

        Args:
            predictions1: 模型1的預測值
            predictions2: 模型2的預測值
            y_true: 真實值
            alpha: 顯著水準

        Returns:
            dict: 檢驗結果
        """
        # 計算誤差
        errors1 = y_true - predictions1
        errors2 = y_true - predictions2

        # 配對差異
        diff = errors1 - errors2

        # t檢驗
        t_stat, p_value = stats.ttest_rel(errors1, errors2)

        # 判斷
        is_significant = p_value < alpha

        print(f"\n{'='*60}")
        print("配對 t 檢驗 (Paired t-test)")
        print(f"{'='*60}")
        print(f"\n假設檢驗：")
        print(f"  H₀: 兩模型平均誤差相同")
        print(f"  H₁: 兩模型平均誤差不同")
        print(f"  顯著水準 α = {alpha}")
        print(f"\n檢驗結果：")
        print(f"  t 統計量 = {t_stat:.4f}")
        print(f"  p 值 = {p_value:.4e}")
        print(f"  平均誤差差異 = {diff.mean():.2f}")
        print(f"  誤差差異標準差 = {diff.std():.2f}")

        if is_significant:
            print(f"\n結論: ✓ 拒絕 H₀ (p < {alpha})")
            print(f"  兩模型的預測誤差有顯著差異")
        else:
            print(f"\n結論: ✗ 無法拒絕 H₀ (p ≥ {alpha})")
            print(f"  兩模型的預測誤差沒有顯著差異")

        return {
            't_statistic': t_stat,
            'p_value': p_value,
            'mean_diff': diff.mean(),
            'std_diff': diff.std(),
            'is_significant': is_significant
        }

    def mcnemar_test(self, predictions1, predictions2, y_true, threshold=0.1):
        """
        McNemar 檢驗 - 比較兩個模型的分類錯誤模式
        這裡改為比較相對誤差是否超過閾值

        Args:
            predictions1: 模型1的預測值
            predictions2: 模型2的預測值
            y_true: 真實值
            threshold: 相對誤差閾值（例如0.1表示10%）

        Returns:
            dict: 檢驗結果
        """
        # 計算相對誤差
        rel_error1 = np.abs(y_true - predictions1) / y_true
        rel_error2 = np.abs(y_true - predictions2) / y_true

        # 分類：誤差是否超過閾值
        fail1 = rel_error1 > threshold
        fail2 = rel_error2 > threshold

        # 建立列聯表
        # n01: 模型1正確，模型2錯誤
        # n10: 模型1錯誤，模型2正確
        n01 = np.sum(~fail1 & fail2)
        n10 = np.sum(fail1 & ~fail2)

        # McNemar統計量
        if n01 + n10 == 0:
            print("兩模型完全一致，無法進行McNemar檢驗")
            return None

        chi2 = (abs(n01 - n10) - 1)**2 / (n01 + n10)
        p_value = 1 - stats.chi2.cdf(chi2, df=1)

        print(f"\n{'='*60}")
        print("McNemar 檢驗")
        print(f"{'='*60}")
        print(f"\n誤差閾值: {threshold*100}%")
        print(f"\n列聯表：")
        print(f"  模型1正確, 模型2錯誤: {n01}")
        print(f"  模型1錯誤, 模型2正確: {n10}")
        print(f"\n檢驗結果：")
        print(f"  χ² 統計量 = {chi2:.4f}")
        print(f"  p 值 = {p_value:.4e}")

        return {
            'chi2': chi2,
            'p_value': p_value,
            'n01': n01,
            'n10': n10
        }

    def diebold_mariano_test(self, errors1, errors2, h=1, alpha=0.05):
        """
        Diebold-Mariano 檢驗 - 預測準確度比較
        適用於時間序列預測，這裡用於一般迴歸

        Args:
            errors1: 模型1的預測誤差
            errors2: 模型2的預測誤差
            h: 預測步數（horizon）
            alpha: 顯著水準

        Returns:
            dict: 檢驗結果
        """
        # 計算損失函數（平方誤差）
        d = errors1**2 - errors2**2

        # DM統計量
        d_mean = np.mean(d)
        d_var = np.var(d, ddof=1)
        n = len(d)

        dm_stat = d_mean / np.sqrt(d_var / n)

        # p值（雙尾檢驗）
        p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))

        is_significant = p_value < alpha

        print(f"\n{'='*60}")
        print("Diebold-Mariano 檢驗")
        print(f"{'='*60}")
        print(f"\n檢驗結果：")
        print(f"  DM 統計量 = {dm_stat:.4f}")
        print(f"  p 值 = {p_value:.4e}")

        if is_significant:
            if dm_stat > 0:
                print(f"\n結論: 模型2顯著優於模型1 (p < {alpha})")
            else:
                print(f"\n結論: 模型1顯著優於模型2 (p < {alpha})")
        else:
            print(f"\n結論: 兩模型預測準確度沒有顯著差異 (p ≥ {alpha})")

        return {
            'dm_statistic': dm_stat,
            'p_value': p_value,
            'is_significant': is_significant
        }

    def friedman_test(self, predictions_dict, y_true, alpha=0.05):
        """
        Friedman 檢驗 - 比較多個模型（非參數方法）

        H0: 所有模型的效能相同
        H1: 至少有一個模型的效能不同

        Args:
            predictions_dict: {模型名稱: 預測值} 字典
            y_true: 真實值
            alpha: 顯著水準

        Returns:
            dict: 檢驗結果
        """
        # 計算每個模型的絕對誤差
        errors = {}
        for name, pred in predictions_dict.items():
            errors[name] = np.abs(y_true - pred)

        # 轉換為矩陣（每行是一個樣本，每列是一個模型）
        error_matrix = np.column_stack([errors[name] for name in errors.keys()])

        # Friedman檢驗
        statistic, p_value = stats.friedmanchisquare(*error_matrix.T)

        is_significant = p_value < alpha

        print(f"\n{'='*60}")
        print(f"Friedman 檢驗 (比較 {len(predictions_dict)} 個模型)")
        print(f"{'='*60}")
        print(f"\n假設檢驗：")
        print(f"  H₀: 所有模型效能相同")
        print(f"  H₁: 至少有一個模型效能不同")
        print(f"\n檢驗結果：")
        print(f"  χ² 統計量 = {statistic:.4f}")
        print(f"  p 值 = {p_value:.4e}")

        if is_significant:
            print(f"\n結論: ✓ 拒絕 H₀ (p < {alpha})")
            print(f"  模型之間存在顯著差異")
            print(f"  建議進行事後檢定（post-hoc test）找出差異來源")
        else:
            print(f"\n結論: ✗ 無法拒絕 H₀ (p ≥ {alpha})")
            print(f"  模型之間沒有顯著差異")

        # 計算平均秩次
        ranks = stats.rankdata(error_matrix, axis=1)
        mean_ranks = ranks.mean(axis=0)

        print(f"\n平均秩次（越小越好）：")
        for i, name in enumerate(errors.keys()):
            print(f"  {name}: {mean_ranks[i]:.2f}")

        return {
            'statistic': statistic,
            'p_value': p_value,
            'is_significant': is_significant,
            'mean_ranks': dict(zip(errors.keys(), mean_ranks))
        }

    def bootstrap_confidence_interval(self, y_true, predictions, metric_func,
                                     n_bootstrap=1000, confidence=0.95):
        """
        Bootstrap 信賴區間估計

        Args:
            y_true: 真實值
            predictions: 預測值
            metric_func: 評估指標函數，例如 lambda y, p: np.sqrt(mean_squared_error(y, p))
            n_bootstrap: Bootstrap 重抽樣次數
            confidence: 信賴水準

        Returns:
            dict: 包含點估計和信賴區間
        """
        n = len(y_true)
        metrics = []

        # Bootstrap重抽樣
        for _ in range(n_bootstrap):
            indices = np.random.choice(n, n, replace=True)
            y_boot = y_true[indices]
            pred_boot = predictions[indices]
            metric = metric_func(y_boot, pred_boot)
            metrics.append(metric)

        metrics = np.array(metrics)

        # 計算信賴區間
        alpha = 1 - confidence
        lower = np.percentile(metrics, alpha/2 * 100)
        upper = np.percentile(metrics, (1 - alpha/2) * 100)
        point_estimate = metric_func(y_true, predictions)

        print(f"\n{'='*60}")
        print(f"Bootstrap 信賴區間 ({confidence*100:.0f}% 信賴水準)")
        print(f"{'='*60}")
        print(f"\nBootstrap 參數：")
        print(f"  重抽樣次數: {n_bootstrap}")
        print(f"  樣本數: {n}")
        print(f"\n結果：")
        print(f"  點估計: {point_estimate:.4f}")
        print(f"  {confidence*100:.0f}% 信賴區間: [{lower:.4f}, {upper:.4f}]")
        print(f"  區間寬度: {upper - lower:.4f}")

        return {
            'point_estimate': point_estimate,
            'lower': lower,
            'upper': upper,
            'bootstrap_samples': metrics
        }

    def learning_curve_analysis(self, model, X, y, train_sizes=None, cv=5):
        """
        學習曲線分析 - 診斷偏差-變異數權衡

        Args:
            model: 模型對象
            X: 特徵
            y: 目標
            train_sizes: 訓練集大小比例
            cv: 交叉驗證折數

        Returns:
            dict: 學習曲線數據
        """
        from sklearn.model_selection import learning_curve

        if train_sizes is None:
            train_sizes = np.linspace(0.1, 1.0, 10)

        print(f"\n{'='*60}")
        print("學習曲線分析")
        print(f"{'='*60}")

        train_sizes_abs, train_scores, val_scores = learning_curve(
            model, X, y,
            train_sizes=train_sizes,
            cv=cv,
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )

        # 轉換為RMSE
        train_scores_mean = np.sqrt(-train_scores.mean(axis=1))
        train_scores_std = np.sqrt(train_scores.std(axis=1))
        val_scores_mean = np.sqrt(-val_scores.mean(axis=1))
        val_scores_std = np.sqrt(val_scores.std(axis=1))

        print(f"\n訓練集大小    訓練誤差         驗證誤差         偏差-變異數分析")
        print("─" * 70)

        for i, size in enumerate(train_sizes_abs):
            train_err = train_scores_mean[i]
            val_err = val_scores_mean[i]
            gap = val_err - train_err

            if gap > 0.5:
                diagnosis = "高變異數（過擬合）"
            elif train_err > 0.5:
                diagnosis = "高偏差（欠擬合）"
            else:
                diagnosis = "良好"

            print(f"{size:5d} ({size/len(X)*100:4.1f}%)  "
                  f"{train_err:6.4f} ± {train_scores_std[i]:6.4f}  "
                  f"{val_err:6.4f} ± {val_scores_std[i]:6.4f}  "
                  f"{diagnosis}")

        return {
            'train_sizes': train_sizes_abs,
            'train_scores_mean': train_scores_mean,
            'train_scores_std': train_scores_std,
            'val_scores_mean': val_scores_mean,
            'val_scores_std': val_scores_std
        }

    def plot_residual_analysis(self, y_true, predictions, figsize=(15, 5)):
        """
        殘差分析圖

        Args:
            y_true: 真實值
            predictions: 預測值
            figsize: 圖形大小

        Returns:
            Figure對象
        """
        residuals = y_true - predictions

        fig, axes = plt.subplots(1, 3, figsize=figsize)

        # 1. 殘差 vs 預測值
        axes[0].scatter(predictions, residuals, alpha=0.5)
        axes[0].axhline(y=0, color='r', linestyle='--')
        axes[0].set_xlabel('Predicted Values')
        axes[0].set_ylabel('Residuals')
        axes[0].set_title('Residual Plot')

        # 2. Q-Q plot（常態性檢驗）
        stats.probplot(residuals, dist="norm", plot=axes[1])
        axes[1].set_title('Q-Q Plot')

        # 3. 殘差直方圖
        axes[2].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
        axes[2].set_xlabel('Residuals')
        axes[2].set_ylabel('Frequency')
        axes[2].set_title('Residual Distribution')

        plt.tight_layout()
        return fig
