"""
進階統計分析與機器學習示範
展示統計驅動的機器學習pipeline
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# 添加專案路徑
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

from preprocessing.data_loader import DataLoader
from preprocessing.feature_engineering import FeatureEngineer
from analysis.feature_analyzer import FeatureAnalyzer
from models.ensemble_models import EnsembleModelTrainer
from analysis.model_evaluator import ModelEvaluator


def prepare_data(df):
    """準備建模數據"""
    # 編碼特徵
    df_encoded = FeatureEngineer.encode_features(df)
    numeric_df = df_encoded.select_dtypes(include=['number'])

    # 移除不需要的欄位
    drop_cols = ['總額元', '每坪租金', '單價元平方公尺', '車位總額元',
                 '租賃年月日', '建築完成年月', '編號', '租賃筆棟數']
    cols_to_drop = [c for c in drop_cols if c in numeric_df.columns]

    X = numeric_df.drop(columns=cols_to_drop)
    X = X.fillna(X.mean())
    X = X.loc[:, X.var() > 0]

    y = numeric_df['總額元']

    return X, y


def main():
    """主流程"""
    print("\n" + "="*80)
    print("進階統計分析與機器學習 Pipeline")
    print("="*80)

    # ===== 步驟 1: 載入數據 =====
    print("\n[步驟 1/6] 載入數據...")
    loader = DataLoader(data_dir='ml/data')
    df = loader.load_featured_data()
    print(f"✓ 載入 {len(df)} 筆資料")

    # ===== 步驟 2: 特徵統計分析 =====
    print("\n[步驟 2/6] 特徵統計分析...")

    analyzer = FeatureAnalyzer(df, target_col='總額元')

    # 2.1 相關性分析
    print("\n" + "─"*80)
    print("2.1 Pearson 相關性分析")
    print("─"*80)
    corr_pearson = analyzer.correlation_analysis(method='pearson', top_n=15)

    print("\n" + "─"*80)
    print("2.2 Spearman 相關性分析（秩相關）")
    print("─"*80)
    corr_spearman = analyzer.correlation_analysis(method='spearman', top_n=15)

    # 2.2 VIF分析
    print("\n" + "─"*80)
    print("2.3 多重共線性診斷")
    print("─"*80)
    vif_results = analyzer.vif_analysis(threshold=10)

    # 2.3 互信息分析
    print("\n" + "─"*80)
    print("2.4 互信息分析（非線性關係）")
    print("─"*80)
    mi_scores = analyzer.mutual_information_analysis(top_n=15)

    # ===== 步驟 3: 準備建模數據 =====
    print("\n[步驟 3/6] 準備建模數據...")
    X, y = prepare_data(df)

    # 切分數據
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 標準化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"✓ 訓練集: {X_train.shape[0]} 筆")
    print(f"✓ 測試集: {X_test.shape[0]} 筆")
    print(f"✓ 特徵數: {X_train.shape[1]}")

    # ===== 步驟 4: 訓練集成學習模型 =====
    print("\n[步驟 4/6] 訓練集成學習模型...")

    trainer = EnsembleModelTrainer(random_state=42)
    results = trainer.train_all_models(
        X_train_scaled, y_train,
        X_test_scaled, y_test,
        use_log=True
    )

    # ===== 步驟 5: 模型比較 =====
    print("\n[步驟 5/6] 模型效能比較與統計檢驗...")

    # 5.1 基本比較
    comparison_df = trainer.compare_models()

    # 5.2 特徵重要性分析
    print("\n" + "─"*80)
    print("5.1 特徵重要性分析")
    print("─"*80)

    for model_name in ['Random Forest', 'Gradient Boosting']:
        if model_name in trainer.models:
            importance_df = trainer.get_feature_importance(
                model_name,
                X_train.columns.tolist(),
                top_n=15
            )

    # 5.3 統計檢驗
    evaluator = ModelEvaluator(random_state=42)

    # 獲取模型預測
    model_names = list(results.keys())
    if len(model_names) >= 2:
        print("\n" + "─"*80)
        print("5.2 統計假設檢驗")
        print("─"*80)

        pred1 = results[model_names[0]]['predictions']
        pred2 = results[model_names[1]]['predictions']

        # 配對t檢驗
        print(f"\n比較模型: {model_names[0]} vs {model_names[1]}")
        t_test_result = evaluator.paired_t_test(
            pred1, pred2, y_test, alpha=0.05
        )

        # Diebold-Mariano檢驗
        errors1 = y_test - pred1
        errors2 = y_test - pred2
        dm_result = evaluator.diebold_mariano_test(
            errors1, errors2, alpha=0.05
        )

    # Friedman檢驗（如果有3個以上模型）
    if len(model_names) >= 3:
        print("\n" + "─"*80)
        print("5.3 多模型比較（Friedman檢驗）")
        print("─"*80)

        predictions_dict = {
            name: results[name]['predictions']
            for name in model_names
        }
        friedman_result = evaluator.friedman_test(
            predictions_dict, y_test, alpha=0.05
        )

    # ===== 步驟 6: Bootstrap信賴區間 =====
    print("\n[步驟 6/6] Bootstrap 不確定性量化...")

    if len(model_names) >= 1:
        best_model_name = model_names[0]
        best_predictions = results[best_model_name]['predictions']

        print(f"\n分析模型: {best_model_name}")

        # RMSE的信賴區間
        rmse_ci = evaluator.bootstrap_confidence_interval(
            y_test.values,
            best_predictions,
            metric_func=lambda y, p: np.sqrt(mean_squared_error(y, p)),
            n_bootstrap=1000,
            confidence=0.95
        )

        # R²的信賴區間
        from sklearn.metrics import r2_score
        r2_ci = evaluator.bootstrap_confidence_interval(
            y_test.values,
            best_predictions,
            metric_func=lambda y, p: r2_score(y, p),
            n_bootstrap=1000,
            confidence=0.95
        )

    # ===== 總結 =====
    print("\n" + "="*80)
    print("分析完成！")
    print("="*80)

    print("\n主要發現：")
    print(f"1. 最佳模型: {comparison_df.iloc[0]['模型']}")
    print(f"   - R² = {comparison_df.iloc[0]['R²']:.4f}")
    print(f"   - RMSE = {comparison_df.iloc[0]['RMSE']:,.0f} 元")

    if corr_pearson is not None and len(corr_pearson) > 0:
        top_feature = corr_pearson.index[0]
        top_corr = corr_pearson.iloc[0]
        print(f"\n2. 最重要特徵（相關性）: {top_feature}")
        print(f"   - Pearson相關係數 = {top_corr:.4f}")

    if mi_scores is not None and len(mi_scores) > 0:
        top_mi_feature = mi_scores.index[0]
        top_mi = mi_scores.iloc[0]
        print(f"\n3. 最重要特徵（互信息）: {top_mi_feature}")
        print(f"   - 互信息分數 = {top_mi:.4f}")

    if vif_results is not None:
        high_vif = vif_results[vif_results['VIF'] > 10]
        if len(high_vif) > 0:
            print(f"\n4. 共線性警告: {len(high_vif)} 個特徵的VIF > 10")
            print("   建議進一步檢查或移除這些特徵")

    print("\n" + "="*80)
    print()


if __name__ == '__main__':
    main()
