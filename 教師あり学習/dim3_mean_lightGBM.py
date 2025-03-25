import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
import lightgbm as lgb
from lightgbm import LGBMClassifier
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import seaborn as sns
from itertools import product

# 訓練データとテストデータの作成
home_dir = os.path.expanduser("~")
train_path = os.path.join(home_dir, "OneDrive", "デスクトップ", "機械学習", "signate", "SOTA_SIGNATE Student Cup 2021", "csv",  "train.csv")
train = pd.read_csv(train_path)
test_path = os.path.join(home_dir, "OneDrive", "デスクトップ", "機械学習", "signate", "SOTA_SIGNATE Student Cup 2021", "csv",  "test.csv")
test = pd.read_csv(test_path)

# 平均値を計算する関数
def convert_tempo_to_avg(tempo_range):
    low, high = map(int, tempo_range.split('-'))
    return (low + high) / 2  # 平均を返す

# カテゴリ変数の変換
train['tempo'] = train['tempo'].apply(convert_tempo_to_avg)
test['tempo'] = test['tempo'].apply(convert_tempo_to_avg)

unique_categories = pd.concat([train["region"], test["region"]]).unique()
category_map = {category: idx for idx, category in enumerate(unique_categories)}
train["region"] = train["region"].map(category_map)
test["region"] = test["region"].map(category_map)

# train["region"] = train["region"].astype("category")
# test["region"] = test["region"].astype("category")

# 欠損値を含む列のリストを作成
missing_columns_train = train.columns[train.isnull().any()].tolist()
missing_columns_test = test.columns[test.isnull().any()].tolist()

# 両方に存在するカラムのみを使用
common_missing_columns = list(set(missing_columns_train) & set(missing_columns_test))

from sklearn.impute import KNNImputer

# KNNImputerのインスタンス化
imputer = KNNImputer(n_neighbors=10, weights='uniform', metric='nan_euclidean')

# 欠損値を含む列のみを抽出
train_missing = train[common_missing_columns]
test_missing = test[common_missing_columns]

# テストデータセットの欠損値を補完
test_imputed = pd.DataFrame(imputer.fit_transform(test_missing), columns=common_missing_columns)

# トレーニングデータセットの欠損値を補完
train_imputed = pd.DataFrame(imputer.transform(train_missing), columns=common_missing_columns)

# 補完データを元の train, test に適用
train[common_missing_columns] = train_imputed
test[common_missing_columns] = test_imputed

# 共通していない列のリストを作成
train_non_common_missing_columns = list(set(missing_columns_train) - set(common_missing_columns))
test_non_common_missing_columns = list(set(missing_columns_test) - set(common_missing_columns))

# テストデータセットの共通でない欠損値を補完
if test_non_common_missing_columns:
    test_non_common_missing = test[test_non_common_missing_columns]
    test_imputed_non_common = pd.DataFrame(imputer.fit_transform(test_non_common_missing), columns=test_non_common_missing_columns)
    test[test_non_common_missing_columns] = test_imputed_non_common


# 説明変数と予測変数の決定
X = train[["popularity", "duration_ms", "acousticness", "positiveness", "danceability", "loudness", "energy", "liveness", "speechiness", "instrumentalness", "tempo", "region"]]
target = train["genre"]

# 変数変更用
num_trials = 100  # 試行回数
verbose_eval = 0
accuracy_scores = []
param_scales = {"objective": "objective",
         "gbdt": "boosting_type", 
         "num_class": "num_class",
         "depth": "max_depth",
         "leaves": "num_leaves",
         "estimators": "n_estimators",
         "rate": "learning_rate",
         "random": "random_state",
         "samples": "min_child_samples",
         "weight": "min_child_weight",
         "gain": "min_split_gain",
         "alpha": "reg_alpha",
         "lambda": "reg_lambda", 
         "bin": "max_bin"
         }

col_1 = param_scales["samples"]
col_2 = param_scales["weight"]
col_3 = param_scales["gain"]

cv_params = {col_1: [7, 8, 9],  
             col_2: [0.09, 0.1, 0.11], 
             col_3: [0.0009, 0.001, 0.0011] 
             }

for l, n, m in product(cv_params[col_1], cv_params[col_2], cv_params[col_3]):
    for i in range(num_trials):

        # モデルの学習
        X_cv, X_eval, y_cv, y_eval = train_test_split(X, target, test_size=0.60, random_state=None, shuffle=True)
        
        current_seed = 42 + i  # 試行ごとに異なるシード
        
        gbm = lgb.LGBMClassifier(
            objective='multiclass', 
            boosting_type="gbdt", 
            num_class=len(train["genre"].unique()), 
            max_depth=9, 
            num_leaves=min(2**9, 128), 
            n_estimators=185, 
            learning_rate=0.1, 
            min_child_samples=l,
            min_child_weight=m,
            min_split_gain=n,
            reg_alpha=1.1,
            reg_lambda=0.001, 
            max_bin=31, 
            random_state=current_seed
        )
        gbm.fit(
            X_cv, y_cv, 
            eval_set=[(X_eval, y_eval)],  # 評価データセットを指定
            eval_metric='multi_logloss',  # 評価指標
            callbacks=[lgb.early_stopping(stopping_rounds=10000, 
            verbose=True), # early_stopping用コールバック関数
            lgb.log_evaluation(verbose_eval)] # コマンドライン出力用コールバック関数
        )


        # 精度の評価
        test_pred = gbm.predict(X_eval)
        acc = accuracy_score(y_eval, test_pred)

        # 結果を辞書形式で保存
        accuracy_scores.append({col_1:  l, col_2:  n, col_3: m, "trial":  i+1, "accuracy":  acc})

# 結果を DataFrame に保存（必要なら CSV に出力）
df_results = pd.DataFrame(accuracy_scores)
df_results.to_csv("lgbm_tuning_results.csv", index=False)


# 結果を読み込む
df_results = pd.read_csv("lgbm_tuning_results.csv")

# 深さと学習率ごとにグループ化して平均精度を計算
avg_accuracy = df_results.groupby([col_1, col_2, col_3])['accuracy'].mean().reset_index()

# 結果を表示
print("各パラメータ組み合わせごとの平均精度:")
print(avg_accuracy)

# ピボットテーブルを作成して見やすくする
pivot_table = avg_accuracy.pivot_table(index=[col_1, col_2], columns=col_3, values='accuracy')
print("\n平均精度のピボットテーブル:")
print(pivot_table)

# 最も精度が高いパラメータの組み合わせを見つける
best_param_scaless = avg_accuracy.loc[avg_accuracy['accuracy'].idxmax()]
print(f"\n最も精度が高いパラメータの組み合わせ:")
print(f"{col_1}: {best_param_scaless[col_1]}")
print(f"{col_2}: {best_param_scaless[col_2]}")
print(f"{col_3}: {best_param_scaless[col_3]}")
print(f"平均精度: {best_param_scaless['accuracy']:.4f}")

# ヒートマップで可視化
plt.figure(figsize=(10, 8))
sns.heatmap(pivot_table, annot=True, cmap='YlGnBu', fmt='.4f')
plt.title('パラメータごとの平均精度')
plt.tight_layout()
plt.savefig('accuracy_heatmap.png')
plt.show()

# 各パラメータの影響を見るためのラインプロット
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# col_1ごとの平均精度
col_1_avg = df_results.groupby(col_1)['accuracy'].mean()
axes[0].plot(col_1_avg.index, col_1_avg.values, 'o-')
axes[0].set_xlabel(col_1)
axes[0].set_ylabel('Average Accuracy')
axes[0].set_title(f'{col_1}と平均精度の関係')
axes[0].grid(True)

# col_2ごとの平均精度
col_2_avg = df_results.groupby(col_2)['accuracy'].mean()
axes[1].plot(col_2_avg.index, col_2_avg.values, 'o-')
axes[1].set_xlabel(col_2)
axes[1].set_ylabel('Average Accuracy')
axes[1].set_title(f'{col_2}と平均精度の関係')
axes[1].grid(True)

# col_3ごとの平均精度
col_3_avg = df_results.groupby(col_3)['accuracy'].mean()
axes[1].plot(col_3_avg.index, col_3_avg.values, 'o-')
axes[1].set_xlabel(col_3)
axes[1].set_ylabel('Average Accuracy')
axes[1].set_title(f'{col_3}と平均精度の関係')
axes[1].grid(True)

plt.tight_layout()
plt.savefig('param_scaleseter_effects.png')
plt.show()

# 各パラメータの組み合わせにおける精度の分布（箱ひげ図）
plt.figure(figsize=(12, 8))
# パラメータ組み合わせを文字列化
df_results['param_scales_combo'] = df_results[col_1].astype(str) + '_' + df_results[col_2].astype(str)
# hue を使って samples ごとに色分け
sns.boxplot(x='param_scales_combo', y='accuracy', hue=col_3, data=df_results)
plt.xticks(rotation=90)
plt.xlabel('Param_scaleseter Combination (depth_rate)')
plt.ylabel('Accuracy')
plt.title(f'各パラメータ組み合わせにおける精度の分布（{col_3}別）')
plt.legend(title='Samples', bbox_to_anchor=(1.05, 1), loc='upper left')  # 凡例を見やすく配置
plt.tight_layout()
plt.savefig('accuracy_boxplot_samples.png')
plt.show()

min_child_samples: 100.0
reg_alpha: 0.0001
reg_lambda: 1.0
平均精度: 0.5974