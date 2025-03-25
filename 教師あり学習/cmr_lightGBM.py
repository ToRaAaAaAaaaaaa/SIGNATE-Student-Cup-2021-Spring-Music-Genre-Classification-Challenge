import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from collections import defaultdict

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

# 欠損値を含む列のリストを作成
missing_columns_train = train.columns[train.isnull().any()].tolist()
missing_columns_test = test.columns[test.isnull().any()].tolist()

# 両方に存在するカラムのみを使用
common_missing_columns = list(set(missing_columns_train) & set(missing_columns_test))

from sklearn.impute import KNNImputer

# KNNImputerのインスタンス化
imputer = KNNImputer(n_neighbors=5, weights='uniform', metric='nan_euclidean')

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

# # 特徴量のスケーリング（オプション）
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

# モデルの学習
X_train, X_test, y_train, y_test = train_test_split(X, target, test_size=0.5)


# 結果を保存する辞書（リストを保持する）
pre_score = defaultdict(list)
num_trials = 20  # 試行回数
verbose_eval = 0


for i in np.arange(0.08, 0.10, 0.01):
    for n in range(7, 11):
        gbm = lgb.LGBMClassifier(
            objective='multiclass', 
            num_class=len(train["genre"].unique()), 
            max_depth=n, 
            num_leaves=2**n, 
            n_estimators=21, 
            learning_rate=i
        )
        gbm.fit(X_train, y_train, 
                eval_set=[(X_test, y_test)],  # 評価データセットを指
                eval_metric='multi_logloss',  # 評価指標
                callbacks=[lgb.early_stopping(stopping_rounds=35, 
                                              verbose=True), # early_stopping用コールバック関数
                                              lgb.log_evaluation(verbose_eval)] # コマンドライン出力用コールバック関数             # 進捗表示を非表示
        )
        # 精度の評価
        test_pred = gbm.predict(X_test)

        # 各 learning_rate に対するスコアをリストで保持
        pre_score[f"learning_rate={i}, depth={n}"].append(accuracy_score(y_test, test_pred))

# 各 learning_rate に対する平均精度を計算
avg_pre_score = {key: np.mean(values) for key, values in pre_score.items()}

# 結果の表示
for key, avg_score in avg_pre_score.items():
    print(f"{key}: 平均精度 = {avg_score:.4f}")

# print("Training set accuracy: {:.2f}%".format(accuracy_score(y_train, train_pred) * 100))
# print("Test set accuracy: {:.2f}%".format(accuracy_score(y_test, test_pred) * 100))


# 特徴量の重要度比較
# lgb.plot_importance(gbm, figsize=(8,4), max_num_features=11, importance_type='gain')
# plt.show()