import os
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler

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
features = train[["popularity", "duration_ms", "acousticness", "positiveness", "danceability", "loudness", "energy", "liveness", "speechiness", "instrumentalness", "tempo", "region"]]
target = train["genre"]
test_features = test[["popularity", "duration_ms", "acousticness", "positiveness", "danceability", "loudness", "energy", "liveness", "speechiness", "instrumentalness", "tempo", "region"]]

# モデルの学習
lgbc = lgb.LGBMClassifier(
    objective='multiclass', 
    num_class=len(train["genre"].unique()), 
    max_depth=9, 
    num_leaves=min(2**9, 128), 
    n_estimators=185, 
    learning_rate=0.1, 
    min_child_samples=7,
    min_child_weight=0.01,
    min_split_gain=0.0001,
    reg_alpha=1.1,
    reg_lambda=0.001, 
    max_bin=31, 
    random_state=42
)
lgbc.fit(features, target)
my_prediction = lgbc.predict(test_features)

# csvの作成
index = np.array(test["index"]).astype(int)
data = list(zip(index, my_prediction))  # 行単位でデータをまとめる
my_solution = pd.DataFrame(data)
my_solution.to_csv("my_tree_one.csv", index=False, header=False)