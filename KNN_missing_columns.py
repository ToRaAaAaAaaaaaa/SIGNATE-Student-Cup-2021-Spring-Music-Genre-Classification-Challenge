import numpy as np
import pandas as pd
import os

# データの読み込み
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

print('欠損値のある列（トレーニングデータ）：', missing_columns_train)
print('欠損値のある列（テストデータ）：', missing_columns_test)

# 両方に存在するカラムのみを使用
common_missing_columns = list(set(missing_columns_train) & set(missing_columns_test))
print('欠損値のある列（共通）', common_missing_columns)

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

# 修復後の欠損値の確認
print("修復後の欠損値（トレーニングデータ）:")
print(train_imputed.isnull().sum())

print("\n修復後の欠損値（テストデータ）:")
print(test_imputed.isnull().sum())

# 補完データを元の train, test に適用
train[common_missing_columns] = train_imputed
test[common_missing_columns] = test_imputed

# 共通していない列のリストを作成
train_non_common_missing_columns = list(set(missing_columns_train) - set(common_missing_columns))
test_non_common_missing_columns = list(set(missing_columns_test) - set(common_missing_columns))

# # トレーニングデータに対して欠損値補完
# if train_non_common_missing_columns:
#     train_non_common_missing = train[train_non_common_missing_columns]
#     train_imputed_non_common = pd.DataFrame(imputer.fit_transform(train_non_common_missing), columns=train_non_common_missing_columns)
#     train[train_non_common_missing_columns] = train_imputed_non_common

# テストデータセットの共通でない欠損値を補完
if test_non_common_missing_columns:
    test_non_common_missing = test[test_non_common_missing_columns]
    test_imputed_non_common = pd.DataFrame(imputer.fit_transform(test_non_common_missing), columns=test_non_common_missing_columns)
    test[test_non_common_missing_columns] = test_imputed_non_common

# 欠損値を含む列のリストを作成
remissing_columns_train = train.columns[train.isnull().any()].tolist()
remissing_columns_test = test.columns[test.isnull().any()].tolist()

print('欠損値のある列（トレーニングデータ）：', remissing_columns_train)
print('欠損値のある列（テストデータ）：', remissing_columns_test)