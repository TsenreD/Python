import pandas
import pandas as pd
import pyarrow.parquet as pq
import numpy as np
import pyarrow as pa

import pyarrow.csv as csv


def add_features_history(df: pandas.DataFrame) -> pandas.DataFrame:
    """
    Adds features according to the history parquet section
    """
    df['cnt_bad_sell'] = df['cnt_not_sell'] >= 1
    df['cnt_many_takes'] = df['cnt_takes'] >= 4
    return df

def add_features_user_info(df: pandas.DataFrame) -> pandas.DataFrame:
    """
    Adds features according to the user info parquet section
    """
    df['gender_match'] = df['client_gender'] == df['seller_gender']
    feats = ['gender_match']

    return df

def add_features_banks(df: pandas.DataFrame) -> pandas.DataFrame:
    """
    Adds features according to the bank info parquet section
    """
    df['is_bank_256'] = df['tochka_contractor_bank_top_1'] == 'bank_256'
    df['big_revenue'] = (df.revenue != '<10000000') & (df.revenue != 'unknown')
    df.drop(columns=['tochka_contractor_bank_top_1', 'time_tz_diff'], inplace=True)

    return df

def add_features_misc(df: pandas.DataFrame) -> pandas.DataFrame:
    """
    Adds other miscellaneous features
    """
    df['young_and_powerful'] = (df['is_youngest_in_holding'] == True) & (df['is_biggest_in_holding'] == True)

    return df

def add_features(df: pandas.DataFrame) -> pandas.DataFrame:
    """
    Adds all the new features to the copy of a given dataframe
    """
    df = df.copy()
    df = add_features_history(df)
    df = add_features_user_info(df)
    df = add_features_banks(df)
    df = add_features_misc(df)
    return df

def get_train_test_df():
    """
    Builds train and test dataframes from df.csv file
    :return train_df: pandas.DataFrame, test_df: pandas.DataFrame:
    """
    df = pd.read_csv("df.csv")
    df_pre = df.set_index('id')
    df = add_features(df_pre)
    df = pd.get_dummies(df)
    test_df = df[df.target.isnull()].drop(columns=['target'], axis=1)
    train_df = df[df.target.notnull()]
    const_columns_to_remove = []
    for col in train_df.columns:
        if col != 'id' and col != 'target':
            if train_df[col].std() == 0:
                const_columns_to_remove.append(col)

    # Now remove that array of const columns from the data
    train_df.drop(const_columns_to_remove, axis=1, inplace=True)
    test_df.drop(const_columns_to_remove, axis=1, inplace=True)

    return train_df, test_df
