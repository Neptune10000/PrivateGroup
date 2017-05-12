# -*- coding:utf-8 -*-

import pandas as pd
from sklearn.cross_validation import train_test_split
# 只适用刚从tecent下载了pre数据的用户，对列表进行拼接，对测试集和训练集进行划分，不需要每次执行程序都运行
def data_preprocess(directory='../pre/', test_size=0.2):

    # Preprocess data, read csv file, split train set and test set.

    train_file_name = 'train.csv'
    submission_file_name = 'test.csv'
    df_train = pd.read_csv(directory + train_file_name)
    df_sub = pd.read_csv(directory + submission_file_name)
    #time bucketing
    df_train.clickTime = df_train.clickTime % 10000 // 100

    df_user = pd.read_csv(directory + 'user.csv')
    df_ad = pd.read_csv(directory + 'ad.csv')
    df_appc = pd.read_csv(directory + 'app_categories.csv')
    df_pos = pd.read_csv(directory + 'position.csv')

    df_ad_app = pd.merge(df_ad, df_appc, on="appID", suffixes=('_a', '_b'))
    df_train_user = pd.merge(df_train, df_user, on="userID", suffixes=('_a', '_b'))
    df_train_user_app = pd.merge(df_train_user, df_ad_app, on="creativeID", suffixes=('_a', '_b'))
    df_train_all= pd.merge(df_train_user_app, df_pos, on="positionID", suffixes=('_a', '_b'))


    xy_train, xy_test = train_test_split(df_train_all, test_size=test_size, random_state=42)

    df_train_all.to_csv(directory + "train_all.csv", index=None)
    xy_train.to_csv(directory + 'train_div.csv', index=None)
    xy_test.to_csv(directory + 'test_div.csv', index=None)


    df_sub_user = pd.merge(df_sub, df_user, on="userID", suffixes=('_a', '_b'))
    df_sub_user_app = pd.merge(df_sub_user, df_ad_app, on="creativeID", suffixes=('_a', '_b'))
    df_sub_all = pd.merge(df_sub_user_app, df_pos, on="positionID", suffixes=('_a', '_b'))
    df_sub_all.sort(columns="instanceID", axis=0)
    df_sub_all.to_csv(directory+"sub_all.csv",index=None)
