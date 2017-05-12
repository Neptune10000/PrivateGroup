# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import argparse

import tensorflow as tf

from Data_preprocessing import data_preprocess
from model import submitting,training
FLAGS = None

COLUMNS = [u'label', u'clickTime', u'conversionTime', u'creativeID', u'userID',
       u'positionID', u'connectionType', u'telecomsOperator', u'age',
       u'gender', u'education', u'marriageStatus', u'haveBaby', u'hometown',
       u'residence', u'adID', u'camgaignID', u'advertiserID', u'appID',
       u'appPlatform', u'appCategory']


# def data_process(train_file_name, test_file_name,directory):
#     print("data processing...")
#     # 特征整合
#     df_train = pd.read_csv(train_file_name)
#     df_train.clickTime = df_train.clickTime%10000//100
#     df_test = pd.read_csv(test_file_name)
#
#     df_user = pd.read_csv(directory+'user.csv')
#     df_ad = pd.read_csv(directory+'ad.csv')
#     df_app = pd.read_csv(directory+'app_categories.csv')
#     df_pos = pd.read_csv(directory+'position.csv')
#
#     df_ad_app = pd.merge(df_ad, df_app, on="appID", suffixes=('_a', '_b'))
#
#     df_train_user = pd.merge(df_train, df_user, on="userID", suffixes=('_a', '_b'))
#     df_train_user_app = pd.merge(df_train_user, df_ad_app, on="creativeID", suffixes=('_a', '_b'))
#     df_train_all = pd.merge(df_train_user_app, df_pos, on="positionID", suffixes=('_a', '_b'))
#
#     df_test_user = pd.merge(df_test, df_user, on="userID", suffixes=('_a', '_b'))
#     df_test_user_app = pd.merge(df_test_user, df_ad_app, on="creativeID", suffixes=('_a', '_b'))
#     df_test_all = pd.merge(df_test_user_app, df_pos, on="positionID", suffixes=('_a', '_b'))
#     df_train_all.to_csv(directory+"train_all.csv",index=None)
#     df_test_all.to_csv(directory+"test_all.csv",index=None)
#     #数据划分0.8
#     train = shuffle(df_train_all)
#     n = int(train['label'].count() * 0.8)
#     train_div = train[:n]
#     test_div = train[n:]
#     train_div.to_csv(directory+"train_div.csv",index=None)
#     test_div.to_csv(directory+"test_div.csv",index=None)
#     return 0

def train_and_eval(model_type,directory,SUB,batch_size,epoch):
    #模型训练和预测
    if not (os.path.exists(directory+"train_div.csv") or os.path.exists(directory+"sub_all.csv")):
        data_preprocess(directory)

    if(SUB):
        print("use all train data")
        submitting(model_type, directory)
    else:
        print("use only 80% train data")
        training(model_type, directory, batch_size, epoch)

def main(_):
    train_and_eval(FLAGS.model_type,  FLAGS.directory,FLAGS.SUB,FLAGS.batch_size,FLAGS.epoch)

if __name__ =="__main__":
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument("--directory",type=str,
        default="/Users/richardbai/Desktop/TecentCompetition/pre/",
        #default="/data1/NLPRMNT/baihe/TecentCompetition/pre/",
        help="directory for data and model ")
    parser.add_argument("--SUB",type=bool,default=True,help="True for submission False for experiments ")
    parser.add_argument("--model_type",type=str,default="wide",help="wide, deep or wide_n_deep ")
    parser.add_argument("--batch_size",type=int,default="256",help="batch_size ")
    parser.add_argument("--epoch", type=int, default="1", help="epoch ")
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main,argv=[sys.argv[0]] + unparsed)