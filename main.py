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

def train_and_eval(model_type,directory,SUB,batch_size,epoch,model_dir):
    #模型训练和预测
    if not (os.path.exists(directory+"train_div.csv") or os.path.exists(directory+"sub_all.csv")):
        data_preprocess(directory)

    if(SUB):
        print("use all train data")
        submitting(model_type, directory,model_dir)
    else:
        print("use only 80% train data")
        training(model_type, directory, batch_size, epoch,model_dir)

def main(_):
    train_and_eval(FLAGS.model_type,  FLAGS.directory,FLAGS.SUB,FLAGS.batch_size,FLAGS.epoch,FLAGS.model_dir)

if __name__ =="__main__":
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument("--directory",type=str,default="../pre/", help="directory for data ")
        #default="/data1/NLPRMNT/baihe/TecentCompetition/pre/",
    parser.add_argument("--model_dir", type=str, default="../model/", help="directory for model ")
    parser.add_argument("--SUB",type=bool,default=True,help="True for submission False for experiments ")
    parser.add_argument("--model_type",type=str,default="wide",help="wide, deep or wide_n_deep ")
    parser.add_argument("--batch_size",type=int,default="256",help="batch_size ")
    parser.add_argument("--epoch", type=int, default="1", help="epoch ")
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main,argv=[sys.argv[0]] + unparsed)