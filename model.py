# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pandas as pd
import numpy as np
import tempfile
import tensorflow as tf
from tensorflow.contrib.learn import LinearClassifier,DNNClassifier,DNNLinearCombinedClassifier
from sklearn.utils import shuffle

#没用conversionTime creativeID userID appID
CATEGORICAL_COLUMNS = ["clickTime",
                       "positionID", "connectionType", "telecomsOperator",'age',
                       'gender', 'education', 'marriageStatus','haveBaby','hometown',
                       'residence','adID','camgaignID','advertiserID',
                       'appPlatform','appCategory']
#构建模型
def build_estimator(model_dir, model_type):
  """Build an estimator."""
  # Sparse base columns.
  clickTime = tf.contrib.layers.sparse_column_with_integerized_feature(
      "clickTime", bucket_size=24)
  # creativeID = tf.contrib.layers.sparse_column_with_integerized_feature(
  #     "creativeID", bucket_size=7000)
  positionID = tf.contrib.layers.sparse_column_with_integerized_feature(
      "positionID", bucket_size=7646)
  connectionType = tf.contrib.layers.sparse_column_with_integerized_feature(
      "connectionType", bucket_size=5)
  telecomsOperator = tf.contrib.layers.sparse_column_with_integerized_feature(
      "telecomsOperator", bucket_size=4)
  age = tf.contrib.layers.sparse_column_with_integerized_feature(
      "age", bucket_size=81)
  gender =tf.contrib.layers.sparse_column_with_integerized_feature(
      "gender", bucket_size=3)
  education = tf.contrib.layers.sparse_column_with_integerized_feature(
      "education", bucket_size=8)
  marriageStatus = tf.contrib.layers.sparse_column_with_integerized_feature(
      "marriageStatus", bucket_size=4)
  haveBaby= tf.contrib.layers.sparse_column_with_integerized_feature(
      "haveBaby", bucket_size=7)
  hometown= tf.contrib.layers.sparse_column_with_integerized_feature(
      "hometown", bucket_size=365)
  residence= tf.contrib.layers.sparse_column_with_integerized_feature(
      "residence", bucket_size=400)
  adID= tf.contrib.layers.sparse_column_with_integerized_feature(
      "adID", bucket_size=3616)
  camgaignID=tf.contrib.layers.sparse_column_with_integerized_feature(
      "camgaignID", bucket_size=720)
  advertiserID=tf.contrib.layers.sparse_column_with_integerized_feature(
      "advertiserID", bucket_size=91)
  appPlatform=tf.contrib.layers.sparse_column_with_integerized_feature(
      "appPlatform", bucket_size=3)
  appCategory=tf.contrib.layers.sparse_column_with_integerized_feature(
      "appCategory", bucket_size=504)
  wide_columns = [ clickTime,  positionID, connectionType,
                   telecomsOperator,age,gender,education,marriageStatus,haveBaby,
                   hometown,residence,adID,camgaignID,advertiserID,appPlatform,appCategory,
                  # tf.contrib.layers.crossed_column([education, occupation],
                  #                                  hash_bucket_size=int(1e4)),
                  # tf.contrib.layers.crossed_column(
                  #     [age_buckets, education, occupation],
                  #     hash_bucket_size=int(1e6)),
                   tf.contrib.layers.crossed_column([clickTime, connectionType,telecomsOperator],
                                                    hash_bucket_size=int(1e4))
                ]
  deep_columns = [
      tf.contrib.layers.embedding_column(clickTime, dimension=8),
      tf.contrib.layers.embedding_column(positionID, dimension=8),
      tf.contrib.layers.embedding_column(connectionType, dimension=8),
      tf.contrib.layers.embedding_column(telecomsOperator,
                                         dimension=8),
      tf.contrib.layers.embedding_column(age, dimension=8),
      tf.contrib.layers.embedding_column(gender, dimension=8),
      tf.contrib.layers.embedding_column(education, dimension=8),
      tf.contrib.layers.embedding_column(marriageStatus, dimension=8),
      tf.contrib.layers.embedding_column(haveBaby, dimension=8),
      tf.contrib.layers.embedding_column(hometown, dimension=8),
      tf.contrib.layers.embedding_column(residence, dimension=8),
      tf.contrib.layers.embedding_column(adID, dimension=8),
      tf.contrib.layers.embedding_column(camgaignID, dimension=8),
      tf.contrib.layers.embedding_column(advertiserID, dimension=8),
      tf.contrib.layers.embedding_column(appCategory, dimension=8),
      tf.contrib.layers.embedding_column(appPlatform, dimension=8)
  ]
  if model_type == "wide":
    m = LinearClassifier(model_dir=model_dir,
                                          feature_columns=wide_columns)
  elif model_type == "deep":
    m = DNNClassifier(model_dir=model_dir,
                                       feature_columns=deep_columns,
                                       hidden_units=[100, 50])
  else:
    m = DNNLinearCombinedClassifier(
        model_dir=model_dir,
        linear_feature_columns=wide_columns,
        dnn_feature_columns=deep_columns,
        dnn_hidden_units=[100, 50],
        fix_global_step_increment_bug=True)
  return m


def input_fn(df,batch_size=None,mode=None):
  """Input builder function."""
  if mode=='train':
      labels_all = df["label"].values
      shuffle_indices = np.random.permutation(np.arange(len(labels_all)))

      indices=shuffle_indices[0:batch_size]
      labels = labels_all[indices]
      data = {k: df[k].values[indices] for k in CATEGORICAL_COLUMNS}

      categorical_cols = {
          k: tf.SparseTensor(
              indices=[[i, 0] for i in range(batch_size)],
              values=data[k],
              dense_shape=[batch_size, 1])
          for k in CATEGORICAL_COLUMNS}
  else:
      labels =df["label"].values
      categorical_cols = {
          k: tf.SparseTensor(
              indices=[[i, 0] for i in range(df[k].size)],
              values=df[k].values,
              dense_shape=[df[k].size, 1])
          for k in CATEGORICAL_COLUMNS}
  feature_cols = dict(categorical_cols)

  # Converts the label column into a constant Tensor.
  label = tf.constant(labels)
  # Returns the feature columns and the label.
  return feature_cols, label

def submitting(model_type, directory,model_dir):
    #df_train = pd.read_csv(directory + "test_div.csv")
    df_test = pd.read_csv(directory + "sub_all.csv")
    if not model_dir:
        raise Exception("No model file can be loaded")

    m = build_estimator(model_dir, model_type)
    #m.fit(input_fn=lambda: input_fn(df_train), steps=train_steps)
    pred = m.predict_proba(input_fn=lambda: input_fn(df_test), as_iterable=False)
    pred = pred[:, 1]

    df_test['prob'] = pred
    ans = df_test[['instanceID', 'prob']]
    ans.to_csv(directory + model_type+'_submission.csv', index=None)

def training(model_type, directory, batch_size,epoch,model_dir):
    df_train = pd.read_csv(directory + "train_div.csv")
    df_test = pd.read_csv(directory + "test_div.csv")
    train_steps = df_train.shape[0] // batch_size * epoch
    model_dir = tempfile.mkdtemp() if not model_dir else model_dir
    m = build_estimator(model_dir, model_type)
    m.fit(input_fn=lambda: input_fn(df_train,batch_size=batch_size,mode='train'), steps=train_steps)
    results = m.evaluate(input_fn=lambda: input_fn(df_test), steps=1)
    for key in sorted(results):
        print("%s: %s" % (key, results[key]))