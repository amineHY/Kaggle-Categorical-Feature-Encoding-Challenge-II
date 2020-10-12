# -*- coding: UTF-8 -*-
import time
import joblib
import pandas as pd
import xgboost as xgb
from sklearn import metrics, preprocessing
import argparse
import config
import os
import model_dispatcher


def run(model, fold):
    # load the full training data with folds
    df = pd.read_csv(config.TRAINING_FILE)

    print('[INFO]  Data preparation')
    # all columns are features except id, target and kfold columns
    features = [f for f in df.columns if f not in ("id", "target", "kfold")]

    # fill all NaN values with NONE
    # note that I am converting all columns to "strings"
    # it doesnt matter because all are categories
    for col in features:
        df.loc[:, col] = df[col].astype(str).fillna("NONE")

    # now it’s time to label encode the features
    for col in features:
        # initialize LabelEncoder for each feature column
        lbl = preprocessing.LabelEncoder()

        # fit label encoder on all data
        lbl.fit(df[col])

        # transform all the data
        df.loc[:, col] = lbl.transform(df[col])

    # get training data using folds
    df_train = df[df.kfold != fold].reset_index(drop=True)

    # get validation data using folds
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    # get training data
    x_train = df_train[features].values

    # get validation data
    x_valid = df_valid[features].values

    print('[INFO]  Training')
    # fit model on training data (ohe)
    clf = model_dispatcher.models[model]
    clf.fit(x_train, df_train.target.values)

    print('[INFO]  Validation')
    # predict on validation data
    # we need the probability values as we are calculating AUC
    # we will use the probability of 1s
    valid_preds = clf.predict_proba(x_valid)[:, 1]

    print('[INFO]  Evaluation metric')
    # get roc auc score
    auc = metrics.roc_auc_score(df_valid.target.values, valid_preds)

    # print auc
    print("Fold = {}, AUC = {}".format(fold, auc))

    # save the model
    joblib_file = os.path.join(config.MODEL_OUTPUT, f"dt_{fold}.bin")
    joblib.dump(model, joblib_file)

    # Load from file
    # joblib_model = joblib.load(joblib_file)


if __name__ == "__main__":
    tic = time.time()

    # initialize ArgumentParser class of argparse
    parser = argparse.ArgumentParser()

    # add the different arguments you need and their type
    # currently, we only need fold
    # parser.add_argument("--fold", type=int, default=0)

    # enter the model
    parser.add_argument("--model", type=str, default='rf')

    # read the arguments from the command line
    args = parser.parse_args()

    for fold_ in range(5):
        run(args.model, fold_)

        print("[INFO] Elapsed time: {0:2.2f} seconds, fold={1}, model={2}".format(
            time.time()-tic, fold_, args.model))
