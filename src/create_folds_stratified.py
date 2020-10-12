from sklearn import model_selection
import pandas as pd
import config


if __name__ == "__main__":
    # Training data is in a CSV file called train.csv
    df = pd.read_csv(config.TRAINING_FILE)

    # we create a new column called kfold and fill it with -1
    df["kfold"] = -1

    # the next step is to randomize the rows of the data
    df = df.sample(frac=1).reset_index(drop=True)
    y = df.target.values

    # initiate the kfold class from model_selection module
    kf = model_selection.StratifiedKFold(n_splits=5)

    # fill the new kfold column
    for fold, (trn_, val_) in enumerate(kf.split(X=df, y=y)):
        df.loc[val_, 'kfold'] = fold

    # save the new csv with kfold column
    df.to_csv(config.TRAINING_FILE[:-4]+'_folds.csv', index=False)
