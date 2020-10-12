# Kaggle challenge: [Categorical Feature Encoding Challenge II](https://www.kaggle.com/c/cat-in-the-dat-ii/data)

# Run the model

```
python3 src/train.py
```

# Setup

- For this data set we use a tree based model, gradient boosting.
- Encoder: The model is a tree based, hence, we use **Label Encoding** as a method to encode categorical variables. In such setup there is no need to normalize the variables.
- Cross validation : Stratified k-fold cross validation method since the target is skew (the amount of labels 0 and 1s is inbalanced)
- Evaluation metric: The area under the curve (AUC) as the data is skewed

# Output

```
[INFO]  Data preparation
[INFO]  Training
[INFO]  Validation
[INFO]  Evaluation metric
Fold = 0, AUC = 0.7161123146808881
[INFO] Elapsed time: 49.70 seconds, fold=0, model=rf
[INFO]  Data preparation
[INFO]  Training
[INFO]  Validation
[INFO]  Evaluation metric
Fold = 1, AUC = 0.7167649537561351
[INFO] Elapsed time: 99.89 seconds, fold=1, model=rf
[INFO]  Data preparation
[INFO]  Training
[INFO]  Validation
[INFO]  Evaluation metric
Fold = 2, AUC = 0.7158797768053974
[INFO] Elapsed time: 150.55 seconds, fold=2, model=rf
[INFO]  Data preparation
[INFO]  Training
[INFO]  Validation
[INFO]  Evaluation metric
Fold = 3, AUC = 0.71750255322751
[INFO] Elapsed time: 202.63 seconds, fold=3, model=rf
[INFO]  Data preparation
[INFO]  Training
[INFO]  Validation
[INFO]  Evaluation metric
Fold = 4, AUC = 0.7138236220817784
[INFO] Elapsed time: 252.68 seconds, fold=4, model=rf
```
