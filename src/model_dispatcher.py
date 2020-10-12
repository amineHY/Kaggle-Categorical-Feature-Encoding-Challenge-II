from sklearn import ensemble
from xgboost import XGBClassifier
from sklearn import tree

# Define tree based machine learning models
tree = tree.DecisionTreeClassifier(random_state=0)
rf = ensemble.RandomForestClassifier(random_state=0, n_jobs=-1)
xgb = XGBClassifier(
    n_jobs=-1,
    max_depth=7,
    n_estimators=200, random_state=0
)

models = {
    'xgb': xgb,
    'rf': rf,
    'tree': tree
}
