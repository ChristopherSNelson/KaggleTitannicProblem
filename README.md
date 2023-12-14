# KaggleTitannicProblem
Kaggle python code - Note the linear model scores higher than the XGboost approach.

2017
firsttry.py score 0.76
used modelFormula='Survived~Sex+Age+C(Pclass)+Deck+SibSp+Parch
fit with OLS.from_formula from statsmodels

2023
XGboost score 0.71
in training hit 0.82 (overtrained apparently)
features = ["Pclass","Sex","cabin_letter","norm_fare"]
used RandomizedSearchCV over a parameter grid for final model
