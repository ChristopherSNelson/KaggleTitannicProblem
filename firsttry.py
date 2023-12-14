# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

#import numpy as np # linear algebra
#import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

#from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

"""
Chris Nelson 2017 
Machine learning self-tutelage from Kaggle.com example problem. 
Problem: make a classifier for whether passengers survived the Titanic disaster.
"""
## "source /nfs/share/env/python-2.7.env" gives you anacondas python v2.7.9.

from statsmodels.regression.linear_model import OLS #this version of OLS supports R-style formulas for model definition.
from statsmodels.formula.api import logit
#import numpy as np
#from time import time
from pandas import read_csv
#from pandas import merge
from pandas import DataFrame

#load data
trainData=read_csv('../input/train.csv')
testData=read_csv('../input/test.csv')

#parse data
#print testData
#print trainData['Cabin']
#print "_______"
trainData['Deck']=[ str(i)[0] if str(i)!="nan" else i for i in trainData['Cabin'] ]# grab the deck from the cabin string and use that for a new Deck variable.
testData['Deck']=[ str(i)[0] if str(i)!="nan" else i for i in testData['Cabin'] ]
#print trainData['Cabin']

#fill in missing data if necessary.


#make the model
modelFormula='Survived~Sex+Age+C(Pclass)+Deck+SibSp+Parch'#+Fare+SibSp+parch
#modelFormula='Survived~Sex+Age+C(Pclass)+Deck'# R2=0.3724 , not overfit...
#modelFormula='Survived~Sex+Age+C(Pclass)+Deck+Deck:Fare'# breaks logit with singular matrix

# fit model
fitModel= OLS.from_formula(modelFormula, data=trainData).fit()
#fitModel= logit(modelFormula, data=trainData).fit()# this will run... psuedoR2=0.3563


# examine fit model
print (fitModel.summary())
#print fitModel.pred_table()# a little 2x2 confusion matrix. I think.
#margEffects = fitModel.get_margeff()
#print(margEffects.summary())

fitModelpVals= fitModel.pvalues#[0:-1] 
print(fitModelpVals)

#test model against test data.
prediction= fitModel.predict(testData)# this only operates on variables with zero missing information. gotta impute! #########################################
#print len(prediction)
predictionRounded=[1 if p>=0.5 else 0 for p in prediction]# rounding survival prediction.
print(predictionRounded)

Output=DataFrame({'PassengerId':testData['PassengerId'],'Survived':predictionRounded})# dunno if the same ordering assumption for these columns is 100% valid.
Output.to_csv('BabysFirstPrediction.csv', index=False)