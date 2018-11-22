#!/usr/bin/env python


import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import cross_validation, metrics
import matplotlib.pylab as plt
from sklearn.ensemble import RandomForestRegressor


train = pd.read_csv('train.csv',index_col='PassengerId')
test = pd.read_csv('test.csv',index_col='PassengerId')

"header: PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked"
x=train[['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']]
y=train['Survived']


clf = RandomForestClassifier(oob_score=True, random_state=10)
clf.fit(x,y)

y_pred = clf.predict_proba(x)[:,1]
metrics.roc_auc_score(y,y_pred)

#DataPreprocessing
#choose training data to predict age
age_df = train[['Age','Survived','Fare', 'Parch', 'SibSp', 'Pclass']]
age_df_notnull = age_df.loc[(train_data['Age'].notnull())]
age_df_isnull = age_df.loc[(train_data['Age'].isnull())]
X = age_df_notnull.values[:,1:]
Y = age_df_notnull.values[:,0]
# use RandomForestRegression to train data
RFR = RandomForestRegressor(n_estimators=1000, n_jobs=-1)
RFR.fit(X,Y)
predictAges = RFR.predict(age_df_isnull.values[:,1:])
train_data.loc[train_data['Age'].isnull(), ['Age']]= predictAges

	
#ObserveData
#1. if survival ratio correlated with sex?
train.groupby(['Sex','Survived'])['Survived'].count()
#Sex Survived female 0 81 1 233 male 0 468 1 109 Name: Survived, dtype: int64
train[['Sex','Survived']].groupby(['Sex']).mean().plot.bar()

#2. if Pclass correlated with survival ratio?
train_data.groupby(['Pclass','Survived'])['Pclass'].count()
#Pclass Survived 1 0 80 1 136 2 0 97 1 87 3 0 372 1 119 Name: Pclass, dtype: int64
train_data[['Pclass','Survived']].groupby(['Pclass']).mean().plot.bar()
def regression_model():
	return

	
def train():
	return 
	
	
def test():
	return 
	
	
def main():
	return