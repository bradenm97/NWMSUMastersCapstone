import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

df = pd.read_csv('Capstonedata.csv')

#attributes vs mpg graphs

import seaborn as sns
pp = sns.pairplot(data=df,
                  y_vars=['MPG'],
                  x_vars=['cylinders', 'displ', 'FWD','RWD', 'AWD'])

pp2 = sns.pairplot(data=df,
                  y_vars=['MPG'],
                  x_vars=['Gas','Diesel ', 'Manual', 'Automatic','year'])





#creating training and test split

from sklearn.model_selection import train_test_split

X = df.drop('MPG', axis =1)
y = df.MPG.values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#stats model regression

import statsmodels.api as sm

X_sm = X = sm.add_constant(X)
model = sm.OLS(y,X_sm)
print(model.fit().summary())



#sklearn model regression
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.model_selection import cross_val_score
lm = LinearRegression()
lm.fit(X_train, y_train)
print('Sklearn Linear Regression MAE: ')
print(np.mean(cross_val_score(lm,X_train,y_train, scoring = 'neg_mean_absolute_error', cv= 3)))

#lasso regression
lm_l = Lasso(alpha=.13)
print('Sklearn Lasso Regression MAE: ')
print(np.mean(cross_val_score(lm_l,X_train,y_train, scoring = 'neg_mean_absolute_error', cv= 3)))

alpha = []
error = []

for i in range(1,100):
    alpha.append(i/100)
    lml = Lasso(alpha=(i/100))
    error.append(np.mean(cross_val_score(lml ,X_train,y_train, scoring = 'neg_mean_absolute_error', cv= 3)))

# random forest

from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()
np.mean(cross_val_score(rf,X_train,y_train, scoring = 'neg_mean_absolute_error', cv= 3))

#tune model

from sklearn.model_selection import GridSearchCV

#default parameters were used
parameters = {}

ls = GridSearchCV(lm,parameters, scoring = 'neg_mean_absolute_error', cv = 3) 
ls.fit(X_train,y_train)

lmls = GridSearchCV(lml,parameters, scoring = 'neg_mean_absolute_error', cv = 3) 
lmls.fit(X_train,y_train)

gs = GridSearchCV(rf,parameters, scoring = 'neg_mean_absolute_error', cv = 3)
gs.fit(X_train,y_train)


#test ensamble

tpred_lm = ls.best_estimator_.predict(X_test)
tpred_lml = lmls.best_estimator_.predict(X_test)
tpred_rf = gs.best_estimator_.predict(X_test)

from sklearn.metrics import mean_absolute_error
print('Linear Regresion MAE: ', mean_absolute_error(y_test,tpred_lm))
print('Lasso Regression MAE: ', mean_absolute_error(y_test,tpred_lml))
print('Random Forrest MAE: ', mean_absolute_error(y_test,tpred_rf))
from sklearn.metrics import explained_variance_score
print('Linear Explained var score: ', explained_variance_score(y_test,tpred_lm))
print('Lasso Explained var score: ', explained_variance_score(y_test,tpred_lml))
print('RF Explained var score: ', explained_variance_score(y_test,tpred_rf))
from sklearn.metrics import r2_score
print('Linear R squared score: ', r2_score(y_test,tpred_lm))
print('Lasso R squared score: ', r2_score(y_test,tpred_lml))
print('RF R squared score: ', r2_score(y_test,tpred_rf))
