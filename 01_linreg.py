#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 31 17:26:15 2020

@author: townesf
"""

#%%
import numpy as np
import sklearn as skl
import statsmodels as sm
from matplotlib import pyplot as plt

#%%
rng = np.random.default_rng()
X = rng.normal(loc=5.0,scale=3.0,size=(50,9))
X.mean(0) #should be about 5
X.std(0) #should be about 3

#%%
mu = 33
b = np.array((-3,-1.5,-.5,-.25,-0.01,0.01,.5,1,1.5))
#strong negative: 1,2
#weak negative: 3,4
#insignificant effects: 5,6
#weak positive: 7
#strong positive: 8,9
sigma2 = 4 #fairly high level of noise
y = rng.normal(mu+X@b, np.sqrt(sigma2), size=50)

#%% 
fit1 = skl.linear_model.LinearRegression().fit(X,y) #sklearn
plt.scatter(b,fit1.coef_)
plt.axline((0,0),slope=1,color='r')
X1 = sm.tools.tools.add_constant(X)
fit2 = sm.regression.linear_model.OLS(y,X1).fit()
plt.scatter(b,fit2.params[1:])

#%%
