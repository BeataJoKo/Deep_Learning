# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 11:37:15 2024

@author: BeButton
"""

#%%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS

#%%
df = pd.read_csv('Data/auto.csv')

#%%
auto = df[df['horsepower'] != '?']
auto = auto.astype({'horsepower': 'int32'})
auto = auto.set_index('name')

#%%
#sns.pairplot(auto, diag_kws={'color':'red'})
g = sns.pairplot(auto, diag_kind="kde", diag_kws={'color':'red'})
g.map_lower(sns.kdeplot, levels=4, color=".2")
plt.savefig('Model/auto.png')
plt.show()

#%%
sns.pairplot(data=auto,
                  y_vars=['mpg'],
                  x_vars=[x for x in auto.columns if x != 'mpg'], 
                  height=4)
plt.savefig('Model/auto_mpg.png')
plt.show()

#%%
x = auto['horsepower'].values
y = auto['mpg'].values

#%%
x = sm.add_constant(x)
result = sm.OLS(y, x).fit()
print(result.summary())

#%%
print(result.params)
print(result.tvalues)

#%%
x = auto['horsepower'].tolist()
y = auto['mpg'].tolist()
plt.scatter(x, y)

# finding the maximum and minimum
max_x = auto['horsepower'].max()
min_x = auto['horsepower'].min()

# the regression line
x = np.arange(min_x, max_x)

# the substituted equation
y = -0.1578 * x + 39.9359

# plotting the regression line
plt.plot(y, 'r')
plt.show()

#%%
formula = "mpg ~ horsepower"
mod = OLS.from_formula(formula, data=auto)
res = mod.fit()

#%%
print(res.summary())
print(res.params)
print(res.tvalues)

#%%
plt.scatter(auto['horsepower'], auto['mpg'])
fig = sm.graphics.abline_plot(model_results=res, color='red', ax=plt.gca())
plt.savefig('Model/auto_ols.png')

#%%
print(auto.columns)

#%%
formula = "mpg ~ horsepower + cylinders + displacement + weight + acceleration + model_year + origin"
model = OLS.from_formula(formula, data=auto)
result = model.fit()

#%%
print(result.summary())
print(result.params)
print(result.tvalues)

#%%
#fig = sm.graphics.plot_ccpr(result, 'mpg')
#plt.savefig('Model/auto_ols_full.png')





