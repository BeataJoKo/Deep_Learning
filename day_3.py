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
import statsmodels.formula.api as smf
from statsmodels.regression.linear_model import OLS

#%%
df2 = pd.read_csv('Data/auto.csv', sep=',', na_values='?')
df2 = df2.dropna()

df = pd.read_csv('Data/auto.csv')
df = df[df.horsepower != '?'].astype({'horsepower': 'int32'})

#%%
print(df.columns)

#%%
#sns.pairplot(auto, diag_kws={'color':'red'})
g = sns.pairplot(df, diag_kind="kde", diag_kws={'color':'red'})
g.map_lower(sns.kdeplot, levels=4, color=".2")
plt.savefig('Model/auto.png')
plt.show()

#%%
sns.pairplot(data=df,
                  y_vars=['mpg'],
                  x_vars=[x for x in df.columns if x not in ['mpg', 'name']], 
                  height=4)
plt.savefig('Model/auto_mpg.png')
plt.show()

#%%
for i, name in enumerate([x for x in df.columns if x not in ['mpg', 'name']]):
  print(i, name)
  plt.subplot(2, 4, i+1)
  plt.plot(df[name].values, df['mpg'].values, '.', color='r', markersize=2)
  plt.ylabel('mpg')
  plt.xlabel(name)
#plt.subplots_adjust(hspace=0.3, wspace=0.8)
plt.savefig("Model/auto_mpg2.png", dpi=200)
plt.show()

#%%
plt.plot(df['horsepower'].values, df['mpg'].values,'ro')
plt.ylabel('mpg'); plt.xlabel('horsepower')
plt.show()

#%%
formula = "mpg ~ horsepower"
mod = smf.ols(formula, data=df)
res = mod.fit()

#%%
print(res.summary())
print(res.params)
print(res.tvalues)
# R2: 0.606

#%%
horsepower_range = np.arange(min(df['horsepower']), max(df['horsepower']), 1)
hp_values = res.params.Intercept + res.params.horsepower * horsepower_range

plt.plot(df['horsepower'].values, df['mpg'].values,'ro')
plt.plot(horsepower_range, hp_values)
plt.ylabel('mpg'); plt.xlabel('horsepower'); plt.title(f'mpg regressed on Horsepower')
plt.savefig("Model/auto_ols2.png", dpi=200)
plt.show()

#%%
#  res = OLS.from_formula(formula, data=df).fit()
plt.scatter(df['horsepower'].values, df['mpg'].values)
fig = sm.graphics.abline_plot(model_results=res, color='red', ax=plt.gca())
plt.savefig('Model/auto_ols.png')

#%%
formula = "mpg ~ horsepower + cylinders + displacement + weight + acceleration + model_year + origin"
#model = OLS.from_formula(formula, data=df).fit()
model = smf.ols(formula, data=df).fit()

#%%
print(model.summary())
print(model.params)
print(model.tvalues)
# R2: 0.821

#%% Cooks distance plot
fig = sm.graphics.influence_plot(model, criterion="cooks")
fig.tight_layout(pad=1.0)

#%% Components Residuals plot
fig = sm.graphics.plot_ccpr_grid(model)

#%% Fitted plot
fig = sm.graphics.plot_fit(model, "horsepower")

#%%
formula = "mpg ~ horsepower + displacement + weight + acceleration + model_year"
model = smf.ols(formula, data=df).fit()

#%%
print(model.summary())
print(model.params)
print(model.tvalues)
# R2: 0.809

#%%
formula = "mpg ~ weight + model_year"
model = smf.ols(formula, data=df).fit()

#%%
print(model.summary())
print(model.params)
print(model.tvalues)
# R2: 0.808

#%% Array
df_array = np.array(df.iloc[:, : -1])

#%% log
A = np.log(df_array)
df2 = pd.DataFrame(data=A, columns=list(df.columns)[:-1]) # [:-1] excludes the variable "name"

model = smf.ols(f'mpg ~ {" + ".join([f for f in list(df2.columns) if f not in ["mpg", "origin", "cylinders"]])}', data=df2).fit()
print(model.summary())
# R2: 0.889

#%% sqrt
A = np.sqrt(df_array)
df2 = pd.DataFrame(data=A, columns=list(df.columns)[:-1])

model = smf.ols(f'mpg ~ {" + ".join([f for f in list(df2.columns) if f not in ["mpg", "origin", "cylinders"]])}', data=df2).fit()
print(model.summary())
# R2: 0.861

#%% polinomial X^2
A = np.square(df_array)
df2 = pd.DataFrame(data=A, columns=list(df.columns)[:-1])

model = smf.ols(f'mpg ~ {" + ".join([f for f in list(df2.columns) if f not in ["mpg", "origin", "cylinders"]])}', data=df2).fit()
print(model.summary())
# R2: 0.671

#%% 1/X
A = np.reciprocal(df_array)
df2 = pd.DataFrame(data=A, columns=list(df.columns)[:-1])

model = smf.ols(f'mpg ~ {" + ".join([f for f in list(df2.columns) if f not in ["mpg", "origin", "cylinders"]])}', data=df2).fit()
print(model.summary())
# R2: 0.834

#%%





