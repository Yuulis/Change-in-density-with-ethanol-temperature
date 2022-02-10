import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

df = pd.read_csv('data.csv')
df.head(3)
x = df[['temperature']]
y = df[['density']]

model_lr = LinearRegression()
model_lr.fit(x, y)

plt.plot(x, y, 'o')
plt.plot(x, model_lr.predict(x), linestyle="solid")
plt.show()

print('x = temperature, y = density')
print('==============================')
print('y= %.5fx + %.5f' % (model_lr.coef_ , model_lr.intercept_))
print('決定係数R^2 (データの正確さ) : ', model_lr.score(x, y))
