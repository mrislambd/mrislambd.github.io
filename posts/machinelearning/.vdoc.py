# type: ignore
# flake8: noqa
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#| code-fold: false
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression
#
#
#
#
#
#
#
#| code-fold: false
X=np.random.random(100)
y=3*X+2+np.random.randn(100)
#
#
#
#
#
#| code-fold: false
plt.figure(figsize=(9,6))
plt.scatter(X,y)
plt.xlabel('$X$')
plt.ylabel('y')
plt.gca().set_facecolor('#f4f4f4') 
plt.gcf().patch.set_facecolor('#f4f4f4')
plt.show()
#
#
#
#
#
#| code-fold: false
slr=LinearRegression()
#
#
#
#
#
#| code-fold: false
#| code-overflow: scroll
X
#
#
#
#
#
#| code-fold: false
#| code-overflow: wrap
X=X.reshape(-1,1)
X[:10]
#
#
#
#
#| code-fold: false
slr.fit(X,y)
slr.predict([[2],[3]])
#
#
#
#
#
#| code-fold: false
intercept = round(slr.intercept_,4)
slope = slr.coef_
#
#
#
#
#
#| code-fold: false
plt.figure(figsize=(9,6))
plt.scatter(X,y, alpha=0.5,label="Sample Data")
plt.plot(np.linspace(0,1,100),
    slr.predict(np.linspace(0,1,100).reshape(-1,1)),
    'k',
    label='Model $\hat{f}$'
)
plt.plot(np.linspace(0,100,1),
    2*np.linspace(0,100,1)+0.5,
    'r--',
    label='$f$'
)
plt.legend(fontsize=10)
plt.xlabel('$X$')
plt.ylabel('y')
plt.gca().set_facecolor('#f4f4f4') 
plt.gcf().patch.set_facecolor('#f4f4f4')
plt.show()
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
