from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
import pandas as pd

dataset =pd.read_csv('exam_B_dataset.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,1].values


lin_reg = LinearRegression()
poly_reg = PolynomialFeatures(degree =5)  #Se coloca el grado de la ecuacion
X_poly = poly_reg.fit_transform(X) #genera la matriz A
poly_reg.fit(X_poly, y)  ##has el ajuste de la matriz A y el vector y
lin_reg.fit(X_poly, y)

plt.scatter(X,y)
plt.scatter(X, lin_reg.predict(poly_reg.fit_transform(X)))   #Imprimir en una linea recta
plt.show()
