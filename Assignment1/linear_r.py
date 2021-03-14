import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split


class HousePrice:
    def __init__(self):
        name= ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
        self.df = pd.read_csv('./data/housing.csv',delim_whitespace=True,names=name)
        print(self.df.head())
        print(f'${len(self.df)} lines loaded')

    def trim(self):
        self.df.columns = self.df.columns.str.lower().str.replace(' ', '_')
        string_columns = list(self.df.dtypes[self.df.dtypes == 'object'].index)
        for col in string_columns:
            self.df[col] = self.df[col].str.lower().str.replace(' ', '_')

    def rmse(self, y, y_pred):
        error = y_pred - y
        mse = (error ** 2).mean()
        return np.sqrt(mse)

def calculate_adjusted_rSquare(R, n, p):
    metric = (1 - R)*(n-1)/(n-p-1)
    return 1-metric

def main():
    obj = HousePrice()
    obj.trim()
    # Selecting Dataset for Linear Regression
    X_full = obj.df.iloc[:,:13]
    y_full = obj.df.iloc[:,13:]
    selected_feature = ['lstat']
    selected_feature_df = obj.df[selected_feature].copy()
    selected_feature_df['medv'] = obj.df['medv']
    selected_feature_df = selected_feature_df.sort_values(by=['lstat'])
    X = selected_feature_df.iloc[:,:1]
    y = selected_feature_df.iloc[:,1:]

    # Procedure for linear regression
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 30)
    linear_model = LinearRegression()
    linear_model.fit(X_train,y_train)
    y_pred= linear_model.predict(X)
    print("\nR2 Score: " , r2_score(y, y_pred))
    print("RMSE Score: ", obj.rmse(y, y_pred))
    plt.scatter(X, y,  color='black')
    plt.plot(X, y_pred, color='blue', linewidth=3)
    plt.show()

    # Procedure for Polynomial Regression for degree 2
    poly_reg = PolynomialFeatures(degree=2)
    X_poly = poly_reg.fit_transform(X)
    poly_reg.fit(X_poly,y)
    linear_model.fit(X_poly, y)
    y_pred=linear_model.predict(poly_reg.fit_transform(X))
    
    print("\nPolynomial RMSE Score: ", obj.rmse(y,y_pred))
    print("Polynomial R2 Score: " , r2_score(y, y_pred))
    plt.scatter(X, y,  color='black')
    plt.plot(X, y_pred, color='red')
    plt.show()

    # Procedure for Polynomial Regression for degree 20
    poly_reg = PolynomialFeatures(degree=20)
    X_poly = poly_reg.fit_transform(X)
    poly_reg.fit(X_poly,y)
    linear_model.fit(X_poly, y)
    y_pred=linear_model.predict(poly_reg.fit_transform(X))

    plt.scatter(X, y,  color='black')
    plt.plot(X, y_pred, color='red')
    plt.show()

    features_multi = ['indus','nox','rm','age','ptratio','lstat']
    b_features_multi = obj.df[features_multi].copy()
    b_features_multi['medv'] = y['medv']
    b_features_multi.head()
    X = b_features_multi.iloc[:,:len(features_multi)]
    y = b_features_multi.iloc[:,len(features_multi):]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 30)
    linear_model.fit(X_train,y_train)
    ymul_pred = linear_model.predict(X)
    R2 = r2_score(y, ymul_pred)
    print("\nMulti R2 Score = " , R2)
    print("Multi RMSE Score = " , obj.rmse(y, ymul_pred))
    Adj_R2 = calculate_adjusted_rSquare(R2,len(obj.df), len(features_multi))
    print("Adjusted R2:", Adj_R2)

main()