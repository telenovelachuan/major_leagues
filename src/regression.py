import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline


# final_input = pd.read_csv("../data/processed/final.csv")
#
# '''
# Some data cleaning
# '''
# final_input.dropna(subset=['score1', 'score2'], inplace=True)
# columns_to_dummy = ['league', 'team1', 'team2']
# df = pd.get_dummies(final_input, columns=columns_to_dummy)
#
# num_pipeline = Pipeline([
#     ('imputer', SimpleImputer(strategy="median")),
#     ('std_scaler', StandardScaler()),
# ])
# df_columns = df.columns.to_list()
# df[df_columns] = num_pipeline.fit_transform(df[df_columns])
# print "processed data:{}".format(df)
# df.to_csv("../data/processed/training.csv")

df = pd.read_csv("../data/processed/training.csv")

# train/test split
train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
col_to_remove = ["score1", "score2"]  # for one hot encoded
X_train = train_set.drop(columns=col_to_remove, axis=1)
X_test = test_set.drop(columns=col_to_remove, axis=1)
y_train1, y_train2 = train_set["score1"], train_set["score2"]
y_test1, y_test2 = test_set["score1"], test_set["score2"]
print "train/test split done."


'''
Linear Regression
'''
# from sklearn.linear_model import SGDRegressor
# print "fitting SGD regressor..."
# sgd_reg = SGDRegressor(max_iter=1000, tol=1e-3, penalty=None, eta0=0.1)
# sgd_reg.fit(X_train, y_train1)
# print "SGD regressor trained."
# print "SGD intercept: {}, SGD coef: {}".format(sgd_reg.intercept_, sgd_reg.coef_)
# print "doing cross_val_score..."
# scores = cross_val_score(sgd_reg, X_train, y_train1)
# print "scores:{}".format(scores)
# #  scores:[-8.32199806e+24 -5.97997859e+24 -8.19218360e+24]

'''
Polynomial regression
'''
from sklearn.preprocessing import PolynomialFeatures
print "fitting polynomial regressor..."
poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X_train)
lin_reg = LinearRegression()
lin_reg.fit(X_poly, y_train1)
print "polynomial regressor trained."
scores = cross_val_score(lin_reg, X_poly, y_train1)
print "scores:{}".format(scores)



