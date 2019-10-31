import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from joblib import dump


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


def train_n_evaluate(model, model_name, cv=3):
    print "training {} regressor... for score1...".format(model_name)
    model.fit(X_train, y_train1)
    print "{} regressor for score1 trained, saving model...".format(model_name)
    dump(model, '../models/{}_score1.joblib'.format(model_name))
    print "saving model finished, getting validation scores..."
    scores1 = cross_val_score(model, X_train, y_train1, cv=cv, scoring='r2')
    print "cross val scores for score1:{}".format(scores1)

    print "training {} regressor... for score2...".format(model_name)
    model.fit(X_train, y_train2)
    print "{} regressor for score2 trained, saving model...".format(model_name)
    dump(model, '../models/{}_score2.joblib'.format(model_name))
    print "saving model finished, getting validation scores..."
    scores2 = cross_val_score(model, X_train, y_train2, cv=cv, scoring='r2')
    print "cross val scores for score2:{}".format(scores2)

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
# from sklearn.preprocessing import PolynomialFeatures
# print "fitting polynomial regressor..."
# poly_features = PolynomialFeatures(degree=2, include_bias=False)
# X_poly = poly_features.fit_transform(X_train)
# lin_reg = LinearRegression()
# lin_reg.fit(X_poly, y_train1)
# print "polynomial regressor trained."
# scores = cross_val_score(lin_reg, X_poly, y_train1)
# print "scores:{}".format(scores)

'''
Random Forest
'''
from sklearn.ensemble import RandomForestRegressor
rf_reg = RandomForestRegressor(n_estimators=40)
train_n_evaluate(rf_reg, "random_forest", cv=10)
# cross val scores for score1:[0.54887999 0.55289514 0.58159345 0.57460962 0.55528305 0.57258883 0.55020498 0.56175289 0.57008246 0.5858311 ]
# cross val scores for score2:[0.55198016 0.5520537  0.5223262  0.56590597 0.53978307 0.57134336 0.54745679 0.56139037 0.53256986 0.56298315 ]


'''
Neural Network
'''


def save_keras_model(model, file_name):
    model_json = model.to_json()
    with open("../models/{}.json".format(file_name), "w") as json_file:
        json_file.write(model_json)
    model.save_weights("../models/{}.h5".format(file_name))
    print "Keras model saved to disk."


# import tensorflow as tf
# from keras.optimizers import SGD
#
# #X_train = X_train.as_matrix()
# #X_test = X_test.as_matrix()
# y_train = train_set[['score1', 'score2']].values
# y_test = test_set[['score1', 'score2']].values
# print "neural network"
# nn_reg = tf.keras.models.Sequential([
#         tf.keras.layers.Dense(30, activation="sigmoid", input_shape=X_train.shape[1:]),
#         #tf.keras.layers.Dense(30, input_dim=30, activation='relu'),
#         tf.keras.layers.Dense(2, activation="sigmoid")
# ])
# print "add dense layer finished"
# sgd = tf.keras.optimizers.SGD(learning_rate=0.02)
# nn_reg.compile(loss="mean_squared_error", optimizer=sgd, metrics=['mse'])
# print "compilation finished, training data now..."
# history = nn_reg.fit(X_train, y_train, verbose=1, epochs=60)
# print "training finished, saving model..."
# save_keras_model(nn_reg, "NN")
# print "saving model finished, start evaluating..."
# scores = nn_reg.evaluate(X_test, y_test, verbose=1)
# print "scores: {}".format(scores)




