import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from joblib import dump


#final_input = pd.read_csv("../data/processed/final.csv")

'''
Some data cleaning
'''
# final_input.dropna(subset=['score1', 'score2'], inplace=True)
# columns_to_dummy = ['league', 'team1', 'team2']
# df = pd.get_dummies(final_input, columns=columns_to_dummy)
#
# num_pipeline = Pipeline([
#     ('imputer', SimpleImputer(strategy="median")),
#     ('std_scaler', StandardScaler()),
# ])
# df_columns = df.columns.to_list()
# df_columns.remove('score1')
# df_columns.remove('score2')
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
# train_n_evaluate(sgd_reg, "linear_regression", cv=5)
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
# rf_reg = RandomForestRegressor(n_estimators=40, verbose=1, criterion='mse')
# train_n_evaluate(rf_reg, "random_forest", cv=5)
# # cross val scores for score1:[0.54887999 0.55289514 0.58159345 0.57460962 0.55528305 0.57258883 0.55020498 0.56175289 0.57008246 0.5858311 ]
# # cross val scores for score2:[0.55198016 0.5520537  0.5223262  0.56590597 0.53978307 0.57134336 0.54745679 0.56139037 0.53256986 0.56298315 ]
# # 40 estimators with MSE: [0.56336969 0.54826698 0.54049093 0.56902241 0.54205725 0.58976756]


'''
Neural Network
'''


def save_keras_model(model, file_name):
    print "saving keras model..."
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


from keras.layers import Input, Dense
from keras.models import Model
from keras import backend as K
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasRegressor


def coeff_determination(y_true, y_pred):
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return 1 - SS_res/(SS_tot + K.epsilon())


def create_keras_model(activation='selu', optimizer='adam', neuron=50, init='lecun_normal'):
    input_layer = Input(shape=X_train.shape[1:])
    # score1 output
    layer = Dense(units=neuron, activation=activation, kernel_initializer=init, input_shape=X_train.shape[1:])(input_layer)
    for i in range(3):
        layer = Dense(units=neuron, activation=activation, kernel_initializer=init, )(layer)

    # _ = Dense(units=200, activation='selu', kernel_initializer='lecun_normal', )(_)
    # _ = Dense(units=200, activation='selu', kernel_initializer='lecun_normal', )(_)
    # _ = Dense(units=200, activation='selu', kernel_initializer='lecun_normal', )(_)
    # _ = Dense(units=200, activation='selu', kernel_initializer='lecun_normal', )(_)
    # _ = Dense(units=200, activation='selu', kernel_initializer='lecun_normal', )(_)

    # _ = Dense(units=50, activation='sigmoid')(_)

    # _ = Dense(units=100, activation='relu')(_)
    score1_output = Dense(units=1, activation='linear', name='score1_output')(_)
    # score2 output
    #_ = Dense(units=100, activation='selu', kernel_initializer='lecun_normal', input_shape=X_train.shape[1:])(input_layer)
    #_ = Dense(units=100, activation='selu', kernel_initializer='lecun_normal', )(_)
    # _ = Dense(units=100, activation='relu')(_)
    #score2_output = Dense(units=1, activation='linear', name='score2_output')(_)
    # nn_model = Model(inputs=input_layer, outputs=[score1_output, score2_output]) # for multi-output
    nn_model = Model(inputs=input_layer, outputs=[score1_output])

    # nn_model.compile(optimizer='adam',
    #                  loss={'score1_output': 'mse', 'score2_output': 'mse'},
    #                  metrics={'score1_output': coeff_determination, 'score2_output': coeff_determination})
    nn_model.compile(optimizer=optimizer,
                     loss='mse',
                     metrics={'score1_output': coeff_determination})
    return nn_model
    # print nn_model.summary()


model = KerasRegressor(build_fn=create_keras_model, epochs=30)
activations = ['relu', 'sigmoid', 'linear']
# learn_rate = [0.001, 0.01, 0.1, 0.2, 0.3]
neurons = [50, 100, 200]
inits = ["lecun_normal", "he_normal", "lecun_uniform"]
optimizers = ["sgd", "adam", "nadam"]
param_grid = dict(activation=activations, optimizer=optimizers, neuron=neurons, init=inits)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)
grid_result = grid.fit(X_train, y_train1)
print "Best: {} if using {}".format(grid_result.best_score_, grid_result.best_params_)

#
# print "begin to fit model..."
# nn_model.fit(X_train, {"score1_output": y_train1, "score2_output": y_train2}, epochs=30, verbose=1)
# save_keras_model(nn_model, "NN")
# y_test = test_set[['score1', 'score2']].values
# scores = nn_model.evaluate(X_test, {"score1_output": y_test1, "score2_output": y_test2}, verbose=1)
# print "scores: {}".format(scores)



