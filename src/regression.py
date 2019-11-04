import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from joblib import dump
from keras.models import load_model, model_from_json


final_input = pd.read_csv("../data/processed/final.csv")

'''
Some data cleaning
'''
final_input.dropna(subset=['score1', 'score2'], inplace=True)
columns_to_dummy = ['league', 'team1', 'team2']
df = pd.get_dummies(final_input, columns=columns_to_dummy)

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('std_scaler', StandardScaler()),
])
df_columns = df.columns.to_list()
df_columns.remove('score1')
df_columns.remove('score2')
df[df_columns] = num_pipeline.fit_transform(df[df_columns])
print "processed data:{}".format(df)
df.to_csv("../data/processed/training.csv")


def prepare_data_for_predict_testset():
    # preparing data for prediction
    predict_case_indexes = [29, 1175, 3282, 6888, 7505]
    col_to_remove = ["score1", "score2"]
    test_set = pd.read_csv('../data/processed/test_set.csv')
    # test_row_indexes = [test_set.iloc[idx]['Unnamed: 0'] for idx in predict_case_indexes]
    original_df = pd.read_csv("../data/processed/processed.csv")
    training_df = pd.read_csv("../data/processed/training.csv")
    row_ndarray = training_df[training_df.index.isin(predict_case_indexes)]
    rows_display = original_df[original_df.index.isin(predict_case_indexes)]
    row_ndarray.drop(columns=col_to_remove, axis=1, inplace=True)
    print "cases to predict:\n{}".format([row for _, row in rows_display[['date', 'team1', 'team2']].iterrows()])
    rows_ndarray = row_ndarray.to_numpy()
    return rows_ndarray


def prepare_data_for_predict_set():
    processed = pd.read_csv("../data/processed/processed.csv")
    processed.drop(columns=['date', 'score1', 'score2'], axis=1, inplace=True)
    columns_to_dummy = ['league', 'team1', 'team2']
    df = pd.get_dummies(processed, columns=columns_to_dummy)

    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="constant", fill_value=0)),
        ('std_scaler', StandardScaler()),
    ])
    df_columns = df.columns.to_list()

    df[df_columns] = num_pipeline.fit_transform(df[df_columns])

    # print "processed predict set:{}".format(df)
    # df.to_csv("../data/processed/predict_set_processed.csv")
    #df = df.iloc[25831:].drop(columns=[['score1', 'score2']], axis=1, inplace=True)
    predict_case_indexes = [37, 68, 584, 432, 247]
    original_df = processed
    rows = [df[df.index.isin([idx])] for idx in predict_case_indexes]
    rows_display = [original_df[original_df.index.isin([idx + 25831])] for idx in predict_case_indexes]
    print "cases to predict:\n{}".format([row[['year', 'month', 'day', 'team1', 'team2']] for row in rows_display])
    rows_ndarray = [row.to_numpy() for row in rows]
    print "rows_ndarray[0]:{}".format(rows_ndarray[0])
    print "shape:{}".format(len(rows_ndarray[0]))
    return rows_ndarray


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
from sklearn.linear_model import SGDRegressor
print "fitting SGD regressor..."
sgd_reg = SGDRegressor(max_iter=1000, tol=1e-3, penalty=None, eta0=0.1)
train_n_evaluate(sgd_reg, "linear_regression", cv=5)
#  scores:[-8.32199806e+24 -5.97997859e+24 -8.19218360e+24]

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

'''
Random Forest
'''
from sklearn.ensemble import RandomForestRegressor
rf_reg = RandomForestRegressor(n_estimators=40, verbose=1, criterion='mse')
train_n_evaluate(rf_reg, "random_forest", cv=5)
# cross val scores for score1:[0.54887999 0.55289514 0.58159345 0.57460962 0.55528305 0.57258883 0.55020498 0.56175289 0.57008246 0.5858311 ]
# cross val scores for score2:[0.55198016 0.5520537  0.5223262  0.56590597 0.53978307 0.57134336 0.54745679 0.56139037 0.53256986 0.56298315 ]
# 40 estimators with MSE: [0.56336969 0.54826698 0.54049093 0.56902241 0.54205725 0.58976756]


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
        layer = Dense(units=neuron, activation=activation, kernel_initializer=init)(layer)

    score_output = Dense(units=1, activation='relu', name='score_output')(layer)
    nn_model = Model(inputs=input_layer, outputs=[score_output])

    # nn_model.compile(optimizer='adam',
    #                  loss={'score1_output': 'mse', 'score2_output': 'mse'},
    #                  metrics={'score1_output': coeff_determination, 'score2_output': coeff_determination})
    nn_model.compile(optimizer=optimizer,
                     loss='mse',
                     metrics={'score_output': coeff_determination})
    return nn_model
    # print nn_model.summary()


'''
Run Grid Search to find best hyperparameters for score1
'''
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


def fit_model(score):
    print "begin to fit model for {}}...".format(score)
    # train/test split
    df = pd.read_csv("../data/processed/training.csv")
    train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
    col_to_remove = ["score1", "score2"]
    X_train = train_set.drop(columns=col_to_remove, axis=1)
    X_test = test_set.drop(columns=col_to_remove, axis=1)
    y_train1, y_train2 = train_set["score1"], train_set["score2"]
    y_test1, y_test2 = test_set["score1"], test_set["score2"]
    print "train/test split done."

    nn_model = create_keras_model(activation='relu', optimizer='adam', neuron=100, init='lecun_normal')
    print "Keras model constructed"
    nn_model.fit(X_train, y_train1 if score == 'score1' else y_train2, epochs=50, verbose=1)
    save_keras_model(nn_model, "NN_{}".format(score))
    scores = nn_model.evaluate(X_test, y_test1 if score == 'score1' else y_test2, verbose=1)
    print "scores: {}".format(scores)


def load_model_n_predict(score, row_value_array):
    print "load model and predict {}...".format(score)
    with open('../models/NN_{}.json'.format(score)) as f:
        model = model_from_json(f.read())
    model.load_weights('../models/NN_{}.h5'.format(score))
    print "loading finished"
    prediction = model.predict(row_value_array)
    print "prediction for {} :{}".format(score, prediction)


rows = prepare_data_for_predict_set()
for row in rows:
    load_model_n_predict('score1', row)
    load_model_n_predict('score2', row)




