A data science project on the soccer_spi dataset to explore the features of European football data and build regression models to predict the scores for each team.

# Feature explorative ideas
1. append club SPI global rankings and league Intl SPI rankings into the match dataset for every match as new features
2. explore recent form for clubs(streaks):
        a) games won in row before the current game
        b) games lost in row before the current game
        c) games won during last 5 games
        d) games won during last 10 games
        e) games lost during last 5 games
        f) games lost during last 10 games
        g) games drawn during last 5 games
        h) games drawn during last 10 games
3. explore recent scores for teams:
        a) goals scored during last 5 matches
        b) goals conceded during last 5 matches
        c) goals scored during last 10 matches
        d) goals conceded during last 10 matches
        e) goal difference for last 5 matches
        f) goal difference for last 10 matches
4. explore generic metrics on teams:
        a) ability to handle the easy: goal scored during last 5 games against weak teams
        b) ability to fight the giants: goal scored during last 5 games again strong teams
        c) ability to keep focus: goals conceded during last 5 games against weak teams
        d) ability to stand firm: goals conceded during last 5 games against strong teams
(*weak teams are referring to teams with the lowest 25% spi score for each league, and strong teams vice versa)


# Data preprocessing
1. use Python to implement the logic of finding past N match results and goal scorings, and winning/losing streaks for teams.
2. use Excel pivotable to append club SPI global rankings and league Intl SPI rankings into spi matches dataset.
3. drop rows with empty scores for training data
4. one-hot encode column "league", "team1" and "team2"
5. for other columns, use SimpleImputer to fill in mean value
6. use StandardScaler for standardization
7. use train_test_split to split final dataset into training and testing data

# Build regression models
Tried 4 regression models on the training dataset, with k-fold cross validation(cv set to 3 here).
models used:

1. Linear Regression
Utilized the linear regression with stochastic gradient descent methodology SGDRegressor from sklearn, with tolerance set to 0.001 and starting learning rate to be 0.1.
The model converged quickly but, fitted poorly since there's hardly any direct linear correlation between team score and other feature combinations.

2. Polynomial Regression
Similarly, I used the PolynomialFeatures in sklearn to add a polynomial kernal into linear regression to implement a polynomial regression. Polynomial degree started from 2 which means it's quadratic.
Unfortunately the model was hard to converge even before larger degrees were tried.

3. Random Forest
Although famous for classification, random forest is a good choice here for regression task. I built two random forests using sklearn RandomForestRegressor with 40 trees, to fit the data for both score1 and score2.
Both forests converged in an acceptable time and fitted the data much better than linear models(with r2 score between 0.56 ~ 0.59).

4. Neural Network
Uses keras Tensorflow API to build a Sequential neural network that outputs both the score values for regression.
- model architecture: I build an NN with one input layer, 3 dense layers(units as hyperparameter) and one output layer(more layers slowed down training on my macbook...)
- hyperparameters: I uses GridSearchCV by sklearn to grid search the best hyperparameters for keras model(used KerasRegressor for model wrapper). The best hyperparameters are:
	(1) activation: relu (out of relu/sigmoid/linear)
	For hidden layers, grid search chooses relu. For output layer, I choose relu because scores are never negative, and relu does not limit any score loof, such as sigmoid does.
	(2) optimizer: adam (out of sgd/adam/nadam)
	(3) units: 100 (out of 50/100/200)
	more units per layer increases learning ability, but slows down converging.
	(4) initialization: lecun_normal (out of lecun_normal/he_normal/lecun_uniform)
	(5) loss: I used MSE for regression tasks.
	(6) metrics: instead of MSE, I customized a function to calculate r2 score as metrics.
400 epoch's training could approach the r2 score of around 0.78. I plan to have a try on GCP so that the model could fit more, and dropout layers can be added to prevent overfitting.
The trained models are used to predict real football match scores. I picked out 5 games in both testing set and last weeks' live football games of European major leagues.
	(1) On testing set.
	| Date | Team1 | Team2 | score1 by model | score2 by model | score1 in real | score2 in real |
	| --- | --- | --- | --- | --- | --- | --- |
	| 2016-08-20 | Burnley | Liverpool | 1.94 | -0.04 | 2 | 0 |
	| 2002-04-17 | Barcelona | Athletic Bilbao | 2.86 | -0.03 | 3 | 0 |
	| 2017-08-19 | Swansea City | Manchester United | 0.02 | 3.6 | 0 | 4 |
	| 2017-12-23 | Genoa | Benevento | 1.04 | -0.02 | 1 | 0 |
	| 2018-01-31 | Tottenham Hotspur | Manchester United | 1.86 | -0.03 | 2 | 0 |
