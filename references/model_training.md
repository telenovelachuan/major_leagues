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
How can I miss NN :-)
Uses keras Tensorflow API to build a Sequential neural network that outputs both the score values for regression.
- model architecture: I build an NN with one input layer, 3 dense layers(units as hyperparameter) and one output layer(more layers slowed down training on my macbook...)
- hyperparameters: I uses GridSearchCV by sklearn to grid search the best hyperparameters for keras model(used KerasRegressor for model wrapper). The best hyperparameters are:
	* activation: relu (out of relu/sigmoid/linear)
	For hidden layers, grid search chooses relu. For output layer, I choose relu because scores are never negative, and relu does not limit any score loof, such as sigmoid does.
	* optimizer: adam (out of sgd/adam/nadam)
	* units: 100 (out of 50/100/200)
	more units per layer increases learning ability, but slows down converging.
	* initialization: lecun_normal (out of lecun_normal/he_normal/lecun_uniform)
	* loss: I used MSE for regression tasks.
	* metrics: instead of MSE, I customized a function to calculate r2 score as metrics.
- 400 epoch's training could approach the r2 score of around 0.78. I plan to have a try on GCP so that the model could fit more, and dropout layers can be added to prevent overfitting.
The trained models are used to predict real football match scores. I picked out 5 games in both testing set and last weeks' live football games of European major leagues.
	* On testing set
	
	| Date | Team1 | Team2 | score1 by model | score2 by model | score1 in real | score2 in real |
	| --- | --- | --- | --- | --- | --- | --- |
	| 2016-08-20 | Burnley | Liverpool | 1.94 | -0.04 | 2 | 0 |
	| 2002-04-17 | Barcelona | Athletic Bilbao | 2.86 | -0.03 | 3 | 0 |
	| 2017-08-19 | Swansea City | Manchester United | 0.02 | 3.6 | 0 | 4 |
	| 2017-12-23 | Genoa | Benevento | 1.04 | -0.02 | 1 | 0 |
	| 2018-01-31 | Tottenham Hotspur | Manchester United | 1.86 | -0.03 | 2 | 0 |
	
	* On live football games last week(weekends of 2019-10-19 and 2019-10-26)
	
	| Date | Team1 | Team2 | score1 by model | score2 by model | score1 in real | score2 in real |
	| --- | --- | --- | --- | --- | --- | --- |
	| 2019-10-26 | Bayern Munich | FC Union Berlin | 2.98 | 0.99 | 2 | 1 |
	| 2019-10-18 | Eintracht Frankfurt | Bayer Leverkusen | 0.77 | 1.75 | 3 | 0 |
	| 2019-10-19 | Everton | West Ham United | 1.03 | 0.98 | 2 | 0 |
	| 2019-10-27 | Liverpool | Tottenham Hotspur | 3.57 | 0.9 | 2 | 1 |
	| 2019-10-20 | Parma | Genoa | 3.16 | 0.99 | 5 | 1 |
	
The model works quite well on testing set! And also not so bad on the real football games last week, at least on predicting win/lose. Well generally speaking, predicting unusual match scores are difficult, which may require far more features, or is even unpredictable.
