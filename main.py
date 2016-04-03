import pandas
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import KFold
from sklearn import cross_validation
import numpy as np

def clean_data(titanic, titanic_mean_age):
	titanic["Age"] = titanic["Age"].fillna(titanic_mean_age)
	titanic["Fare"] = titanic["Fare"].fillna(titanic_test["Fare"].median())
	titanic.loc[titanic["Sex"] == "male", "Sex"] = 0
	titanic.loc[titanic["Sex"] == "female", "Sex"] = 1

	# Find all the unique values for "Embarked".
	titanic["Embarked"] = titanic["Embarked"].fillna('S')
	titanic.loc[titanic["Embarked"] == "S", "Embarked"] = 0
	titanic.loc[titanic["Embarked"] == "C", "Embarked"] = 1
	titanic.loc[titanic["Embarked"] == "Q", "Embarked"] = 2
	#print(titanic["Embarked"].unique())
	return titanic

def linear_regression(predictors, titanic):
	# Initialize our algorithm class
	alg = LinearRegression()
	# Generate cross validation folds for the titanic dataset.  It return the row indices corresponding to train and test.
	# We set random_state to ensure we get the same splits every time we run this.
	kf = KFold(titanic.shape[0], n_folds=3, random_state=1)

	predictions = []
	for train, test in kf:
	    # The predictors we're using the train the algorithm.  Note how we only take the rows in the train folds.
	    train_predictors = (titanic[predictors].iloc[train,:])
	    # The target we're using to train the algorithm.
	    train_target = titanic["Survived"].iloc[train]
	    # Training the algorithm using the predictors and target.
	    alg.fit(train_predictors, train_target)
	    # We can now make predictions on the test fold
	    test_predictions = alg.predict(titanic[predictors].iloc[test,:])
	    predictions.append(test_predictions)
	# The predictions are in three separate numpy arrays.  Concatenate them into one.  
	# We concatenate them on axis 0, as they only have one axis.
	predictions = np.concatenate(predictions, axis=0)

	# Map predictions to outcomes (only possible outcomes are 1 and 0)
	predictions[predictions > .5] = 1
	predictions[predictions <=.5] = 0

	accuracy_list = [x == y for x, y in zip(titanic["Survived"], predictions)]

	num_acc = sum(accuracy_list)
	accuracy = sum(accuracy_list) / len(accuracy_list)
	accuracy = accuracy.item()

def logistic_regression(predictors, titanic, titanic_test):
	# Initialize our algorithm
	alg = LogisticRegression(random_state=1)
	# Compute the accuracy score for all the cross validation folds.  (much simpler than what we did before!)
	scores = cross_validation.cross_val_score(alg, titanic[predictors], titanic["Survived"], cv=3)
	# Take the mean of the scores (because we have one for each fold)
	print(scores.mean())

	# Train the algorithm using all the training data
	alg.fit(titanic[predictors], titanic["Survived"])

	# Make predictions using the test set.
	predictions = alg.predict(titanic_test[predictors])

def create_submission(titanic, predictions):
	# Create a new dataframe with only the columns Kaggle wants from the dataset.
	submission = pandas.DataFrame({
	        "PassengerId": titanic["PassengerId"],
	        "Survived": predictions
	    })

	submission.to_csv("data/kaggle.csv", index=False)

# ==================================================================================

titanic = pandas.read_csv("data/train.csv")
titanic_test = pandas.read_csv("data/test.csv")

titanic_mean_age = titanic["Age"].median()
titanic = clean_data(titanic, titanic_mean_age)
titanic_test = clean_data(titanic_test, titanic_mean_age)

# The columns we'll use to predict the target
predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
predictions = logistic_regression(predictors, titanic, titanic_test)
create_submission(titanic_test, predictions)
