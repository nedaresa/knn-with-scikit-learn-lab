
# KNN with scikit-learn 


To use scikit-learn's implementation of a KNN classifier on the classic Titanic dataset from Kaggle!

## Objectives


* Use KNN to make classification predictions on a real-world dataset
* Perform a parameter search for 'k' to optimize model performance
* Evaluate model performance and interpret results


Start by importing the dataset, stored in the `titanic.csv` file, and cleaning it.

Preprocessing the Data
- Remove unnecessary columns (PassengerId, Name, Ticket, and Cabin).
- Convert Sex to a binary encoding, where female is 0 and male is 1.
- Detect and deal with any null values in the dataset.
- For Age, replace null values with the median age for the dataset.
- One-Hot Encode categorical columns such as Embarked.
- Store the target column, Survived, in a separate variable and remove it from the DataFrame.

# Creating Training and Testing Sets

Split data into training and testing sets. 

* Import `train_test_split` from the `sklearn.model_selection` module
* Use `train_test_split` to split thr data into training and testing sets.

# Normalizing the Data
Normalize after splitting data into training and testing sets to avoid information "leaking" from test set into training set. Normalization (also sometimes called Standardization or Scaling) means making sure that all of data is represented at the same scale. 

Since KNN is a distance-based classifier, if data is in different scales, then larger scaled features have a larger impact on the distance between points.

- Import and instantiate a StandardScaler object.
- Use the scaler's .fit_transform() method to create a scaled version of our training dataset.
- Use the scaler's .transform() method to create a scaled version of our testing dataset.
- The result returned by the fit_transform and transform calls will be numpy arrays, not a pandas DataFrame. Create a new pandas DataFrame out of this object called scaled_df. To set the column names back to their original state, set the columns parameter to one_hot_df.columns.
- Print out the head of scaled_df to ensure everything worked correctly.


The scaler also scaled our binary/one-hot encoded columns, too! Although it doesn't look as pretty, this has no negative effect on the model. Each 1 and 0 have been replaced with corresponding decimal values, but each binary column still only contains 2 values, meaning the overall information content of each column has not changed.

# Fitting a KNN Model
Time to train a KNN classifier and validate its accuracy.


- Import KNeighborsClassifier from the sklearn.neighbors module.
- Instantiate a classifier. For now, you can just use the default parameters.
- Fit the classifier to the training data/labels
- Use the classifier to generate predictions on the test data. Store these predictions inside the variable test_preds.


Import all the necessary evaluation metrics from sklearn.metrics and complete the print_metrics() function so that it prints out Precision, Recall, Accuracy, and F1-Score when given a set of labels (the true values) and preds (the models predictions).

Finally, use print_metrics() to print out the evaluation metrics for the test predictions stored in test_preds, and the corresponding labels in y_test.


# Improving Model Performance
Try to find the optimal number of neighbors to use for the classifier: Iterate over multiple values of k and find the value of k that returns the best overall performance.
For each iteration:
- Create a new KNN classifier, and set the n_neighbors parameter to the current value for k, as determined by the loop.
- Fit this classifier to the training data.
- Generate predictions for X_test using the fitted classifier.
- Calculate the F1-score for these predictions.
- Compare this F1-score to best_score. If better, update best_score and best_k.
- Once all iterations are complete, print out the best value for k and the F1-score it achieved.