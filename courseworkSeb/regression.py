"""
Header



"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import *
from sklearn.linear_model import *
from sklearn.pipeline import *
from sklearn.metrics import *
from sklearn.model_selection import *
from sklearn.feature_selection import *

########################################################################################
# Global Variables
fontsize = 20


########################################################################################
# Functions

def data_sorting(
        data,
        target_title,
        data_percentage,
        print_data):

    train, test = train_test_split(data, test_size=data_percentage)  # using data_percentage% of our data.
    train_target = train[target_title]
    test_target = test[target_title]
    train = train.drop(target_title, axis=1)
    test = test.drop(target_title, axis=1)
    if not print_data:
        return train, train_target, test, test_target
    else:
        return print(train, train_target, test, test_target)


# def df_to_sorted_array(
#         data
# ):
#     # sorted_indices = np.argsort(X)
#     # X = np.array(X)
#     # X = np.reshape(X, (-1, 1))
#     # X = X[sorted_indices]
#     sorted_indices = np.argsort(data)
#     arr = np.array(data).tolist()
#     arr = np.reshape(arr, (-1, 1))
#     arr = arr[sorted_indices]
#     return arr

def plot_learning_curve(
        estimator,
        title,
        X,
        y,
        axes=None,
        ylim=None,
        cv=None,
        n_jobs=None,
        train_sizes=np.linspace(0.1, 1.0, 5),
):
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(20, 5))

    axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(
        estimator,
        X,
        y,
        cv=cv,
        n_jobs=n_jobs,
        train_sizes=train_sizes,
        return_times=True,
    )
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(
        train_sizes,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.1,
        color="r",
    )
    axes[0].fill_between(
        train_sizes,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.1,
        color="g",
    )
    axes[0].plot(
        train_sizes, train_scores_mean, "o-", color="r", label="Training score"
    )
    axes[0].plot(
        train_sizes, test_scores_mean, "o-", color="g", label="Cross-validation score"
    )
    axes[0].legend(loc="best")

    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, "o-")
    axes[1].fill_between(
        train_sizes,
        fit_times_mean - fit_times_std,
        fit_times_mean + fit_times_std,
        alpha=0.1,
    )
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("fit_times")
    axes[1].set_title("Scalability of the model")

    # Plot fit_time vs score
    fit_time_argsort = fit_times_mean.argsort()
    fit_time_sorted = fit_times_mean[fit_time_argsort]
    test_scores_mean_sorted = test_scores_mean[fit_time_argsort]
    test_scores_std_sorted = test_scores_std[fit_time_argsort]
    axes[2].grid()
    axes[2].plot(fit_time_sorted, test_scores_mean_sorted, "o-")
    axes[2].fill_between(
        fit_time_sorted,
        test_scores_mean_sorted - test_scores_std_sorted,
        test_scores_mean_sorted + test_scores_std_sorted,
        alpha=0.1,
    )
    axes[2].set_xlabel("fit_times")
    axes[2].set_ylabel("Score")
    axes[2].set_title("Performance of the model")
    return plt



























######################################################################
# Split & Shuffle Dataset
scaled_data = pd.read_csv('/Users/seb/PycharmProjects/ACS341/courseworkSeb/scaled_dataset.csv')
clean_data = pd.read_csv('/Users/seb/PycharmProjects/ACS341/courseworkSeb/clean_dataset.csv')
scaled_data = scaled_data.astype('float')

train, train_target, test, test_target = data_sorting(scaled_data, 'Failed_Yes', 0.2, False)

######################################################################
# https://github.com/ageron/handson-ml2/blob/master/04_training_linear_models.ipynb
# Regression Model
# Try mini-batch gradient descent.
# predict TORQUE from ROTATIONAL SPEED

y = train['Torque_Nm']
X = train['Rotational_speed_rpm']
y_test = test['Torque_Nm']
X_test = test['Rotational_speed_rpm']

# X = df_to_sorted_array(X)
# y = df_to_sorted_array(y)
# X_test = df_to_sorted_array(X_test)
# y_test = df_to_sorted_array(y_test)


##################################
# Regression setup

plt.figure(figsize=(9, 9))
sns.scatterplot(x=X, y=y, data=train)


# lesson: always turn into an array before fitting
sorted_indices = np.argsort(X)
sorted_indices_test = np.argsort(X_test)
X = np.array(X)
y = np.array(y)
X = np.reshape(X, (-1, 1))
y = np.reshape(y, (-1, 1))
X = X[sorted_indices]
y = y[sorted_indices]

X_test = np.array(X_test)
y_test = np.array(y_test)
X_test = np.reshape(X_test, (-1, 1))
y_test = np.reshape(y_test, (-1, 1))
X_test = X_test[sorted_indices_test]
y_test = y_test[sorted_indices_test]

# Linear Regression
lin_reg = LinearRegression()
lin_reg.fit(X, y)
y_predict_lin_reg = lin_reg.predict(X)
print("Cross validation score for linear regression using neg MSE = \n", cross_val_score(lin_reg, X_test, y_test, cv=5, scoring='neg_mean_squared_error'))

plt.plot(X, y_predict_lin_reg, c="limegreen")


# Polynomial Regression
degree = 3
poly = PolynomialFeatures(degree=degree, include_bias=False)

poly_features = poly.fit(X)
poly_features = poly.transform(X)
poly_regression = lin_reg.fit(poly_features, y)
y_predict_poly_reg = poly_regression.predict(poly_features)
print("Cross validation score for polynomial regression using neg MSE = \nc", cross_val_score(poly_regression, X_test, y_test, cv=5, scoring='neg_mean_squared_error'))

plt.plot(X, y_predict_poly_reg, c="red", linewidth=3)
plt.legend(['Raw Data', 'Linear Regression', 'Polynomial Regression (3rd order)'], prop={'size': 15})
plt.title("Polynomial regression with degree 1 and "+str(degree), fontsize=fontsize)
plt.xlabel("Rotational Speed (rpm)", fontsize=fontsize*2/3)
plt.ylabel("Torque (Nm)", fontsize=fontsize*2/3)
plt.savefig("/Users/seb/PycharmProjects/ACS341/courseworkSeb/graphs_outputted/regression_figure.png")

######################################################################
# Cross Validation

cv = ShuffleSplit(n_splits=150, test_size=0.2, random_state=0)

plot_learning_curve(poly_regression, 'Learning Curves With Polynomial of Degree: ' + str(degree), X, y, ylim=None, cv=cv, n_jobs=5)

plot_learning_curve(lin_reg, 'Learning Curves With Polynomial of Degree: '+str(1), X, y, ylim=None, cv=cv, n_jobs=5)


print("Poly Coeffs: " + str(poly_regression.coef_) +"\nPoly intercept: " + str(poly_regression.intercept_))

print("The R2 score for poly Regression is : " + str(r2_score(y, y_predict_poly_reg)*100) + "%")
print("The R2 score for linear Regression is : " + str(r2_score(y, y_predict_lin_reg)*100) + "%")

print("The MSE for the polynomial regression: " + str(mean_squared_error(y, y_predict_poly_reg)*100) + "%")
print("The MSE for the linear regression: " + str(mean_squared_error(y, y_predict_lin_reg)*100) + "%")

print("The MSE for the polynomial regression: " + str(mean_squared_error(y, y_predict_poly_reg)*100) + "%")
print("The MSE for the linear regression: " + str(mean_squared_error(y, y_predict_lin_reg)*100) + "%")

print("The MAE for the polynomial regression: " + str(mean_absolute_error(y, y_predict_poly_reg)*100) + "%")# Note this is abs error
print("The MAE for the linear regression: " + str(mean_absolute_error(y, y_predict_lin_reg)*100) +"%")# Note this is abs error

# print("The explained variance score for the polynomial regression: " + str(explained_variance_score(y, y_predict_poly_reg)*100) + "%")
# print("The explained variance score for the linear regression: " + str(explained_variance_score(y, y_predict_lin_reg)*100) + "%")

print("The max error is for the poly regression is: " + str(max_error(y, y_predict_poly_reg)*100) + "%")
print("The max error is for the poly linear is: " + str(max_error(y, y_predict_lin_reg)*100) + "%")


# print(poly_regression.score(X_test, y_test))
######################################################################
# Logistic Regression
# https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFE.html
# we use a cv = 5 fold (cv generator used is stratified K-folds)

logit_regression = LogisticRegressionCV(cv=5, random_state=0, max_iter=500).fit(train, np.ravel(train_target))
# np.ravel()  ravel is used to flatten the array

selector = RFE(logit_regression, n_features_to_select=len(test.columns), step=1)
selector = selector.fit(test, test_target)

score = logit_regression.score(test, test_target)
predictions = logit_regression.predict(test)

logit_confusion_matrix = confusion_matrix(test_target, predictions)
# print(logit_confusion_matrix)
plt.figure(figsize=(9, 9))
sns.heatmap(logit_confusion_matrix, annot=True, fmt=".3f", linewidths=.5, square=True, cmap='Blues_r')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
all_sample_title = 'Accuracy Score: {0}'.format(score)
plt.title(all_sample_title, size=15)

plt.figure()

logit_accuracy = accuracy_score(test_target, predictions)
print("The logit accuracy is given as: " + str(logit_accuracy*100)+"%")

logit_precision = precision_score(test_target, predictions)
print("The logit precision is given as: " + str(logit_precision*100)+"%")

logit_recall = recall_score(test_target, predictions)
print("The logit recall is given as: " + str(logit_recall*100)+"%")

logit_confusion_matrix = confusion_matrix(test_target, predictions)
print("The confusion matrix: ", logit_confusion_matrix)


###
# Metrics to be used to measure logistic regression against:
# Confusion matrix
# accuracy
# precision

######################################################################
# Show plots

plt.show()
