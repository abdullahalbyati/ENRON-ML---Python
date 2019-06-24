#!/usr/bin/python

import sys
import pickle
import numpy as np
import pandas as pd

sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

import sklearn
import scipy

from scipy import stats
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score, train_test_split

from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

print("I have used the following Versions:")
print("Numpy Version:", np.__version__)
print("Pandas Version:", pd.__version__)
print("SkLearn Version:", sklearn.__version__)
print("Scipy Version:", scipy.__version__)

### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi', 'salary', 'to_messages', 'total_payments', 'bonus', 'total_stock_value', 'expenses', 'from_poi_to_this_person', 'exercised_stock_options', 'from_messages', 'other', 'from_this_person_to_poi', 'shared_receipt_with_poi', 'restricted_stock']

### Load the dictionary containing the dataset
# with open("final_project_dataset.pkl", "r") as data_file:
#     data_dict = pickle.load(data_file)
# data_dict

with open('final_project_dataset_new.pkl', 'rb') as f:
    data_dict = pickle.load(f)
    
# Converting Dictionary to Numpy Array

name_keys = sorted(list(data_dict.keys()))
rows = len(name_keys)

data_keys = list(data_dict[name_keys[0]].keys())
cols = len(data_keys) + 1

print(rows, cols)

dataset = {}
for c in range(cols):
    col = []
    for r in range(rows):
        if c == 0:
            col.append(name_keys[r])
        else:
            value = data_dict[name_keys[r]][data_keys[c - 1]]
            if value == "NaN":
                col.append(np.nan)
            else:
                col.append(value)
    if c == 0:
        dataset["poi_name"] = col
    else:
        dataset[data_keys[c - 1]] = col



data_frame = pd.DataFrame(dataset)

data_frame.head(10)

# Dataset Summary
class_counts = data_frame["poi"].value_counts()
class_priors = class_counts / rows
print(class_counts)
print(class_priors)
data_frame.info()
# Removing unrequired columns/features
poi_names = data_frame.pop('poi_name')
poi_labels = data_frame.pop('poi')
emails = data_frame.pop('email_address')

# Counting NaN Values for each Column
nan_vals = data_frame.isnull().sum(axis = 0)
print(nan_vals)

# Removing Columns with 50% or more Nan Values
nan_thresh = 0.5
nan_percents = np.array(nan_vals) / float(rows)
print(nan_percents)

required_features = list(np.array(data_frame.columns)[nan_thresh - nan_percents > 0])
data_frame = data_frame[required_features]
print(data_frame.columns)

# Replacing NaNs with the Medians of Respective Features
for df_col in data_frame.columns:
    data_frame[df_col].fillna(data_frame[df_col].median(), inplace=True)

print(data_frame.head(5))

z_scores = np.abs(stats.zscore(data_frame))
print(z_scores)

# Datapoints with 3 Std Devs more/less
threshold = 3
z_thresh = np.where(z_scores > 3)
z_thresh

outliers, outlier_counts = np.unique(z_thresh[0], return_counts=True)
outlier_thresh = cols / 2
outlier_indices = np.where(outlier_counts > outlier_thresh)
outlier_idx = -1
for oi in outlier_indices:
    outlier_idx = oi
data_frame.drop(outlier_idx)

### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

# Train Test Split
x_train, x_test, y_train, y_test = train_test_split(data_frame, poi_labels, random_state = 100)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.

clf = GaussianNB(class_priors)
clf.fit(x_train, y_train)
preds = clf.predict(x_test)
print("---------------------------------------------------")
print("Gaussian Naive Bayes Accuracy:", accuracy_score(y_test, preds))
print("Gaussian Naive Bayes CV Score:", cross_val_score(clf, data_frame, poi_labels, cv=5).mean())
print("Gaussian Naive Bayes Precision:", precision_score(y_test, preds, average="weighted"))
print("Gaussian Naive Bayes Recall:", recall_score(y_test, preds, average="weighted"))
print("Gaussian Naive Bayes F1-Score:", f1_score(y_test, preds, average="weighted"))

clf = KNeighborsClassifier(n_neighbors=7)
clf.fit(x_train, y_train)
preds = clf.predict(x_test)
print("---------------------------------------------------")
print("KNN Accuracy:", accuracy_score(y_test, preds))
print("KNN CV Score:", cross_val_score(clf, data_frame, poi_labels, cv=5).mean())
print("KNN Precision:", precision_score(y_test, preds, average="weighted"))
print("KNN Recall:", recall_score(y_test, preds, average="weighted"))
print("KNN F1-Score:", f1_score(y_test, preds, average="weighted"))

clf = LogisticRegression()
clf.fit(x_train, y_train)
preds = clf.predict(x_test)
print("---------------------------------------------------")
print("Logistic Regression Accuracy:", accuracy_score(y_test, preds))
print("Logistic Regression CV Score:", cross_val_score(clf, data_frame, poi_labels, cv=5).mean())
print("Logistic Regression Precision:", precision_score(y_test, preds, average="weighted"))
print("Logistic Regression Recall:", recall_score(y_test, preds, average="weighted"))
print("Logistic Regression F1-Score:", f1_score(y_test, preds, average="weighted"))

# clf = SVC(gamma='scale', degree=3)
# clf.fit(x_train, y_train, dtype=np.float)
# preds = clf.predict(x_test)
# print("---------------------------------------------------")
# print("SVM Accuracy:", accuracy_score(y_test, preds))
# print("SVM CV Score:", cross_val_score(clf, data_frame, poi_labels, cv=5).mean())
# print("SVM Precision:", precision_score(y_test, preds, average="weighted"))
# print("SVM Recall:", recall_score(y_test, preds, average="weighted"))
# print("SVM F1-Score:", f1_score(y_test, preds, average="weighted"))


clf = DecisionTreeClassifier(criterion = "entropy", random_state = 100, max_depth=5, min_samples_leaf=5)
clf.fit(x_train, y_train)
preds = clf.predict(x_test)
print("---------------------------------------------------")
print("Decision Tree Accuracy:", accuracy_score(y_test, preds))
print("Decision Tree CV Score:", cross_val_score(clf, data_frame, poi_labels, cv=5).mean())
print("Decision Tree Precision:", precision_score(y_test, preds, average="weighted"))
print("Decision Tree Recall:", recall_score(y_test, preds, average="weighted"))
print("Decision Tree F1-Score:", f1_score(y_test, preds, average="weighted"))

clf = RandomForestClassifier(n_estimators = 20, criterion = "entropy", random_state = 100, max_depth=3, min_samples_leaf=5)
clf.fit(x_train, y_train)
preds = clf.predict(x_test)
print("---------------------------------------------------")
print("Random Forrest Accuracy:", accuracy_score(y_test, preds))
print("Random Forrest CV Score:", cross_val_score(clf, data_frame, poi_labels, cv=5).mean())
print("Random Forrest Precision:", precision_score(y_test, preds, average="weighted"))
print("Random Forrest Recall:", recall_score(y_test, preds, average="weighted"))
print("Random Forrest F1-Score:", f1_score(y_test, preds, average="weighted"))

clf = AdaBoostClassifier(n_estimators = 20, random_state = 100)
clf.fit(x_train, y_train)
preds = clf.predict(x_test)
print("---------------------------------------------------")
print("AdaBoost Accuracy:", accuracy_score(y_test, preds))
print("AdaBoost CV Score:", cross_val_score(clf, data_frame, poi_labels, cv=5).mean())
print("AdaBoost Precision:", precision_score(y_test, preds, average="weighted"))
print("AdaBoost Recall:", recall_score(y_test, preds, average="weighted"))
print("AdaBoost F1-Score:", f1_score(y_test, preds, average="weighted"))

clf = GradientBoostingClassifier(n_estimators = 20)
clf.fit(x_train, y_train)
preds = clf.predict(x_test)
print("---------------------------------------------------")
print("Gradient Boosting Accuracy:", accuracy_score(y_test, preds))
print("Gradient Boosting CV Score:", cross_val_score(clf, data_frame, poi_labels, cv=5).mean())
print("Gradient Boosting Precision:", precision_score(y_test, preds, average="weighted"))
print("Gradient Boosting Recall:", recall_score(y_test, preds, average="weighted"))
print("Gradient Boosting F1-Score:", f1_score(y_test, preds, average="weighted"))

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# I am choosing the AdaBoost Classifier

# Fine Tuning
num_estimators = range(1, 31, 2)
learning_rates = list(np.linspace(0.1, 2, 20, dtype=np.float32))

scores = np.zeros((len(num_estimators), len(learning_rates)))

for ni, ne in enumerate(num_estimators):
    for li, lr in enumerate(learning_rates):
        # Classifier with the Changing Parameters
        clf = AdaBoostClassifier(n_estimators = ne, learning_rate = lr, random_state = 100)
        clf.fit(x_train, y_train)
        preds = clf.predict(x_test)
        
        # Computing Cusomized Score
        score = accuracy_score(y_test, preds) + \
                cross_val_score(clf, data_frame, poi_labels, cv=5).mean() + \
                precision_score(y_test, preds, average="weighted") + \
                recall_score(y_test, preds, average="weighted") + \
                f1_score(y_test, preds, average="weighted")
        scores[ni][li] = score / 5.0

max_score = np.max(scores)
max_idxs = np.where(scores == max_score)
best_ne = num_estimators[max_idxs[0][0]]
best_lr = learning_rates[max_idxs[1][0]]
print("Maximum Score =", max_score, " with n_estimators =", best_ne, "and learning rate =", best_lr)


clf = AdaBoostClassifier(n_estimators = best_ne, learning_rate=best_lr, random_state = 100)
clf.fit(x_train, y_train)
preds = clf.predict(x_test)
print("AdaBoost Accuracy:", accuracy_score(y_test, preds))
print("AdaBoost CV Score:", cross_val_score(clf, data_frame, poi_labels, cv=5).mean())
print("AdaBoost Precision:", precision_score(y_test, preds, average="weighted"))
print("AdaBoost Recall:", recall_score(y_test, preds, average="weighted"))
print("AdaBoost F1-Score:", f1_score(y_test, preds, average="weighted"))

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)