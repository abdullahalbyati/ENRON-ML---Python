import sys
import pickle
import numpy as np
import pandas as pd

sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
import tester 
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
features_list = ['poi',
                'salary',
                'bonus', 
                'long_term_incentive', 
                'deferred_income', 
                'deferral_payments',
                'loan_advances', 
                'other',
                'expenses', 
                'director_fees',
                'total_payments',
                'exercised_stock_options',
                'restricted_stock',
                'restricted_stock_deferred',
                'total_stock_value',
                'to_messages',
                'from_messages',
                'from_this_person_to_poi',
                'from_poi_to_this_person']
### Load the dictionary containing the dataset
# with open("final_project_dataset.pkl", "r") as data_file:
#     data_dict = pickle.load(data_file)
# data_dict

with open('final_project_dataset_new.pkl', 'rb') as f:
    data_dict = pickle.load(f)
    
# Transform data from dictionary to the Pandas DataFrame
data_frame = pd.DataFrame.from_dict(data_dict, orient = 'index')

name_keys = sorted(list(data_dict.keys()))
rows = len(name_keys)
class_counts = data_frame["poi"].value_counts()
class_priors = class_counts / rows
print(class_counts)
print(class_priors)
#Order columns in DataFrame, exclude email column
data_frame = data_frame[features_list]
data_frame = data_frame.replace('NaN', np.nan)
data_frame.info()

#split of POI and non-POI in the dataset
poi_non_poi = data_frame.poi.value_counts()
poi_non_poi.index=['non-POI', 'POI']
print "POI / non-POI split"
poi_non_poi


# Counting NaN Values for each Column
nan_vals = data_frame.isnull().sum(axis = 0)
print(nan_vals)

# Counting NaN Values for each Column
nan_vals = data_frame.isnull().sum(axis = 0)
print(nan_vals)

from sklearn.preprocessing import Imputer

# Replacing 'NaN' in financial features with 0
data_frame.iloc[:,:15] = data_frame.iloc[:,:15].fillna(0)

email_features = ['to_messages', 'from_messages', 'from_this_person_to_poi', 'from_poi_to_this_person']

imp = Imputer(missing_values='NaN', strategy='median', axis=0)

#impute missing values of email features 
data_frame.loc[data_frame[data_frame.poi == 1].index,email_features] = imp.fit_transform(data_frame[email_features][data_frame.poi == 1])
data_frame.loc[data_frame[data_frame.poi == 0].index,email_features] = imp.fit_transform(data_frame[email_features][data_frame.poi == 0])

### Task 3: Create new feature(s)
#Create new feature(s)
data_frame["fraction_from_poi"] = data_frame["from_poi_to_this_person"].\
divide(data_frame["to_messages"], fill_value = 0)

data_frame["fraction_to_poi"] = data_frame["from_this_person_to_poi"].\
divide(data_frame["from_messages"], fill_value = 0)

data_frame["fraction_from_poi"] = data_frame["fraction_from_poi"].fillna(0.0)
data_frame["fraction_to_poi"] = data_frame["fraction_to_poi"].fillna(0.0)

### Store to my_dataset for easy export below.
my_dataset = data_frame.to_dict('index')

nan_vals = data_frame.isnull().sum(axis = 0)
print(nan_vals)

#Decision tree using features with non-null importance
clf = DecisionTreeClassifier(random_state = 75)
clf.fit(data_frame.iloc[:,1:], data_frame.iloc[:,:1])

# show the features with non null importance, sorted and create features_list of features for the model
features_importance = []
for i in range(len(clf.feature_importances_)):
    if clf.feature_importances_[i] > 0:
        features_importance.append([data_frame.columns[i+1], clf.feature_importances_[i]])
features_importance.sort(key=lambda x: x[1], reverse = True)
for f_i in features_importance:
    print f_i
features_list = [x[0] for x in features_importance]
features_list.insert(0, 'poi')


# Train Test Split
poi_labels = data_frame.pop('poi')
x_train, x_test, y_train, y_test = train_test_split(data_frame, poi_labels, random_state = 100)
clf = GaussianNB()
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


'''
Source:https://towardsdatascience.com/metrics-to-evaluate-your-machine-learning-algorithm-f10ba6e38234
Classification Accuracy is what we usually mean, when we use the term accuracy.
It is the ratio of number of correct predictions to the total number of input samples.

Precision : It is the number of correct positive results divided by the number of positive results
predicted by the classifier.

Recall : It is the number of correct positive results divided by the number of all relevant samples 
(all samples that should have been identified as positive).

'''
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

dump_classifier_and_data(clf, my_dataset, features_list)