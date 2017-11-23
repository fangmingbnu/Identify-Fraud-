#!/usr/bin/python
import sys
sys.path.append("../tools/")
import pickle
import matplotlib.pyplot as plt
import pandas
import numpy as np
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn import cross_validation
from sklearn.cross_validation import KFold
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline


### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
initial_features_list = ['poi', 'salary', 'to_messages', 'deferral_payments', 'total_payments',
                 'exercised_stock_options', 'bonus', 'restricted_stock', 'shared_receipt_with_poi',
                 'restricted_stock_deferred', 'total_stock_value', 'expenses', 'loan_advances', 'from_messages',
                 'other', 'from_this_person_to_poi', 'director_fees', 'deferred_income', 'long_term_incentive',
                 'email_address', 'from_poi_to_this_person']

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

# get a brief preview of data
print data_dict
print 'number of datapoint = ', len(data_dict)
print data_dict['METTS MARK'].keys()
print 'number of features = ', len(data_dict['METTS MARK'].keys())

df = pandas.DataFrame.from_dict(data_dict, orient='index', dtype=np.float)
print df.describe()

poi = 0
for i in data_dict:
    if data_dict[i]['poi']:
        poi += 1
print 'number of poi = ', poi


### Task 2: Remove outliers
# from the preview of data_dict, the data point --- "TOTAL" is the key which need to be removed. So I remove such data point from the data set.
data_dict.pop('TOTAL', 0)


### Create two new features including "fraction_from_poi_email" and "fraction_to_poi_email"
def dict_to_list(key,normalizer):
    new_list=[]
    for i in data_dict:
        ### Remove the data points which contain the "NA" features.
        if data_dict[i][key] == "NaN" or data_dict[i][normalizer] == "NaN":
            new_list.append("0")
        elif data_dict[i][key] >= 0:
            new_list.append(float(data_dict[i][key])/float(data_dict[i][normalizer]))
    return new_list

fraction_from_poi_email = dict_to_list("from_poi_to_this_person", "to_messages")
fraction_to_poi_email = dict_to_list("from_this_person_to_poi", "from_messages")

### Insert new features into data_dict

initial_features_list.append("fraction_from_poi_email")
initial_features_list.append("fraction_to_poi_email")

count = 0
for i in data_dict:
    data_dict[i]["fraction_from_poi_email"] = fraction_from_poi_email[count]
    data_dict[i]["fraction_to_poi_email"] = fraction_to_poi_email[count]
    count += 1
my_dataset = data_dict

### Task 3: Feature selection
### Store to my_dataset for easy export below.
all_features_list = initial_features_list
all_features_list.remove("email_address")
data = featureFormat(my_dataset, all_features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Combine pipeline , Kbest, and gridsearchCV method to decide how many features should be selected for this model

pipe = Pipeline([('select_k_best', SelectKBest()), ('classify', tree.DecisionTreeClassifier())])
N_FEATURES_OPTIONS = [i for i in range(1, 22)]
param_grid = [{'select_k_best': [SelectKBest(f_classif)], 'select_k_best__k': N_FEATURES_OPTIONS}]

## check the precision score
grid = GridSearchCV(pipe, cv=3, n_jobs=1, param_grid=param_grid, scoring='%s_macro' % 'precision')
grid.fit(features,  labels)
mean_precision_scores = np.array(grid.cv_results_['mean_test_score'])
## check the recall score
grid = GridSearchCV(pipe, cv=3, n_jobs=1, param_grid=param_grid, scoring='%s_macro' % 'precision')
grid.fit(features,  labels)
mean_recall_scores = np.array(grid.cv_results_['mean_test_score'])
## make a plot to show the relationship between performance vs features numbers

plt.figure(1)
plt.plot(N_FEATURES_OPTIONS, mean_precision_scores)
plt.plot(N_FEATURES_OPTIONS, mean_recall_scores)
plt.xlabel('feature number', fontsize=14)
plt.ylabel('evaluation', fontsize=14)
plt.legend(('precision', 'recall'), fontsize=14)
plt.show()

### From the plot, we need to choose the 7 features.
### Using Kbest method to choose the K best features.
test = SelectKBest(f_classif, 7)
test.fit_transform(features, labels)
indices = test.get_support(True)
features_list = []
print test.scores_
for i in indices:
    print all_features_list[i+1]
    features_list.append(all_features_list[i+1])
print features_list

### so I choose features 'salary', 'exercised_stock_options', 'bonus', 'total_stock_value', 'deferred_income', 'fraction_to_poi_email'.

### So the final features list is listed below:

features_list = ['poi', 'salary', 'exercised_stock_options', 'bonus', 'total_stock_value', 'deferred_income', 'long_term_incentive', 'fraction_to_poi_email']

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
# In this study, a two important metric to evaluate the performance of classifiers is precision and recall.
# precision is the fraction of relevant instances among the retrieved instances, while recall is the fraction of relevant instances
# that have been retrieved over the total amount of relevant instances.
# Besides that, the accuracy is also applied to evaluate the performance of classifiers.


features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(features, labels, test_size = 0.5, random_state=42)

# try the GaussianNB method at first
clf = GaussianNB()
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)

my_dataset = data_dict
print my_dataset
print 'naive bayes acc = ' , accuracy_score(pred, labels_test)
print 'naive bayes recall = ', recall_score(labels_test, pred)
print 'naive bayes precision = ', precision_score(labels_test, pred)

# try the DecisionTree method second
clf = tree.DecisionTreeClassifier()
clf = clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
print 'tree acc = ', accuracy_score(pred, labels_test)
print 'tree recall = ', recall_score(labels_test, pred)
print 'tree precision = ', precision_score(labels_test, pred)

# through the comparision, it is concluded that decision tree method is better.

data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### using cross validation (K-fold) to evaluate classifier

acclist = []
precisionlist = []
recalllist = []
kf = KFold(len(labels), 5)
for train_indices, test_indices in kf:
    # make train and test data
    features_train= [features[ii] for ii in train_indices]
    features_test= [features[ii] for ii in test_indices]
    labels_train=[labels[ii] for ii in train_indices]
    labels_test=[labels[ii] for ii in test_indices]
    # train data and perform prediction
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(features_train, labels_train)
    pred = clf.predict(features_test)
    # obtain accuracy, precision, and recall score
    acc = accuracy_score(pred, labels_test)
    pre = precision_score(labels_test, pred)
    rec = recall_score(labels_test, pred)
    # make a list to obtain an average
    acclist.append(acc)
    precisionlist.append(pre)
    recalllist.append(rec)
print 'KFold tree acc = ' ,(1.0*sum(acclist))/len(acclist)
print 'KFold tree precision = ' ,(1.0*sum(precisionlist))/len(precisionlist)
print 'KFold tree recall = ' ,(1.0*sum(recalllist))/len(recalllist)

### using another cross validation method (StratifiedKFold) to evaluate classifier
acclist = []
precisionlist = []
recalllist = []
skf = StratifiedKFold(n_splits=5)
for train_indices, test_indices in skf.split(features, labels):
    # make train and test data
    features_train= [features[ii] for ii in train_indices]
    features_test= [features[ii] for ii in test_indices]
    labels_train=[labels[ii] for ii in train_indices]
    labels_test=[labels[ii] for ii in test_indices]
    # train data and perform prediction
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(features_train, labels_train)
    pred = clf.predict(features_test)
    # obtain accuracy, precision, and recall score
    acc = accuracy_score(pred, labels_test)
    pre = precision_score(labels_test, pred)
    rec = recall_score(labels_test, pred)
    # make a list to obtain an average
    acclist.append(acc)
    precisionlist.append(pre)
    recalllist.append(rec)
print 'StratifiedKFold tree acc = ' ,(1.0*sum(acclist))/len(acclist)
print 'StratifiedKFold tree precision = ' ,(1.0*sum(precisionlist))/len(precisionlist)
print 'StratifiedKFold tree recall = ' ,(1.0*sum(recalllist))/len(recalllist)

# As we all know, in this dataset the most data point is non poi.
# Thus the StratifiedKFold method is much better, since the folds are made by preserving the percentage of samples for each class.
# So we chose StratifiedKFold method as the final evaluation method.

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!

# Since the accuracy result is not qualified result, a fine tuning process is done here.
# In this study we tune the parameter --- "min_samples_split" for the decision tree classifier.
# I choose use GridSearchCV method to find the best parameter for this classifier.


acclist = []
precisionlist = []
recalllist = []
skf = StratifiedKFold(n_splits=5)
for train_indices, test_indices in skf.split(features, labels):
    features_train = [features[ii] for ii in train_indices]
    features_test = [features[ii] for ii in test_indices]
    labels_train = [labels[ii] for ii in train_indices]
    labels_test = [labels[ii] for ii in test_indices]
    # make train and test data
    parameters = {'min_samples_split': [2, 12]}
    tre = tree.DecisionTreeClassifier()
    clf = GridSearchCV(tre, parameters)
    clf = clf.fit(features_train, labels_train)
    pred = clf.predict(features_test)
    # obtain accuracy, precision, and recall score
    acc = accuracy_score(pred, labels_test)
    pre = precision_score(labels_test, pred)
    rec = recall_score(labels_test, pred)
    # make a list to obtain an average
    acclist.append(acc)
    precisionlist.append(pre)
    recalllist.append(rec)
print 'after tuning acc = ', (1.0 * sum(acclist)) / len(acclist)
print 'after tuning precision = ', (1.0 * sum(precisionlist)) / len(precisionlist)
print 'after tuning recall = ', (1.0 * sum(recalllist)) / len(recalllist)


# After tuning, the classifier performance is much better.


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf.best_estimator_, my_dataset, features_list)
