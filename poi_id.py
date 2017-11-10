#!/usr/bin/python
import sys
import pickle
sys.path.append("../tools/")
import matplotlib.pyplot as plt
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi', 'salary', 'bonus','fraction_from_poi_email', 'fraction_to_poi_email','shared_receipt_with_poi'] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

# get a brief preview of data
print data_dict

### Task 2: Remove outliers

# from the preview of data_dict, the TOTAL is the key which need to be removed.

data_dict.pop('TOTAL', 0)


### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.


### create two lists of new features (fraction_from_poi_email, fraction_to_poi_email)
def dict_to_list(key,normalizer):
    new_list=[]
    for i in data_dict:
        if data_dict[i][key] == "NaN" or data_dict[i][normalizer] == "NaN":
            new_list.append(0.)
        elif data_dict[i][key] >= 0:
            new_list.append(float(data_dict[i][key])/float(data_dict[i][normalizer]))
    return new_list

fraction_from_poi_email = dict_to_list("from_poi_to_this_person", "to_messages")
fraction_to_poi_email = dict_to_list("from_this_person_to_poi", "from_messages")

### insert new features into data_dict

count=0
for i in data_dict:
    data_dict[i]["fraction_from_poi_email"] = fraction_from_poi_email[count]
    data_dict[i]["fraction_to_poi_email"] = fraction_to_poi_email[count]
    count +=1
my_dataset = data_dict

### Extract features and labels from dataset for local testing

data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)


### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.

# from sklearn.model_selection import train_test_split

from sklearn import cross_validation
features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(features, labels, test_size = 0.5, random_state=42)

# try the GaussianNB method at first
clf = GaussianNB()
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
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
print 'tree feature importances = ', clf.feature_importances_

# through the comparision, it is concluded that decision tree method is better.

# using another way to identify the accuracy of decision tree method.
acclist = []
precisionlist = []
recalllist = []
from sklearn.cross_validation import KFold
kf = KFold(len(labels),5)
for train_indices, test_indices in kf:
    features_train= [features[ii] for ii in train_indices]
    features_test= [features[ii] for ii in test_indices]
    labels_train=[labels[ii] for ii in train_indices]
    labels_test=[labels[ii] for ii in test_indices]
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(features_train, labels_train)
    pred = clf.predict(features_test)
    acc = accuracy_score(pred, labels_test)
    pre = precision_score(labels_test, pred)
    rec = recall_score(labels_test, pred)
    acclist.append(acc)
    precisionlist.append(pre)
    recalllist.append(rec)
print 'tree acc = ' ,(1.0*sum(acclist))/len(acclist)
print 'tree precision = ' ,(1.0*sum(precisionlist))/len(precisionlist)
print 'tree recall = ' ,(1.0*sum(recalllist))/len(recalllist)


### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!

# Since the accuracy result is not qualified result, a fine tuning process is done here.



from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

from sklearn.model_selection import GridSearchCV
parameters = {'min_samples_split': [2, 10]}
tre =  tree.DecisionTreeClassifier()
clf = GridSearchCV(tre, parameters)
clf = clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
print 'tree acc = ', accuracy_score(pred, labels_test)
print 'recall = ', recall_score(labels_test, pred)
print 'precision = ', precision_score(labels_test, pred)


# After tuning, it is much better.
















### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
