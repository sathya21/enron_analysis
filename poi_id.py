#!/usr/bin/python

from sklearn import grid_search


def get_gaussian():
   from sklearn.naive_bayes import GaussianNB
   param_grid = {}
   clf = grid_search.GridSearchCV(GaussianNB(), param_grid)
   return clf

def get_decision_tree():
    from sklearn import tree
    from sklearn import grid_search
    param_grid = {'min_samples_split': [2,3,4,5,6],'max_features':[3,4]}

    clf = grid_search.GridSearchCV(tree.DecisionTreeClassifier(), param_grid)
    return clf

def get_random_forest():
    from sklearn.ensemble import RandomForestClassifier

    param_grid = {'n_estimators': [10,50],'min_samples_split':[2,3,4,5],'max_features':[5]}

    clf = grid_search.GridSearchCV(RandomForestClassifier(),param_grid,n_jobs=5)
    return clf

def get_svc():
    from sklearn import svm
    from sklearn import grid_search

    param_grid = [
        {'C': [10], 'kernel': ['linear']},
        #{'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
        ]
    clf = grid_search.GridSearchCV(svm.SVC(),param_grid, n_jobs=1)
    return clf

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features = ['poi','salary','bonus', 'total_stock_value', 'total_payments','long_term_incentive','deferred_income', 'exercised_stock_options','from_poi_to_this_person','from_this_person_to_poi','to_messages','from_messages','shared_receipt_with_poi']
features_list = ['poi','total_payments','total_stock_value']
### Load the dictionary containing the dataset'restricted_stock
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)




### Task 2: Remove outliers
data_dict.pop('TOTAL',0)


### Task 3: Create new feature(s)
poi_mail_ratio =0

for key in data_dict:
    total = 0

    for feature in features:
        value = data_dict[key][feature]
        if value=="NaN":
            value = 0
        data_dict[key][feature] = value

    for feature in features_list:
        if feature != "poi":
            total = total + data_dict[key][feature]

    #print 'poi, %s, key, %s, total, %f'%(data_dict[key]['poi'],key,total)
    if (data_dict[key]['to_messages'] == 0 and data_dict[key]['from_messages'] == 0):
        poi_mail_ratio =0
    else:
        poi_to_mail_ratio = (data_dict[key]['from_this_person_to_poi'] )/ float(data_dict[key]['to_messages'])
        poi_from_mail_ratio =0

        if data_dict[key]['from_messages'] >= data_dict[key]['from_poi_to_this_person']:
            poi_from_mail_ratio = (data_dict[key]['from_poi_to_this_person'] )/ float(data_dict[key]['from_messages'])
        #print 'poi mail from %f'%poi_from_mail_ratio

        #if data_dict[key]['poi'] == False:
            #print 'key %s poi_to_mail_ratio %f to_messages %d from_this_person_to_poi %d shared_receipt_with_poi %d'%(key,poi_to_mail_ratio,data_dict[key]['to_messages'],data_dict[key]['from_this_person_to_poi'],data_dict[key]['shared_receipt_with_poi'])
            #print 'key %s poi_from_mail_ratio %f from_messages %d from_poi_to_this_person %d'%(key,poi_from_mail_ratio,data_dict[key]['from_messages'],data_dict[key]['from_poi_to_this_person'])

    data_dict[key]['poi_to_mail_ratio'] = poi_to_mail_ratio
    data_dict[key]['poi_from_mail_ratio'] = poi_from_mail_ratio
    print 'poi, %s, key, %s, total, %f stock, %f, to, %f, from, %f'%(data_dict[key]['poi'],key,data_dict[key]['total_payments'],data_dict[key]['total_stock_value'],poi_to_mail_ratio,poi_from_mail_ratio)



features_list.append('poi_to_mail_ratio')
features_list.append('poi_from_mail_ratio')



print features_list

### Store to my_dataset for easy export below.
my_dataset = data_dict


### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)




print 'lables %s feature%s'%(labels[0],features[0])

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.

#clf = get_gaussian()

clf = get_decision_tree()

#clf = get_random_forest()

#clf = get_svc()


#clf = grid_search.GridSearchCV(AdaBoostClassifier(),param_grid)
### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info:
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

print features_train[0]


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
