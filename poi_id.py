#!/usr/bin/python

import sys
import pickle
import matplotlib.pyplot
import numpy
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import test_classifier, dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary', 'bonus', 'total_stock_value', 
'exercised_stock_options', 'deferred_income']   # You will need to use more features

### Load the dictionary containing the dataset
data_dict = pickle.load(open("final_project_dataset.pkl", "r") )

### Task 2: Remove outliers
data_dict.pop("TOTAL")
data_dict.pop('THE TRAVEL AGENCY IN THE PARK')

### See how many POI and how many non-POI
num_poi = 0
num_non_poi = 0
for key in data_dict.keys():
    if data_dict[key]['poi'] == True:
        num_poi += 1
    elif data_dict[key]['poi'] == False:
        num_non_poi += 1
        
### Task 3: Create new feature(s)
for key in data_dict.keys():
    if data_dict[key]['from_poi_to_this_person'] == 'NaN' or data_dict[key]['from_this_person_to_poi'] == 'NaN' or data_dict[key]['from_messages'] == 'NaN' or data_dict[key]['to_messages'] == 'NaN':
        data_dict[key]['fraction_poi_related_emails'] = 'NaN'
    else:
        data_dict[key]['fraction_poi_related_emails'] = (float(data_dict[key]['from_poi_to_this_person']) + 
    float(data_dict[key]['from_this_person_to_poi'])) / (float(data_dict[key]['from_messages']) + 
    float(data_dict[key]['to_messages']))
    
### Feature scaling
salary = []
bonus = []
total_stock_value = []
shared_receipt_with_poi = []
fraction_poi_related_emails = []
loan_advances = []
exercised_stock_options = []
deferral_payments = []
deferred_income = []
director_fees = []
expenses = []
long_term_incentive = []
other = []
restricted_stock = []
restricted_stock_deferred = []
total_payments = []
from_poi_to_this_person = []
from_this_person_to_poi = []

for key in data_dict.keys():
    if data_dict[key]['salary'] == 'NaN':
        salary.append(float(0))
    else:
        salary.append(data_dict[key]['salary'])
for key in data_dict.keys():
    if data_dict[key]['bonus'] == 'NaN':
        bonus.append(float(0))
    else:
        bonus.append(data_dict[key]['bonus'])
for key in data_dict.keys():
    if data_dict[key]['total_stock_value'] == 'NaN':
        total_stock_value.append(float(0))
    else: 
        total_stock_value.append(data_dict[key]['total_stock_value'])
for key in data_dict.keys():
    if data_dict[key]['shared_receipt_with_poi'] == 'NaN':
        shared_receipt_with_poi.append(float(0))
    else:
        shared_receipt_with_poi.append(data_dict[key]['shared_receipt_with_poi'])
for key in data_dict.keys():
    if data_dict[key]['fraction_poi_related_emails'] == 'NaN':
        fraction_poi_related_emails.append(float(0))
    else:
        fraction_poi_related_emails.append(data_dict[key]['fraction_poi_related_emails'])
for key in data_dict.keys():
    if data_dict[key]['loan_advances'] == 'NaN':
        loan_advances.append(float(0))
    else:
        loan_advances.append(data_dict[key]['loan_advances'])
for key in data_dict.keys():
    if data_dict[key]['exercised_stock_options'] == 'NaN':
        exercised_stock_options.append(float(0))
    else:
        exercised_stock_options.append(data_dict[key]['exercised_stock_options'])
for key in data_dict.keys():
    if data_dict[key]['deferral_payments'] == 'NaN':
        deferral_payments.append(float(0))
    else:
        deferral_payments.append(data_dict[key]['deferral_payments'])
for key in data_dict.keys():
    if data_dict[key]['deferred_income'] == 'NaN':
        deferred_income.append(float(0))
    else:
        deferred_income.append(data_dict[key]['deferred_income'])
for key in data_dict.keys():
    if data_dict[key]['director_fees'] == 'NaN':
        director_fees.append(float(0))
    else:
        director_fees.append(data_dict[key]['director_fees'])
for key in data_dict.keys():
    if data_dict[key]['expenses'] == 'NaN':
        expenses.append(float(0))
    else:
        expenses.append(data_dict[key]['expenses'])
for key in data_dict.keys():
    if data_dict[key]['long_term_incentive'] == 'NaN':
        long_term_incentive.append(float(0))
    else:
        long_term_incentive.append(data_dict[key]['long_term_incentive'])
for key in data_dict.keys():
    if data_dict[key]['other'] == 'NaN':
        other.append(float(0))
    else:
        other.append(data_dict[key]['other'])
for key in data_dict.keys():
    if data_dict[key]['restricted_stock'] == 'NaN':
        restricted_stock.append(float(0))
    else:
        restricted_stock.append(data_dict[key]['restricted_stock'])
for key in data_dict.keys():
    if data_dict[key]['restricted_stock_deferred'] == 'NaN':
        restricted_stock_deferred.append(float(0))
    else:
        restricted_stock_deferred.append(data_dict[key]['restricted_stock_deferred'])
for key in data_dict.keys():
    if data_dict[key]['total_payments'] == 'NaN':
        total_payments.append(float(0))
    else:
        total_payments.append(data_dict[key]['total_payments'])
for key in data_dict.keys():
    if data_dict[key]['from_poi_to_this_person'] == 'NaN':
        from_poi_to_this_person.append(float(0))
    else:
        from_poi_to_this_person.append(data_dict[key]['from_poi_to_this_person'])
for key in data_dict.keys():
    if data_dict[key]['from_this_person_to_poi'] == 'NaN':
        from_this_person_to_poi.append(float(0))
    else:
        from_this_person_to_poi.append(data_dict[key]['from_this_person_to_poi'])
    
from sklearn import preprocessing
scalar = preprocessing.MinMaxScaler()
salary = scalar.fit_transform(numpy.asarray(salary)) # salary must have float element
bonus = scalar.fit_transform(numpy.asarray(bonus))
total_stock_value = scalar.fit_transform(numpy.asarray(total_stock_value))
shared_receipt_with_poi = scalar.fit_transform(numpy.asarray(shared_receipt_with_poi))
fraction_poi_related_emails = scalar.fit_transform(numpy.asarray(fraction_poi_related_emails))
loan_advances = scalar.fit_transform(numpy.asarray(loan_advances))
exercised_stock_options = scalar.fit_transform(numpy.asarray(exercised_stock_options))
deferral_payments = scalar.fit_transform(numpy.asarray(deferral_payments))
deferred_income = scalar.fit_transform(numpy.asarray(deferred_income))
director_fees = scalar.fit_transform(numpy.asarray(director_fees))
expenses = scalar.fit_transform(numpy.asarray(expenses))
long_term_incentive = scalar.fit_transform(numpy.asarray(long_term_incentive))
other = scalar.fit_transform(numpy.asarray(other))
restricted_stock = scalar.fit_transform(numpy.asarray(restricted_stock))
restricted_stock_deferred = scalar.fit_transform(numpy.asarray(restricted_stock_deferred))
total_payments = scalar.fit_transform(numpy.asarray(total_payments))
from_poi_to_this_person = scalar.fit_transform(numpy.asarray(from_poi_to_this_person))
from_this_person_to_poi = scalar.fit_transform(numpy.asarray(from_this_person_to_poi))

nsalary = 0
nbonus = 0
ntotal_stock_value = 0
nshared_receipt_with_poi = 0
nfraction_poi_related_emails = 0
nloan_advances = 0
nexercised_stock_options = 0
ndeferral_payments = 0
ndeferred_income = 0
ndirector_fees = 0
nexpenses = 0
nlong_term_incentive = 0
nother = 0
nrestricted_stock = 0
nrestricted_stock_deferred = 0
ntotal_payments = 0
nfrom_poi_to_this_person = 0
nfrom_this_person_to_poi = 0

for key in data_dict.keys():
    data_dict[key]['salary'] = salary[nsalary]
    nsalary += 1
for key in data_dict.keys():
    data_dict[key]['bonus'] = bonus[nbonus]
    nbonus += 1
for key in data_dict.keys():
    data_dict[key]['total_stock_value'] = total_stock_value[ntotal_stock_value]
    ntotal_stock_value += 1
for key in data_dict.keys():
    data_dict[key]['shared_receipt_with_poi'] = shared_receipt_with_poi[nshared_receipt_with_poi]
    nshared_receipt_with_poi += 1
for key in data_dict.keys():
    data_dict[key]['fraction_poi_related_emails'] = fraction_poi_related_emails[nfraction_poi_related_emails]
    nfraction_poi_related_emails += 1
for key in data_dict.keys():
    data_dict[key]['loan_advances'] = loan_advances[nloan_advances]
    nloan_advances += 1
for key in data_dict.keys():
    data_dict[key]['exercised_stock_options'] = exercised_stock_options[nexercised_stock_options]
    nexercised_stock_options += 1
for key in data_dict.keys():
    data_dict[key]['deferral_payments'] = deferral_payments[ndeferral_payments]
    ndeferral_payments += 1
for key in data_dict.keys():
    data_dict[key]['deferred_income'] = deferred_income[ndeferred_income]
    ndeferred_income += 1
for key in data_dict.keys():
    data_dict[key]['director_fees'] = director_fees[ndirector_fees]
    ndirector_fees += 1
for key in data_dict.keys():
    data_dict[key]['expenses'] = expenses[nexpenses]
    nexpenses += 1
for key in data_dict.keys():
    data_dict[key]['long_term_incentive'] = long_term_incentive[nlong_term_incentive]
    nlong_term_incentive += 1
for key in data_dict.keys():
    data_dict[key]['other'] = other[nother]
    nother += 1
for key in data_dict.keys():
    data_dict[key]['restricted_stock'] = restricted_stock[nrestricted_stock]
    nrestricted_stock += 1
for key in data_dict.keys():
    data_dict[key]['restricted_stock_deferred'] = restricted_stock_deferred[nrestricted_stock_deferred]
    nrestricted_stock_deferred += 1
for key in data_dict.keys():
    data_dict[key]['total_payments'] = total_payments[ntotal_payments]
    ntotal_payments += 1
for key in data_dict.keys():
    data_dict[key]['from_poi_to_this_person'] = from_poi_to_this_person[nfrom_poi_to_this_person]
    nfrom_poi_to_this_person += 1
for key in data_dict.keys():
    data_dict[key]['from_this_person_to_poi'] = from_this_person_to_poi[nfrom_this_person_to_poi]
    nfrom_this_person_to_poi += 1    
    
    
### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Univariate Feature Selection
from sklearn.feature_selection import SelectPercentile, f_classif
selector = SelectPercentile(f_classif)
selector.fit(features, labels)
print selector.scores_

### Plot the data to see if outliers exist
for point in data:
    salary = point[1]
    total_payments = point[2]
    if point[0] == True:
        matplotlib.pyplot.scatter(salary, total_payments, c = 'r', alpha = 0.8, s = 50)
    else:
        matplotlib.pyplot.scatter(salary, total_payments, c = 'b', alpha = 0.8, s = 50)

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()

#for point in data:
#    bonus = point[3]
#    total_stock_value = point[4]
#    color = point[0]
#    matplotlib.pyplot.scatter(bonus, total_stock_value, c = color, alpha = 0.8, s = 50)

#matplotlib.pyplot.xlabel("bonus")
#matplotlib.pyplot.ylabel("total_stock_value")
#matplotlib.pyplot.show()

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()    # Provided to give you a starting point. Try a varity of classifiers.
clf.fit(features, labels)

#from sklearn.svm import SVC
#clf = SVC(kernel = 'linear')
#clf.fit(features, labels)

### Recursive feature selection for SVM
#from sklearn.feature_selection import RFECV
#selector = RFECV(clf, step=1)
#selector = selector.fit(features, labels)
#selector.support_ 
#selector.ranking_

#from sklearn import tree
#clf = tree.DecisionTreeClassifier(min_samples_split = 1, splitter = 'random')
#clf.fit(features, labels)
#print clf.feature_importances_


### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script.
### Because of the small size of the dataset, the script uses stratified
### shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

### Validation
from sklearn.cross_validation import KFold
kf = KFold(len(my_dataset), 3)

for train_indices, test_indices in kf:
    features_train = [features[i] for i in train_indices]
    features_test = [features[i] for i in test_indices]
    labels_train = [labels[i] for i in train_indices]
    labels_test = [labels[i] for i in test_indices]

clf2 = GaussianNB()
clf2.fit(features_train, labels_train)

true_negatives = 0
false_negatives = 0
true_positives = 0
false_positives = 0
predictions = clf2.predict(features_test)
for prediction, truth in zip(predictions, labels_test):
    if prediction == 0 and truth == 0:
        true_negatives += 1
    elif prediction == 0 and truth == 1:
        false_negatives += 1
    elif prediction == 1 and truth == 0:
        false_positives += 1
    else:
        true_positives += 1
total_predictions = true_negatives + false_negatives + false_positives + true_positives
accuracy = 1.0*(true_positives + true_negatives)/total_predictions
precision = 1.0*true_positives/(true_positives+false_positives)
recall = 1.0*true_positives/(true_positives+false_negatives)
f1 = 2.0 * true_positives/(2*true_positives + false_positives+false_negatives)
f2 = (1+2.0*2.0) * precision*recall/(4*precision + recall)
print 'accuracy is', accuracy
print 'precision is', precision
print 'recall is', recall
print 'f1 is', f1
print 'f2 is', f2

test_classifier(clf, my_dataset, features_list)

### Dump your classifier, dataset, and features_list so 
### anyone can run/check your results.

dump_classifier_and_data(clf, my_dataset, features_list)