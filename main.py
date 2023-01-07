import pandas as pd
import numpy as np
import sys
import json
import argparse
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import recall_score, f1_score, accuracy_score
from sklearn.model_selection import cross_val_score
from ID3Implementation import ID3Implementation

def add_classification_reports(searched_class_values, results, cr):
    for searched_class_value, values in cr.items():
        if searched_class_value == 'accuracy':
            results[searched_class_value] += values
        else:
            for metric, value in values.items():
                results[searched_class_value][metric] = results[searched_class_value].get(metric,0) + value
    return results

def divide_results(results, divider):
    for searched_class_value, values in results.items():
        if searched_class_value == 'accuracy':
            results[searched_class_value] /= divider
        else:
            for metric, value in values.items():
                results[searched_class_value][metric] /= divider
    return results

def ordinary_test(df, predicted_atribute, how_many, weights):
    results = {}
    searched_class_values = df[predicted_atribute].unique()
    for value in searched_class_values:
        results[value] = {}
    results['accuracy'] = 0
    results['macro avg'] = {}
    results['weighted avg'] = {}
    for i in range(0,how_many):
        training_set, testing_set = train_test_split(df, test_size=0.2,stratify=df[predicted_atribute])
        id3 = ID3Implementation(training_set, predicted_atribute, weights)
        predictions = id3.predict(testing_set)
        cr = classification_report(testing_set[predicted_atribute], predictions, output_dict=True)
        results = add_classification_reports(searched_class_values, results, cr)
    return divide_results(results, how_many)

def decision_tree_classifier(df, predicted_atribute, how_many, weights):
    results = {}
    searched_class_values = df[predicted_atribute].unique()
    for value in searched_class_values:
        results[value] = {}
    results['accuracy'] = 0
    results['macro avg'] = {}
    results['weighted avg'] = {}
    for i in range(0,how_many):
        clf = DecisionTreeClassifier()
        X = df.drop(predicted_atribute, axis=1)
        le = preprocessing.LabelEncoder()
        for (columnName, columnData) in X.iteritems():
            X[columnName] = le.fit_transform(columnData)
        y = df[predicted_atribute]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        clf = clf.fit(X_train,y_train)
        y_pred = clf.predict(X_test)
        cr = classification_report(y_test, y_pred, output_dict=True)
        results = add_classification_reports(searched_class_values, results, cr)
    return divide_results(results, how_many)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("filename")
    parser.add_argument("predicted_atribute")
    parser.add_argument("output_file")
    parser.add_argument("times")
    parser.add_argument("test")
    args = parser.parse_args()

    try:
        df = pd.read_csv(args.filename)
    except FileNotFoundError:
        sys.exit("No datafile found.")

    predicted_atribute = args.predicted_atribute
    output_file = args.output_file
    times = args.times
    test = args.test

    if predicted_atribute not in df.keys():
        sys.exit("There is no such atribute in this datafile.")

    results = None
    if(test == '1'):
        results = ordinary_test(df, predicted_atribute,  int(times), False)
    if(test == '2'):
        results = decision_tree_classifier(df, predicted_atribute,  int(times), False)
    if(test == '5'):
        results = ordinary_test(df, predicted_atribute,  int(times), True)
    with open('output/'+output_file, 'w') as convert_file:
        convert_file.write(json.dumps(results))

if __name__ == "__main__":
    main()
