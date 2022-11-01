import pandas as pd
import numpy as np
import sys
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, precision_score
from sklearn.metrics import recall_score, f1_score, accuracy_score
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score
from ID3Implementation import ID3Implementation

def add_classification_reports(searched_class_values, results, cr):
    for searched_class_value, values in cr.items():
        if searched_class_value in searched_class_values:
            for metric, value in values.items():
                results[searched_class_value][metric] = results[searched_class_value].get(metric,0) + value
        elif searched_class_value == 'accuracy':
            results[searched_class_value] += values
    return results
def divide_results(results, divider):
    for searched_class_value, values in results.items():
        if searched_class_value == 'accuracy':
            results[searched_class_value] /= divider
        else:
            for metric, value in values.items():
                results[searched_class_value][metric] /= divider
    return results

def ordinary_test(df, predicted_atribute, how_many):
    results = {}
    searched_class_values = df[predicted_atribute].unique()
    for value in searched_class_values:
        results[value] = {}
    results['accuracy'] = 0
    for i in range(0,how_many):
        training_set, testing_set = train_test_split(df, test_size=0.2,stratify=df[predicted_atribute])
        id3 = ID3Implementation(training_set,predicted_atribute)
        predictions = id3.predict(testing_set)
        cr = classification_report(testing_set[predicted_atribute], predictions, output_dict=True)
        results = add_classification_reports(searched_class_values, results, cr)
    return divide_results(results, how_many)
def main():

    print("Datafile to read from.")
    filename = input()
    try:
        df = pd.read_csv(filename)
    except FileNotFoundError:
        sys.exit("No datafile found.")

    print('Predicted atribute.')
    predicted_atribute = input()

    print('Output file.')
    output_file = input()

    print('How many tests.')
    times = input()

    if predicted_atribute not in df.keys():
        sys.exit("There is no such atribute in this datafile.")

    print('Which test?')
    print('1) Ordinary test')
    print('2) DecisionTreeClassifier test')
    print('3) Over-sampling test')
    print('4) Under-sampling test')
    print('5) Added weights test')
    test = input()
    results = None
    if(test == '1'):
        results = ordinary_test(df, predicted_atribute,  int(times))
    with open('output/'+output_file, 'w') as convert_file:
        convert_file.write(json.dumps(results))

if __name__ == "__main__":
    main()
