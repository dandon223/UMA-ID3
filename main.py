import pandas as pd
import numpy as np
import sys
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, precision_score
from sklearn.metrics import recall_score, f1_score, accuracy_score
from ID3Implementation import ID3Implementation




def main():

    df = pd.read_csv("input/play.csv")
    df = pd.read_csv("input/carData.csv")
    df = pd.read_csv("input/lungCancerData.csv")

    training_set, testing_set = train_test_split(df, test_size=0.2,random_state=10,stratify=df["LUNG_CANCER"])
    id3 = ID3Implementation(training_set,"LUNG_CANCER")
    id3.show(id3.node)
    print(training_set[training_set["AGE"]==21])
    print(testing_set[testing_set["AGE"]==21])
    predictions = id3.predict(testing_set)
    print(classification_report(testing_set["LUNG_CANCER"], predictions))
    print(confusion_matrix(testing_set["LUNG_CANCER"], predictions))
    print(accuracy_score(testing_set["LUNG_CANCER"], predictions))




if __name__ == "__main__":
    main()
