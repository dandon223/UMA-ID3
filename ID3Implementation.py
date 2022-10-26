import pandas as pd
import numpy as np
import sys
from sklearn.model_selection import train_test_split

class ID3Implementation:
    def __init__(self, training_set, searched_class):
        self.training_set = training_set
        self.searched_class = searched_class
        self.node = Node();
        self.node = self.create_tree(self.training_set)

    def predict(self, testing_set):
        predictions = []
        for index, row in testing_set.iterrows():
            #print(row)
            current_node = self.node
            while(True):
                #print(current_node.value)
                if type(current_node) == LeafNode:
                    predictions.append(current_node.value)
                    break
                current_node = current_node.children.get(row[current_node.value])
                if(current_node == None):
                    predictions.append("None")
                    break
        return predictions

    # metoda ta rekursywnie buduje drzewo
    def create_tree(self, set):
        training_set = set
        attribute = self.best_ig(training_set)
        # jesli brak atrybutow do wykorzystania, to ustalamy klase rowna wiekszosciowej w zbiorze
        if(attribute == None):
            node = LeafNode()
            node.value = training_set[self.searched_class].value_counts().idxmax()
            return node
        print("best_attribute = "+ attribute)
        node = Node()
        node.value = attribute
        attribute_values = self.training_set[attribute].unique()
        # petla dla wszystkich wartosci atrybutu w calym zbiorze treningowym
        for value in attribute_values:
            sub_training_set = training_set[training_set[attribute] == value].reset_index(drop=True)
            # sprawdzamy ile roznych wartosci klasy jest w naszym wyodrebnionym podzbiorze
            searched_class_values, searched_class_values_counters = np.unique(sub_training_set[self.searched_class], return_counts=True)
            # jesli jest ich wiecej niz jeden to musimy znowu dzielic nasz podzbior
            if len(searched_class_values_counters) > 1:
                node.children[value] = self.create_tree(sub_training_set)
            # jesli dla rozpatrywanego podzbioru nie ma elementow, tworzymy lisc z klasa wiekszosciowa calego zbioru
            elif searched_class_values.size == 0:
                new_node = LeafNode()
                new_node.value = training_set[self.searched_class].value_counts().idxmax()
                node.children[value] = new_node
            # jesli jest tylko jedna wartosc klasy w naszym podzbiorze, to jest to nasz lisc
            else:
                new_node = LeafNode()
                new_node.value = searched_class_values[0]
                node.children[value] = new_node
        return node



    def best_ig(self,training_set):
        biggest_ig = 0
        best_attribute = None
        entropy_value = self.entropy(training_set)
        for attribute in training_set.keys():
            if attribute == self.searched_class:
                continue
            information_gain = entropy_value - self.attribute_entropy(training_set, attribute)
            if information_gain > biggest_ig:
                best_attribute = attribute
                biggest_ig = information_gain
        return best_attribute

    def attribute_entropy(self,training_set, attribute):
        attribute_entropy = 0
        for attribute_value in training_set[attribute].unique():
            attribute_value_probability = len(training_set[training_set[attribute] == attribute_value]) / len(training_set)
            attribute_entropy += attribute_value_probability * self.entropy(training_set[training_set[attribute] == attribute_value])
        return attribute_entropy

    def entropy(self,training_set):
        values = training_set[self.searched_class].unique()
        set_size = len(training_set[self.searched_class])
        entropy = 0
        for value in values:
            probability = training_set[self.searched_class].value_counts()[value] / set_size
            entropy += -probability * np.log2(probability)
        return entropy

    def show(self,node):
        if type(node) is Node:
            print("Node| attribute = "+ node.value)
            for key,value in node.children.items():
                print(node.value +","+str(key))
                self.show(value)
        else:
            print("LeafNode| value = "+ node.value)



class Node:
    def __init__(self):
        self.value = None
        self.children = {}

class LeafNode:
    def __init__(self):
        self.value = None
