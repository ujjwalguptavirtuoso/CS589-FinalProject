import math
import random
import pandas as pd
import numpy as np
from collections import Counter

class Node:
    def __init__(self, attribute=None, threshold=None, is_leaf=False, class_label=None):
        self.attribute = attribute
        self.threshold = threshold
        self.is_leaf = is_leaf
        self.class_label = class_label
        self.branches = {}

class Tree:
    def __init__(self, max_attrs, max_depth):
        self.tree = None
        self.max_attrs = max_attrs
        self.min_size_for_split = 10
        self.max_depth = max_depth

    def calculate_entropy(self, probabilities):
        return sum(-probability * math.log2(probability) for probability in probabilities if probability > 0)
    
    def get_attribute_value_probabilities_per_label(self, n, labels):
        return [freq / n for freq in Counter(labels).values()]
    
    def calculate_weighted_average(self, probabilities, measures):
        return sum(probability * measure for probability, measure in zip(probabilities, measures))

    def get_best_split_numerical(self, training_data, attribute):
        sorted_data = training_data.sort_values(attribute)
        values = sorted_data[attribute].values
        labels = sorted_data.iloc[:, -1].values
        
        # mean of adjacent values as thresholds
        thresholds = [(values[i] + values[i + 1]) / 2 for i in range(len(values) - 1)]
        best_gain = -float("inf")
        best_threshold = None
        
        n = len(training_data)
        entropy_dataset = self.calculate_entropy(self.get_attribute_value_probabilities_per_label(len(labels), labels))
        
        for threshold in thresholds:
            left = training_data[training_data[attribute] <= threshold]
            right = training_data[training_data[attribute] > threshold]
            
            if len(left) < self.min_size_for_split or len(right) < self.min_size_for_split:
                continue

            left_entropy = self.calculate_entropy(self.get_attribute_value_probabilities_per_label(len(left.iloc[:, -1]), left.iloc[:, -1]))
            right_entropy = self.calculate_entropy(self.get_attribute_value_probabilities_per_label(len(right.iloc[:, -1]), right.iloc[:, -1]))
            
            gain = entropy_dataset - self.calculate_weighted_average([len(left) / n, len(right) / n], [left_entropy, right_entropy])
            
            if gain > best_gain:
                best_gain = gain
                best_threshold = threshold
                
        return best_gain, best_threshold

    def get_best_attribute(self, training_data):
        x_train = training_data.iloc[:, :-1]
        y_train = training_data.iloc[:, -1]
        n = len(training_data)
        entropy_dataset = self.calculate_entropy(self.get_attribute_value_probabilities_per_label(len(y_train), y_train))
        best_gain = -float("inf")
        best_attr = None
        best_threshold = None

        attrs = random.sample(list(x_train.columns), min(self.max_attrs, len(x_train.columns)))
        
        for attribute in attrs:
            if pd.api.types.is_numeric_dtype(x_train[attribute]):
                gain, threshold = self.get_best_split_numerical(training_data, attribute)
                
                if gain > best_gain:
                    best_gain = gain
                    best_attr = attribute
                    best_threshold = threshold
            else:
                attr_probs, attr_entropies = [], []
                for value in x_train[attribute].unique():
                    subset = training_data[training_data[attribute] == value]
                    count = len(subset)
                    probs = self.get_attribute_value_probabilities_per_label(len(subset.iloc[:, -1]), subset.iloc[:, -1])
                    entropy = self.calculate_entropy(probs)
                    attr_probs.append(count / n)
                    attr_entropies.append(entropy)
                gain = entropy_dataset - self.calculate_weighted_average(attr_probs, attr_entropies)
                
                if gain > best_gain:
                    best_gain = gain
                    best_attr = attribute
                    best_threshold = None
                    
        return best_attr, best_threshold

    def build(self, data):
        self.tree = self._build_node(data, depth=0)

    def _build_node(self, data, depth):
        x, y = data.iloc[:, :-1], data.iloc[:, -1]
        if len(y.unique()) == 1 or len(data) < self.min_size_for_split or depth >= self.max_depth or x.shape[1] == 0:
            return Node(is_leaf=True, class_label=Counter(y).most_common(1)[0][0])
        attr, thr = self.get_best_attribute(data)
        if attr is None:
            return Node(is_leaf=True, class_label=Counter(y).most_common(1)[0][0])
        node = Node(attribute=attr, threshold=thr)
        if thr is not None:
            left = data[data[attr] <= thr].drop(columns=[attr])
            right = data[data[attr] > thr].drop(columns=[attr])
            node.branches['left'] = self._build_node(left, depth+1)
            node.branches['right'] = self._build_node(right, depth+1)
        else:
            for val in data[attr].unique():
                subset = data[data[attr] == val].drop(columns=[attr])
                node.branches[val] = self._build_node(subset, depth+1)
        return node

    def predict(self, row):
        node = self.tree
        while not node.is_leaf:
            if node.threshold is not None:
                branch = 'left' if row[node.attribute] <= node.threshold else 'right'
            else:
                branch = row[node.attribute]
            node = node.branches.get(branch, next(iter(node.branches.values())))
        return node.class_label

class RandomForestClassifier:
    def __init__(self, ntree, depth):
        self.ntree = ntree
        self.depth = depth
        self.trees = []

    def fit(self, training_data):
        self.trees = []

        for i in range(self.ntree):
            sample = self.bootstrap(training_data)
            tree = Tree(int(math.sqrt(training_data.shape[1] - 1)), self.depth)
            tree.build(sample)
            self.trees.append(tree)

    def bootstrap(self, data):
        sampled = data.sample(n=len(data), replace=True)
        return sampled

    def predict(self, instance):
        return Counter([tree.predict(instance) for tree in self.trees]).most_common(1)[0][0]

    def predict_labels(self, test_data):
        return [self.predict(test) for _, test in test_data.iterrows()]

def calculate_metrics(predictions, actuals):
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    accuracy = np.mean(predictions == actuals)
    classes = np.unique(actuals)
    f1_scores = []

    for c in classes:
        tp = np.sum((predictions == c) & (actuals == c))
        fp = np.sum((predictions == c) & (actuals != c))
        fn = np.sum((predictions != c) & (actuals == c))
        precision = tp/(tp+fp) if (tp+fp) else 0
        recall = tp/(tp+fn) if (tp+fn) else 0
        f1_scores.append(2*precision*recall/(precision+recall) if (precision+recall) else 0)
    
    return accuracy, sum(f1_scores) / len(f1_scores)