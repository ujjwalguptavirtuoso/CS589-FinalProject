import csv
from neural_network_implementation import NeuralNet
import matplotlib.pyplot as plt
import random
from sklearn.datasets import load_digits

def read_data(file_path, delimiter, label_location_beginning=False):
    file = open(file_path, 'r')
    data = (list)(csv.reader(file, delimiter=delimiter))
    data = data[1:]
    final_data = []
    if label_location_beginning:
        for i in data:
            final_data.append(i[1:] + [i[0]])
    else:
        final_data = data

    data = [list(map(float, x)) for x in final_data]
    file.close()
    return data

def shuffle_data(data):
    random.shuffle(data)
    return data

def prepare_data(data):
    instances = [x[:-1] for x in data]
    instances = [list(map(float, x)) for x in instances]
    labels = [float(x[-1]) for x in data]
    return instances, labels


def get_stratified_folds(data, k):
    folds = [[] for i in range(k)]
    data_split = dict()

    for d in data:
        d_class = int(d[-1])
        if d_class in data_split:
            data_split[d_class].append(d)
        else:
            data_split[d_class] = [d]

    for d_class in data_split:
        for i in range(len(data_split[d_class])):
            folds[i % k].append(data_split[d_class][i])

    return folds


def normalize_data(instances):
    columns = len(instances[0])
    rows = len(instances)

    for i in range(columns):
        x_max = instances[0][i]
        x_min = instances[0][i]
        for j in range(1, rows):
            if instances[j][i] > x_max:
                x_max = instances[j][i]
            if instances[j][i] < x_min:
                x_min = instances[j][i]

        # Only normalize if there's variability in the column
        if x_max != x_min:
            for j in range(rows):
                instances[j][i] = (instances[j][i] - x_min) / (x_max - x_min)
        else:
            # If no variability in the column, set constant values to 0 or leave them unchanged
            for j in range(rows):
                instances[j][i] = 0.0

    return instances

def get_final_data(data_path, delimeter, total_classes, label_location_beginning=False, num_folds=10):
    data = read_data(data_path, delimeter, label_location_beginning)
    stratified_data = get_stratified_folds(data, num_folds)
    final_stratified_data = []

    for s in stratified_data:
        final_instances = []
        final_labels = []
        for i in range(len(s)):
            current_label = [0 for _ in range(total_classes)]
            instance = s[i][:-1]
            l = int(s[i][-1])
            current_label[l - 1] = 1
            final_labels.append(current_label)
            final_instances.append(instance)
        normalized_instances = normalize_data(final_instances)
        final_stratified_data.append([normalized_instances, final_labels])

    return final_stratified_data

def get_final_data_from_rows(data_rows, total_classes, num_folds=10):
    stratified_data = get_stratified_folds(data_rows, num_folds)
    final_stratified_data = []

    for fold in stratified_data:
        final_instances = []
        final_labels = []

        for row in fold:
            instance = row[:-1]
            label_index = int(row[-1])
            one_hot_label = [0 for _ in range(total_classes)]
            one_hot_label[label_index] = 1

            final_instances.append(instance)
            final_labels.append(one_hot_label)

        normalized_instances = normalize_data(final_instances)
        final_stratified_data.append([normalized_instances, final_labels])

    return final_stratified_data

def get_train_test(stratified_data, test_fold_index):
    training_instances = []
    testing_instances = []
    training_labels = []
    testing_labels = []
    for i in range(len(stratified_data)):
        d = stratified_data[i]
        if i == test_fold_index:
            testing_labels += (d[1])
            testing_instances += (d[0])
        else:
            training_instances += (d[0])
            training_labels += (d[1])

    return training_instances, training_labels, testing_instances, testing_labels


def get_max_index(x):
    index = 0
    ma = 0
    for i in range(len(x)):
        if x[i] > ma:
            ma = x[i]
            index = i

    return index


def calculate_metrics(predicted_labels, expected_labels, output_classes):
    confusion_matrix = [[0 for i in range(output_classes)] for j in range(output_classes)]

    for i in range(len(predicted_labels)):
        predicted_class = get_max_index(predicted_labels[i])
        true_class = get_max_index(expected_labels[i])

        confusion_matrix[true_class][predicted_class] += 1

    accuracy = 0
    for i in range(len(confusion_matrix)):
        accuracy += confusion_matrix[i][i]
    precisions = []
    recalls = []

    for i in range(len(confusion_matrix)):
        current_recall = confusion_matrix[i][i] / sum(confusion_matrix[i])
        recalls.append(current_recall)

    for i in range(len(confusion_matrix)):
        current_precision = confusion_matrix[i][i]
        total = 0
        for j in range(len(confusion_matrix)):
            total += confusion_matrix[j][i]
        if total == 0:
            continue
        current_precision /= total
        precisions.append(current_precision)

    total_recall = sum(recalls) / len(recalls)
    total_precision = sum(precisions) / len(precisions)
    f1_score = 2 * total_precision * total_recall / (total_recall + total_precision)
    accuracy /= len(predicted_labels)

    return accuracy, f1_score


def test_neural_net(layers, reg_lambda, learning_rate, data, output_classes):
    accuracies, f_scores = [], []

    for i in range(len(data)):
        training_instances, training_labels, testing_instances, testing_labels = get_train_test(data, i)
        nn = NeuralNet(layers, reg_lambda, training_instances, training_labels, learning_rate)

        #m : number of iterations
        for i in range(1000):
            nn.back_propagation()

        predicted_labels = []
        for i in testing_instances:
            activations, z_list = nn.forward_propagation(i)
            predicted = activations[-1]
            predicted_labels.append(predicted)

        accuracy, f1_score = calculate_metrics(predicted_labels, testing_labels, output_classes)
        accuracies.append(accuracy)
        f_scores.append(f1_score)

    return sum(accuracies) / len(accuracies), sum(f_scores) / len(f_scores)


def generate_learning_curve(layers, reg_lambda, alpha, full_data, output_classes, dataset_name, step_size, max_train_size=None):
    # Use first fold as test, remaining as train
    train_x, train_y, test_x, test_y = get_train_test(full_data, 0)

    if max_train_size is None:
        max_train_size = len(train_x)

    step_sizes = list(range(step_size, max_train_size + 1, step_size))
    results = [("TrainingSize", "CostJ")]

    for size in step_sizes:
        current_train_x = train_x[:size]
        current_train_y = train_y[:size]

        nn = NeuralNet(layers, reg_lambda, current_train_x, current_train_y, alpha)

        for epoch in range(1000):  # fixed m = 1000
            nn.back_propagation()

        predicted_labels = []
        for x in test_x:
            activations, _ = nn.forward_propagation(x)
            predicted_labels.append(activations[-1])

        _, total_cost = nn.get_error(predicted_labels, test_y)
        results.append((size, total_cost))
        print(f"Training size: {size}, Cost J on test set: {total_cost:.5f}")

    # Plotting the learning curve from the results list
    training_sizes = [row[0] for row in results[1:]]
    costs = [row[1] for row in results[1:]]

    plt.figure(figsize=(8, 6))
    plt.plot(training_sizes, costs, marker='o')
    plt.xlabel("Number of Training Examples")
    plt.ylabel("Cost J on Test Set")
    plt.title(f"Learning Curve on {dataset_name} (α = {alpha})")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def preprocess_digits():
    # Loading the digits dataset
    digits = load_digits()
    X = digits.data  # Shape: (1797, 64)
    y = digits.target.reshape(-1, 1)  # Shape: (1797, 1)

    data = [list(X[i]) + [int(y[i])] for i in range(len(X))]

    data = shuffle_data(data)

    return data

def digits_nn():
    alpha = 0.1
    data_rows = preprocess_digits()

    data = get_final_data_from_rows(data_rows, total_classes=10, num_folds=10)

    input_dim = 64  # 64 pixels per image
    output_dim = 10  # 10 classes
    print('Training neural network for digits data')

    layers1 = [input_dim, 32, 16, output_dim]
    reg_lambda1 = 0.1
    metrics1 = test_neural_net(layers1, reg_lambda1, alpha, data, output_dim)
    print(metrics1)

    layers2 = [input_dim, 24, 12, output_dim]
    reg_lambda2 = 0.25
    metrics2 = test_neural_net(layers2, reg_lambda2, alpha, data, output_dim)
    print(metrics2)

    layers3 = [input_dim, 16, output_dim]
    reg_lambda3 = 0.05
    metrics3 = test_neural_net(layers3, reg_lambda3, alpha, data, output_dim)
    print(metrics3)

    layers4 = [input_dim, 64, 32, 16, output_dim]
    reg_lambda4 = 0.01
    metrics4 = test_neural_net(layers4, reg_lambda4, alpha, data, output_dim)
    print(metrics4)

    layers5 = [input_dim, 48, 24, output_dim]
    reg_lambda5 = 0.5
    metrics5 = test_neural_net(layers5, reg_lambda5, alpha, data, output_dim)
    print(metrics5)

    layers6 = [input_dim, 56, 28, output_dim]
    reg_lambda6 = 0.75
    metrics6 = test_neural_net(layers6, reg_lambda6, alpha, data, output_dim)
    print(metrics6)

    metrics = [metrics1, metrics2, metrics3, metrics4, metrics5, metrics6]
    headers = ["Accuracy", "F1-Score"]

    # Save metrics to CSV
    with open("digits-metrics.csv", 'w') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(metrics)


def parkinsons():
    alpha = 0.1  # Learning rate alpha
    data = get_final_data("parkinsons.csv", ",", 2, False, num_folds=10)  # 22 features, 2 classes

    print('Training neural network for parkinsons data')
    layers1 = [22, 16, 8, 2]
    reg_lambda1 = 0.1
    metrics1 = test_neural_net(layers1, reg_lambda1, alpha, data, 2)
    print(metrics1)

    layers2 = [22, 12, 6, 2]
    reg_lambda2 = 0.25
    metrics2 = test_neural_net(layers2, reg_lambda2, alpha, data, 2)
    print(metrics2)

    layers3 = [22, 10, 2]
    reg_lambda3 = 0.05
    metrics3 = test_neural_net(layers3, reg_lambda3, alpha, data, 2)
    print(metrics3)

    layers4 = [22, 20, 10, 5, 2]
    reg_lambda4 = 0.01
    metrics4 = test_neural_net(layers4, reg_lambda4, alpha, data, 2)
    print(metrics4)

    layers5 = [22, 25, 10, 2]
    reg_lambda5 = 0.5
    metrics5 = test_neural_net(layers5, reg_lambda5, alpha, data, 2)
    print(metrics5)

    layers6 = [22, 30, 15, 2]
    reg_lambda6 = 0.75
    metrics6 = test_neural_net(layers6, reg_lambda6, alpha, data, 2)
    print(metrics6)

    metrics = [metrics1, metrics2, metrics3, metrics4, metrics5, metrics6]
    headers = ["Accuracy", "F1-Score"]

    with open("parkinsons-metrics.csv", 'w') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(metrics)


def preprocess_rice_data(file_path):
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        raw_data = list(reader)

    processed_data = []

    for row in raw_data:
        new_row = []

        for i in range(7):
            new_row.append(float(row[i]))

        # Convert the label to numeric (Cammeo -> 0, Osmancik -> 1)
        label = 0 if row[7] == "Cammeo" else 1
        new_row.append(label)

        processed_data.append(new_row)

    return processed_data


def rice():
    alpha = 0.2
    raw_data = preprocess_rice_data("rice.csv")
    data = get_final_data_from_rows(raw_data, 2, num_folds=10)  # 7 features, 2 classes

    print('Training neural network for rice data')
    layers1 = [7, 18, 12, 2]
    reg_lambda1 = 0.1
    metrics1 = test_neural_net(layers1, reg_lambda1, alpha, data, 2)
    print(metrics1)

    layers2 = [7, 14, 8, 2]
    reg_lambda2 = 0.2
    metrics2 = test_neural_net(layers2, reg_lambda2, alpha, data, 2)
    print(metrics2)

    layers3 = [7, 10, 5, 2]
    reg_lambda3 = 0.05
    metrics3 = test_neural_net(layers3, reg_lambda3, alpha, data, 2)
    print(metrics3)

    layers4 = [7, 22, 15, 8, 2]
    reg_lambda4 = 0.01
    metrics4 = test_neural_net(layers4, reg_lambda4, alpha, data, 2)
    print(metrics4)

    # Reduced number of neurons in the 5th and 6th runs
    layers5 = [7, 20, 10, 2]
    reg_lambda5 = 0.3
    metrics5 = test_neural_net(layers5, reg_lambda5, alpha, data, 2)
    print(metrics5)

    layers6 = [7, 25, 12, 2]
    reg_lambda6 = 0.5
    metrics6 = test_neural_net(layers6, reg_lambda6, alpha, data, 2)
    print(metrics6)

    metrics = [metrics1, metrics2, metrics3, metrics4, metrics5, metrics6]
    headers = ["Accuracy", "F1-Score"]

    with open("rice-metrics.csv", 'w') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(metrics)


def preprocess_credit_data(file_path):
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        raw_data = list(reader)

    processed_data = []

    # Updated indices
    cat_indices = [0, 3, 4, 5, 6, 8, 9, 11, 12]
    num_indices = [1, 2, 7, 10, 13, 14]

    # Create one-hot encodings
    one_hot_maps = {}
    for i in cat_indices:
        unique_vals = sorted(set(row[i] for row in raw_data))
        one_hot_maps[i] = {val: idx for idx, val in enumerate(unique_vals)}

    for row in raw_data:
        new_row = []

        # One-hot encoding categorical features
        for i in cat_indices:
            one_hot = [0.0] * len(one_hot_maps[i])
            if row[i] in one_hot_maps[i]:
                one_hot[one_hot_maps[i][row[i]]] = 1.0
            new_row.extend(one_hot)

        # Numeric values features
        for i in num_indices:
            new_row.append(float(row[i]))

        # Label
        new_row.append(float(row[-1]))

        processed_data.append(new_row)

    return processed_data

def credit_approval():
    alpha = 0.05  # Learning rate alpha
    raw_data = preprocess_credit_data("credit_approval.csv")
    data = get_final_data_from_rows(raw_data, 2, num_folds=10)

    input_dim = len(raw_data[0]) - 1
    print('input dimension is: ' + str(input_dim))

    print('Training neural network for credit approval data')
    layers1 = [46, 16, 8, 2]
    reg_lambda1 = 0.1
    metrics1 = test_neural_net(layers1, reg_lambda1, alpha, data, 2)
    print(metrics1)

    layers2 = [46, 12, 6, 2]
    reg_lambda2 = 0.25
    metrics2 = test_neural_net(layers2, reg_lambda2, alpha, data, 2)
    print(metrics2)

    layers3 = [46, 10, 2]
    reg_lambda3 = 0.05
    metrics3 = test_neural_net(layers3, reg_lambda3, alpha, data, 2)
    print(metrics3)

    layers4 = [46, 20, 10, 5, 2]
    reg_lambda4 = 0.01
    metrics4 = test_neural_net(layers4, reg_lambda4, alpha, data, 2)
    print(metrics4)

    layers5 = [46, 18, 10, 2]
    reg_lambda5 = 0.5
    metrics5 = test_neural_net(layers5, reg_lambda5, alpha, data, 2)
    print(metrics5)

    layers6 = [46, 22, 12, 2]
    reg_lambda6 = 0.75
    metrics6 = test_neural_net(layers6, reg_lambda6, alpha, data, 2)
    print(metrics6)

    metrics = [metrics1, metrics2, metrics3, metrics4, metrics5, metrics6]
    headers = ["Accuracy", "F1-Score"]

    with open("credit-approval-metrics.csv", 'w') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(metrics)

if __name__ == "__main__":

    # Training the neural network for digits dataset
    digits_nn()
    best_layers_digits = [64, 56, 28, 10]
    best_lambda_digits = 0.75
    alpha_digits = 0.1
    data_rows = preprocess_digits()
    digits_data = get_final_data_from_rows(data_rows, total_classes=10, num_folds=10)
    generate_learning_curve(best_layers_digits, best_lambda_digits, alpha_digits, digits_data, 10, 'digits-dataset', 100)

    # Training the neural network for parkinson's dataset
    parkinsons()
    best_layers_parkinsons = [22, 10, 2]
    best_lambda_parkinsons = 0.05
    alpha_parkinsons = 0.1
    parkinsons_data = get_final_data("parkinsons.csv", ",", 2, False, num_folds=10)
    generate_learning_curve(best_layers_parkinsons, best_lambda_parkinsons, alpha_parkinsons, parkinsons_data, 2, 'parkinsons-dataset', 10)


    # Training the neural network for rice dataset
    rice()
    best_layers_rice = [7, 18, 12, 2]
    best_lambda_rice = 0.1
    alpha_rice = 0.2
    raw_data = preprocess_rice_data("rice.csv")
    rice_data = get_final_data_from_rows(raw_data, 2, num_folds=10)
    generate_learning_curve(best_layers_rice, best_lambda_rice, alpha_rice, rice_data, 2, 'rice-dataset', 200)

    # Training the neural network for credit approval dataset
    credit_approval()
    best_layers_credit = [46, 16, 8, 2]
    best_lambda_credit = 0.1
    alpha_credit = 0.05
    raw_data = preprocess_credit_data("credit_approval.csv")
    credit_data = get_final_data_from_rows(raw_data, 2, num_folds=10)
    generate_learning_curve(best_layers_credit, best_lambda_credit, alpha_credit, credit_data, 2, 'credit-approval-dataset', 20)