import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
from rf import RandomForestClassifier, calculate_metrics
from sklearn.datasets import load_digits

def kfold_stratified(df, dataset_name, k, ntree_values, depths):
    def stratify_data(df):
        classes = df.iloc[:, -1].unique()
        splits = {
            c: df[df.iloc[:, -1] == c].sample(frac=1, random_state=0).reset_index(drop=True)
            for c in classes
        }

        return classes, splits, {c: len(splits[c]) // k for c in classes}

    def get_fold_data(classes, splits, fold_sizes, fold):
        train_parts, test_parts = [], []

        for c in classes:
            start, end = fold * fold_sizes[c], (fold + 1) * fold_sizes[c]
            test_parts.append(splits[c].iloc[start:end])
            train_parts.append(pd.concat([splits[c].iloc[:start], splits[c].iloc[end:]]))
        train_df = pd.concat(train_parts).reset_index(drop=True)
        test_df = pd.concat(test_parts).reset_index(drop=True)

        return train_df, test_df

    def evaluate_model(train_df, test_df, ntree_values, depths):
        fold_results = {n: {'accuracy': [], 'f1': []} for n in ntree_values}

        for i, trees in enumerate(ntree_values):
            print(f"[{dataset_name}] Fold {fold + 1}/{k} - ntree={trees}, depth={depths[i]}")
            rf = RandomForestClassifier(trees, depths[i])
            rf.fit(train_df)
            preds = rf.predict_labels(test_df)
            accuracy, f1 = calculate_metrics(np.array(preds), test_df.iloc[:, -1].values)
            fold_results[trees]['accuracy'].append(accuracy)
            fold_results[trees]['f1'].append(f1)

        return fold_results

    results = {n: {'accuracy': [], 'f1': []} for n in ntree_values}
    classes, splits, fold_sizes = stratify_data(df)

    for fold in range(k):
        train_df, test_df = get_fold_data(classes, splits, fold_sizes, fold)
        fold_results = evaluate_model(train_df, test_df, ntree_values, depths)
        for trees in ntree_values:
            results[trees]['accuracy'].extend(fold_results[trees]['accuracy'])
            results[trees]['f1'].extend(fold_results[trees]['f1'])

    return {
        trees: {
            'accuracy': np.mean(results[trees]['accuracy']),
            'f1': np.mean(results[trees]['f1'])
        } for trees in ntree_values
    }

def plot_metrics(metrics, ntree_values, name):
    accs = [metrics[trees]['accuracy'] for trees in ntree_values]
    f1s  = [metrics[trees]['f1'] for trees in ntree_values]
    plt.figure()
    plt.plot(ntree_values, accs, marker='o', label='Accuracy')
    plt.plot(ntree_values, f1s, marker='s', label='F1 Score')
    plt.title(f"{name} - Accuracy & F1 vs Number of Trees")
    plt.xlabel('Number of Trees')
    plt.ylabel('Accuracy/F1 Score')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{name}_rf_metrics.png")
    plt.close()

def shuffle_data(data):
    random.shuffle(data)
    return data

if __name__ == '__main__':
    ntree_values = [3, 5, 10, 15, 20, 25]
    depths = [10, 8, 7, 6, 4, 3]

    # load digits dataset
    digits = load_digits()
    X = digits.data
    y = digits.target.reshape(-1, 1)
    data = [list(X[i]) + [int(y[i])] for i in range(len(X))]
    data = shuffle_data(data)
    cols = [f"pixel{i}" for i in range(X.shape[1])] + ['target']
    df_d = pd.DataFrame(data, columns=cols)

    # load other datasets
    df_p = pd.read_csv('parkinsons.csv')
    df_r = pd.read_csv('rice.csv')
    df_c = pd.read_csv('credit_approval.csv').dropna().reset_index(drop=True)

    datasets = [
        (df_d, 'Digits'),
        (df_c, 'Credit Approval'),
        (df_p, 'Parkinsons'),
        (df_r, 'Rice'),
    ]

    for df, dataset_name in datasets:
        metrics = kfold_stratified(df, dataset_name, 10, ntree_values, depths)

        for trees, metrics_val in metrics.items():
            print(f"Trees={trees}, Acc={metrics_val['accuracy']:.3f}, F1={metrics_val['f1']:.3f}")

        plot_metrics(metrics, ntree_values, dataset_name)
