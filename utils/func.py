from sklearn.model_selection import KFold
from sklearn.utils import resample
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import copy
from scipy import spatial

def bootstrap_kfold_cv(X, y, n_splits=10, random_state=None):

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for train_index, test_index in kf.split(X):
        X_train_resampled, y_train_resampled = resample(
            X[train_index], y[train_index], replace=True, random_state=random_state
        )
        yield X_train_resampled, y_train_resampled, X[test_index], y[test_index]

def error(i, ensemble_proba, selected_models, target):
    iproba = ensemble_proba[i, :, :]
    sub_proba = ensemble_proba[selected_models, :, :]
    pred = 1.0 / (1 + len(sub_proba)) * (sub_proba.sum(axis=0) + iproba)

def predict_with_subforest(trees, X):

    all_preds = np.array([tree.predict(X) for tree in trees])
    maj_vote = np.apply_along_axis(lambda x: np.bincount(x.astype(int)).argmax(), axis=0, arr=all_preds)

    return maj_vote

def create_subforest_from_estimators(rf: RandomForestClassifier, selected_estimators: list):

    sub_rf = copy.deepcopy(rf)
    sub_rf.estimators_ = selected_estimators
    sub_rf.n_estimators = len(selected_estimators)

    return sub_rf

def find(parent, i):
    if parent[i] == i:
        return i
    return find(parent, parent[i])


def union(parent, rank, x, y):
    root_x = find(parent, x)
    root_y = find(parent, y)
    if rank[root_x] < rank[root_y]:
        parent[root_x] = root_y
    elif rank[root_x] > rank[root_y]:
        parent[root_y] = root_x
    else:
        parent[root_y] = root_x
        rank[root_x] += 1


def generate_tree(matrix, mode='max', seed=42):
    if mode == 'random':
        np.random.seed(seed)
    V = matrix.shape[0]

    matrix_tree = np.zeros((V, V))


    parent = []
    rank = []
    for node in range(V):
        parent.append(node)
        rank.append(0)


    edges = []
    if mode == 'random':

        for i in range(V):
            for j in range(i + 1, V):
                edges.append((np.random.uniform(0, 1), i, j))
    else:

        for i in range(V):
            for j in range(i + 1, V):
                if matrix[i, j] != 0:
                    edges.append((matrix[i, j], i, j))

    if mode == 'max':
        edges.sort(reverse=True, key=lambda item: item[0])
    elif mode == 'min':
        edges.sort(key=lambda item: item[0])
    elif mode == 'random':
        np.random.shuffle(edges)


    result = []
    for edge in edges:
        weight, u, v = edge
        root_u = find(parent, u)
        root_v = find(parent, v)

        if root_u != root_v:
            result.append((u, v, weight))
            union(parent, rank, root_u, root_v)

            if len(result) == V - 1:
                break


    for u, v, weight in result:
        matrix_tree[u, v] = weight
        matrix_tree[v, u] = weight

    return matrix_tree


def generate_features(n):
    features = [f'f{i}' for i in range(n)]

    all_features = set(features)
    all_features = sorted(all_features)
    return all_features

# OO
def reference_vector(i, ensemble_proba, target):

    ref = 2 * (ensemble_proba.mean(axis=0).argmax(axis=1) == target) - 1.0
    ipred = 2 * (ensemble_proba[i, :].argmax(axis=1) == target) - 1.0
    return 1.0 - spatial.distance.cosine(ref, ipred)
