import argparse
from models.gcn import GCNModel
from utils.func import bootstrap_kfold_cv
from utils.func import generate_tree
from utils.func import reference_vector
from utils.func import generate_features
from utils.func import predict_with_subforest
from data_loader import load_dataset
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from datetime import datetime
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from PyPruning import RankPruningClassifier
from sklearn.tree import export_text
import hgcn.optimizers as optimizers
import networkx as nx
from scipy.special import logit
from sklearn.model_selection import KFold
import xgboost as xgb
from hgcn.models.base_models import NCModel
from hyperlib.embedding.sarkar import sarkar_embedding
import re



def main(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Loading dataset: {args.dataset_name}")
    X, y, num_features, num_classes = load_dataset(args.dataset_name)

    # result
    acc_original = []
    acc_gnn = []
    acc_base_50 = []
    acc_base_100 = []
    acc_base_200 = []
    snum = []
    runtime1 = []
    runtime2 = []
    tree_num = args.num_trees
    half_num = tree_num // 2

    if args.ensemble_model == 'random_forest' or args.ensemble_model == 'bagging':

        if args.baseline == 'EPIC':
            base_metric = RankPruningClassifier.individual_contribution
        else:
            base_metric = RankPruningClassifier.individual_margin_diversity


        for r in range(args.rounds):
            kf = bootstrap_kfold_cv(X, y, n_splits=10, random_state=67) # 10-fold
            k = 0
            for X_train, y_train, X_test, y_test in kf:
                k += 1
                print(f"Round {r * 10 + k}")

                if args.ensemble_model == 'random_forest':
                    rf_classifier = RandomForestClassifier(n_estimators=200, random_state=8)

                else:
                    rf_classifier = BaggingClassifier(estimator=DecisionTreeClassifier(criterion='entropy'),
                                                      n_estimators=tree_num, max_samples=0.5, random_state=42)

                rf_classifier.fit(X_train, y_train)


                y_pred = rf_classifier.predict(X_test)

                original_accuracy = accuracy_score(y_test, y_pred)

                # GRE runtime calculation
                start_time = datetime.now()

                max_feature_index = num_features - 1
                feature_counts_per_tree = []
                for i, tree_in_forest in enumerate(rf_classifier.estimators_):
                    tree_text = export_text(tree_in_forest)
                    lines = tree_text.split('\n')
                    feature_counts = [0] * (max_feature_index + 1)
                    for line in lines:
                        if 'feature_' in line:
                            feature = int(line.split('feature_')[1].split()[0])
                            feature_counts[feature] += 1
                    feature_counts_per_tree.append(feature_counts)

                tree_predictions = []
                for tree in rf_classifier.estimators_:
                    tree_predictions.append(tree.predict(X_test))
                tree_predictions = np.array(tree_predictions)

               
                matrix = np.zeros((len(rf_classifier.estimators_), len(rf_classifier.estimators_)))
                for i in range(len(rf_classifier.estimators_)):
                    for j in range(i + 1, len(rf_classifier.estimators_)):
                        matrix[i, j] = np.sum(np.abs(tree_predictions[j] - tree_predictions[i]))
                        matrix[j, i] = matrix[i, j]


                matrix = matrix / (matrix.max() - matrix.min())

                known_node_indices = list(range(half_num))


                known_labels = torch.full((tree_num,), -1, dtype=torch.long).to(device)

                all_estimators = rf_classifier.estimators_

                n_trees_to_use = min(half_num, len(all_estimators))
                selected_estimators = all_estimators[:n_trees_to_use]


                model = GCNModel(num_features, 16, num_classes).to(device)

                n_prune = args.n_prune
                pruned_model = RankPruningClassifier.RankPruningClassifier(
                    n_prune,
                    metric = base_metric)
                pruned_model.prune(X_train, y_train, selected_estimators)
                idx = pruned_model.selected_indices_

                for i in range(100):
                    if i in idx:
                        known_labels[i] = 1
                    else:
                        known_labels[i] = 0

                pruned_pred = pruned_model.predict(X_test)
                pruned_100_accuracy = accuracy_score(y_test, pruned_pred)

                matrix = torch.tensor(matrix).to(device)
                feature_counts_per_tree = torch.tensor(feature_counts_per_tree).to(device)

                feature_counts_per_tree = feature_counts_per_tree.float()
                matrix = matrix.float()
                optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
                for epoch in range(tree_num):
                    model.train()
                    optimizer.zero_grad()
                    output = model(torch.tensor(feature_counts_per_tree[:half_num], dtype=torch.float),
                                   torch.tensor(matrix[:half_num, :half_num], dtype=torch.float))
                    loss = torch.nn.functional.cross_entropy(output, torch.tensor(y_train[:half_num], dtype=torch.long))
                    loss.backward()
                    optimizer.step()


                model.eval()
                output = model(feature_counts_per_tree[half_num:].to(device),
                               matrix[half_num:, half_num:].to(device))
                result = output[:, 0] - output[:, 1]
                average_diff = result.mean()
                bias = torch.tensor([0.0, average_diff - 0.0015]).to(device)
                output = output + bias
                predicted_labels = torch.argmax(output, dim=1)

                for i in range(half_num):
                    known_labels[i + half_num] = predicted_labels[i]


                selected_trees = [i for i, label in enumerate(predicted_labels) if label == 1]
                selected_rf_trees = [rf_classifier.estimators_[i] for i in selected_trees]
                
                gnn_pred = predict_with_subforest(selected_rf_trees, X_test)
                gnn_accuracy = accuracy_score(y_test, gnn_pred)

                end_time = datetime.now()
                elapsed_time = (end_time - start_time).total_seconds()
                runtime1.append(elapsed_time)

                print(f"Original Accuracy: {original_accuracy:.2f}%, GRE Accuracy: {gnn_accuracy:.2f}%")

                # baseline runtime
                start_time = datetime.now()
                pruned_model = RankPruningClassifier.RankPruningClassifier(
                    len(selected_rf_trees),
                    metric=base_metric)
                pruned_model.prune(X_train, y_train, all_estimators)
                pruned_pred = pruned_model.predict(X_test)
                pruned_200_accuracy = accuracy_score(y_test, pruned_pred)
                end_time = datetime.now()
                elapsed_time = (end_time - start_time).total_seconds()
                runtime2.append(elapsed_time)
                print(f"Baseline Runtime: {elapsed_time:.2f} seconds")


                acc_original.append(100.0 * original_accuracy)
                acc_base_100.append(100.0 * pruned_100_accuracy)
                acc_gnn.append(100.0 * gnn_accuracy)
                acc_base_200.append(100.0 * pruned_200_accuracy)
                snum.append(len(selected_trees))

        print()
        print()
        print(f"Original Accuracy: {np.mean(acc_original):.2f} ± {np.std(acc_original):.2f}%")
        print(f"GRE Accuracy: {np.mean(acc_gnn):.2f} ± {np.std(acc_gnn):.2f}%")
        print(f"Baseline Accuracy: {np.mean(acc_base_200):.2f} ± {np.std(acc_base_200):.2f}%")
        print(f"Average GRE Runtime: {np.mean(runtime1):.2f} seconds")
        print(f"Average Baseline Runtime: {np.mean(runtime2):.2f} seconds")


    elif args.ensemble_model == 'xgboost':

        class Args_hgcn:
            def __init__(self, **kwargs):
                self.manifold = 'PoincareBall'
                self.c = 0.5
                self.cuda = -1
                self.device = 'cuda:0' if self.cuda != -1 else 'cpu'
                self.n_nodes = 3
                self.num_layers = 16
                self.act = "relu"
                self.task = "lp"
                self.dropout = 0.2
                self.bias = 0
                self.use_att = 1
                self.local_agg = 0
                self.r = 2
                self.t = 1
                self.pretrained_embeddings = None
                self.pos_weight = 0
                self.n_heads = 4
                self.alpha = 0.2
                self.double_precision = 0
                self.n_classes = 2
                for key, value in kwargs.items():
                    setattr(self, key, value)

        all_features = generate_features(num_features)
        for r in range(1):


            kf = KFold(n_splits=10, shuffle=True, random_state=67)
            k = 0
            for train_index, test_index in kf.split(X):
                k += 1
                print("Round" + str(r * 10 + k) )
                X_train, X_test = X[train_index], X[test_index]
                Y_train, Y_test = y[train_index], y[test_index]
                dtrain = xgb.DMatrix(X_train, label=Y_train)
                dtest = xgb.DMatrix(X_test, Y_test)
                dtest_only = xgb.DMatrix(X_test)

                base_score = 0.5
                E1_dim = 2
                E2_dim = 15
                E3_dim = E1_dim + E2_dim


                params = {
                    "max_depth": 15,
                    "eta": 0.25,
                    "objective": "reg:logistic",

                    "base_score": base_score,
                    'gamma': 0.1,
                    'min_child_weight': 5,
                }

                xg_classifier = xgb.train(params, dtrain, num_boost_round=tree_num)
                trees_dump = xg_classifier.get_dump()

                full = xg_classifier.predict(xgb.DMatrix(X_test), output_margin=False)
                original_accuracy = ((full > 0.5) == Y_test).sum() / len(Y_test)


                start_t1 = datetime.now()

                max_feature_index = num_features - 1
                feature_counts_per_tree = []

                for tree in trees_dump:
                    feature_counts = {feature: 0 for feature in all_features}
                    splits = re.split(r'\n\t*\d+:+leaf=', tree)
                    for split in splits:
                        feature_match = re.search(r'\[(\w+)<\d+(\.\d+)?\]', split)
                        if feature_match:
                            feature = feature_match.group(1)
                            if feature in all_features:
                                feature_counts[feature] += 1

                    current_tree_counts = [feature_counts.get(feature, 0) for feature in all_features]

                    feature_counts_per_tree.append(current_tree_counts)


                tree_predictions = []
                last = []
                for i in range(tree_num):
                    Dtest = xgb.DMatrix(X_test, base_margin=last)
                    if i == tree_num - 1:
                        tree_predictions.append(
                            xg_classifier.predict(Dtest, iteration_range=(i, i + 1), output_margin=False))
                    else:
                        # get raw leaf value for accumulation
                        last = xg_classifier.predict(Dtest, iteration_range=(i, i + 1), output_margin=True)
                        tree_predictions.append(
                            xg_classifier.predict(Dtest, iteration_range=(i, i + 1), output_margin=False))
                tree_predictions = np.array(tree_predictions)

                matrix = np.zeros((tree_num, tree_num))
                for i in range(tree_num):
                    for j in range(i + 1, tree_num):
                        matrix[i, j] = np.sum(np.abs(tree_predictions[j] - tree_predictions[i]))
                        matrix[j, i] = matrix[i, j]


                matrix = matrix / (matrix.max() - matrix.min())
                for i in range(tree_num):
                    matrix[i, i] = 0.0
                adj_matrix = generate_tree(matrix, mode='max', seed=105)



                tree = nx.from_numpy_array(adj_matrix, create_using=nx.Graph())
                rows, cols = np.nonzero(adj_matrix)
                for row, col in zip(rows, cols):
                    value = adj_matrix[row, col]


                root = 3  # precision 35
                embedding1 = sarkar_embedding(tree, root, tau=0.5)


                known_labels = torch.zeros(tree_num, dtype=torch.long)

                # OO
                ensemble_proba = np.zeros((tree_num, X_test.shape[0], 2))

                start_t2 = datetime.now()

                # ensemble_proba
                for i in range(tree_num):

                    iteration_range = (i, i + 1)

                    predictions_proba = xg_classifier.predict(dtest, iteration_range=iteration_range,
                                                              output_margin=False)
                    ensemble_proba[i, :, 1] = predictions_proba
                    ensemble_proba[i, :, 0] = 1 - predictions_proba

                end_t2 = datetime.now()


                target = Y_test
                target = np.array(target).flatten()

                similarities = []

                for i in range(half_num):
                    similarity = reference_vector(i, ensemble_proba, target)
                    similarities.append(similarity)

                n_prune = 15
                idx = np.argsort(similarities)[-n_prune:]
                known_labels[idx] = 1


                idx = idx.tolist()
                scores = np.full((X_test.shape[0],), logit(base_score))
                for i in idx:
                    i = int(i)
                    dtest = xgb.DMatrix(X_test, base_margin=scores)
                    if i == len(idx) - 1:
                        scores = xg_classifier.predict(
                            dtest, iteration_range=(i, i + 1), output_margin=False
                        )
                    else:
                        scores = xg_classifier.predict(
                            dtest, iteration_range=(i, i + 1), output_margin=True
                        )
                pruned_50_accuracy = ((scores > 0.5) == Y_test).sum() / len(Y_test)

                args1 = Args_hgcn(feat_dim=2, model='HNN', dim=6)
                known_node_indices = list(range(100))
                model = NCModel(args1)
                matrix = torch.tensor(matrix)

                feature_counts_per_tree = torch.tensor(feature_counts_per_tree)
                feature_counts_per_tree = feature_counts_per_tree.float()
                embedding1 = np.array(embedding1, ndmin=2)
                embedding1 = embedding1.astype(float)
                embedding1 = torch.tensor(embedding1, dtype=torch.float32)

                if embedding1.dim() == 2 and embedding1.size(0) == 1:
                    embedding1 = embedding1.view(-1)
                embedding1 = embedding1[:tree_num * 2]


                embedding1 = embedding1.view(tree_num, 2)

                matrix = matrix.float()

                optimizer = getattr(optimizers, "Adam")(model.parameters(), lr=0.005)
                lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.95)

                eval_freq = 20
                best_val_metrics = model.init_metric_dict()
                best_test_metrics = None
                best_emb = None
                data = {
                    'features': embedding1,
                    'labels': known_labels,
                    'adj_train_norm': matrix,
                }
                for epoch in range(tree_num):
                    model.train()
                    optimizer.zero_grad()
                    embeddings = model.encode(data['features'], data['adj_train_norm'])
                    train_metrics = model.compute_metrics(embeddings, data, 100)
                    train_metrics['loss'].backward()
                    optimizer.step()
                    lr_scheduler.step()

                E1 = model.encode(data['features'], data['adj_train_norm'])


                args2 = Args_hgcn(feat_dim=num_features, model='HNN', dim=E2_dim)
                model = NCModel(args2)
                matrix = torch.tensor(matrix)
                matrix = matrix.float()

                optimizer = getattr(optimizers, "Adam")(model.parameters(), lr=0.006)
                lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.95)


                data = {
                    'features': feature_counts_per_tree,
                    'labels': known_labels,
                    'adj_train_norm': matrix,
                }
                for epoch in range(tree_num):
                    model.train()
                    optimizer.zero_grad()
                    embeddings = model.encode(data['features'], data['adj_train_norm'])

                    train_metrics = model.compute_metrics(embeddings, data, 100)
                    train_metrics['loss'].backward()
                    optimizer.step()
                    lr_scheduler.step()

                E2 = model.encode(data['features'], data['adj_train_norm'])

                E3 = torch.cat((embedding1 * 10, E2 * 10), dim=1)

                adj_matrix = adj_matrix * 10
                rows, cols = np.nonzero(adj_matrix)
                for row, col in zip(rows, cols):
                    value = adj_matrix[row, col]


                args3 = Args_hgcn(feat_dim=E3_dim, model='HGCN', dim=16)
                model = NCModel(args3)
                adj_matrix = torch.tensor(adj_matrix)
                adj_matrix = adj_matrix.float()


                optimizer = getattr(optimizers, "Adam")(model.parameters(), lr=0.007)
                lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.95)

                eval_freq = 20
                best_val_metrics = model.init_metric_dict()
                best_test_metrics = None
                best_emb = None
                data = {
                    'features': E3,
                    'labels': known_labels,
                    'adj_train_norm': adj_matrix,
                }
                for epoch in range(tree_num):
                    model.train()
                    optimizer.zero_grad()
                    embeddings = model.encode(data['features'], data['adj_train_norm'])
                    # print("embeddings:")
                    # print(embeddings.shape)
                    train_metrics = model.compute_metrics(embeddings, data, 100)
                    train_metrics['loss'].backward(retain_graph=True)
                    optimizer.step()
                    lr_scheduler.step()

                embeddings = model.encode(data['features'], data['adj_train_norm'])
                idx = torch.arange(half_num, tree_num, dtype=torch.long)
                output = model.decode(embeddings, data['adj_train_norm'], idx)  
                result = output[:, 0] - output[:, 1]
                average_diff = result.mean()
                bias = torch.tensor([0.0, average_diff - 0.005])
                output = output + bias
                predicted_labels = torch.argmax(output, dim=1)


                from scipy.special import xlogy

                def calculate_entropy(probs):
                    return -xlogy(probs, probs) - xlogy(1 - probs, 1 - probs)


                for i in range(half_num):
                    known_labels[i + half_num] = predicted_labels[i]

                selected_trees = [i for i, label in enumerate(known_labels) if label == 1]
                scores = np.full((X_test.shape[0],), logit(base_score))
                for i in selected_trees:
                    dtest = xgb.DMatrix(X_test, base_margin=scores)
                    if i == len(selected_trees) - 1:
                        scores = xg_classifier.predict(
                            dtest, iteration_range=(i, i + 1), output_margin=False
                        )
                    else:
                        scores = xg_classifier.predict(
                            dtest, iteration_range=(i, i + 1), output_margin=True
                        )
                    probabilities1 = xg_classifier.predict(
                        dtest, iteration_range=(i, i + 1), output_margin=False
                    )
                gnn_accuracy = ((scores > 0.5) == Y_test).sum() / len(Y_test)

                end_t1 = datetime.now()
                elapsed_time = (end_t1 - start_t1).total_seconds()
                runtime1.append(elapsed_time)



                known_labels[half_num:tree_num] = 0

                start_time2 = datetime.now()


                similarities = []

                for i in range(tree_num):
                    similarity = reference_vector(i, ensemble_proba, target)
                    similarities.append(similarity)

                idx2 = np.argsort(similarities)[-len(selected_trees):]


                idx2 = idx2.tolist()
                scores = np.full((X_test.shape[0],), logit(base_score))
                for i in idx2:
                    i = int(i)
                    dtest = xgb.DMatrix(X_test, base_margin=scores)
                    if i == len(idx2) - 1:
                        scores = xg_classifier.predict(
                            dtest, iteration_range=(i, i + 1), output_margin=False
                        )
                    else:
                        scores = xg_classifier.predict(
                            dtest, iteration_range=(i, i + 1), output_margin=True
                        )

                    probabilities2 = xg_classifier.predict(
                        dtest, iteration_range=(i, i + 1), output_margin=False
                    )
                pruned_100_accuracy = ((scores > 0.5) == Y_test).sum() / len(Y_test)

                end_time2 = datetime.now()
                elapsed_time = (end_time2 - start_time2 + end_t2 - start_t2).total_seconds()

                runtime2.append(elapsed_time)

                print("Original Accuracy with {} estimators is {.2f} %".format(tree_num, 100.0 * original_accuracy))
                print("GRE-H Accuracy is {.2f} %".format(100.0 * gnn_accuracy))
                print("Baseline Accuracy is {.2f} %".format(100.0 * pruned_100_accuracy))


                snum.append(len(selected_trees))
                acc_original.append(100.0 * original_accuracy)
                acc_base_50.append(100.0 * pruned_50_accuracy)
                acc_gnn.append(100.0 * gnn_accuracy)
                acc_base_100.append(100.0 * pruned_100_accuracy)

        print()
        print()
        print(f"Original Accuracy: {np.mean(acc_original):.2f} ± {np.std(acc_original):.2f}%")
        print(f"GRE-H Accuracy: {np.mean(acc_gnn):.2f} ± {np.std(acc_gnn):.2f}%")
        print(f"Baseline Accuracy: {np.mean(acc_base_100):.2f} ± {np.std(acc_base_100):.2f}%")

        pass

    else:
        raise ValueError(f"Unknown ensemble: {args.ensemble_model}")


if __name__ == "__main__":
   
    parser = argparse.ArgumentParser()
    parser.add_argument('--ensemble_model', type=str, default='random_forest')
    parser.add_argument('--dataset_name', type=str, default='taiwan_bankrupt.csv', help="Name of the dataset to use")
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--num_trees', type=int, default=200)
    parser.add_argument('--n_prune', type=int, default=15)
    parser.add_argument('--baseline', type=str, default='EPIC')
    args = parser.parse_args()
    main(args)
