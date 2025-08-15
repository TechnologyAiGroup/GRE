# GRE Ensemble Pruning 

## PyPruning 

Repository:
https://github.com/sbuschjaeger/PyPruning/blob/master/Readme.md

You can install it directly via:

    pip install git+https://github.com/sbuschjaeger/PyPruning.git

Some modifications are required.

The main component used is RankPruningClassifier, which supports various pruning algorithms. The metric parameter allows you to change the pruning strategy. For example:

    metric = RankPruningClassifier.individual_contribution

The default metric is individual_error.

### Modifications to the PyPruning Package

1. In the prune function of PruningClassifier:
Add the following line at line 119:

    self.selected_indices_ = idx

2. In the individual_contribution function of RankPruningClassifier:
Modify line 95 from:

    IC = IC + (V[j, target[j]]  - V[j, predictions[j]] - np.max(V[j,:]) )

to:

    target_j_index = int(target[j])
    predictions_j_index = int(predictions[j])
    IC = IC + (V[j, target_j_index] - V[j, predictions_j_index] - np.max(V[j, :]))

3. In the individual_margin_diversity function of RankPruningClassifier:
Add the following line at line 31:

    target = target.astype(int)

## HGCN

Repository:
https://github.com/HazyResearch/hgcn.git

Clone the repository into the \Lib\site-packages directory of your Python environment.

### Modifications to the HGCN Package

1. In base_models, starting from line 77:


    def compute_metrics(self, embeddings, data, idx_list):
        idx = torch.tensor(list(range(idx_list)), dtype=torch.long)
        output = self.decode(embeddings, data['adj_train_norm'], idx)
        loss = F.nll_loss(output, data['labels'][idx], self.weights)
        acc, f1 = acc_f1(output, data['labels'][idx], average=self.f1_average)
        metrics = {'loss': loss, 'acc': acc, 'f1': f1}
        return metrics

## HyperLib Package

Install via:
    
    pip install hyperlib

## Other Packeges

python 3.9

torch 2.1

scikit-learn 1.3 

xgboost 2.0.3


## Running the Main Script

The main functionality is implemented in main.py. You can configure the ensemble pruning process using command-line arguments.

### Example Usage

    python main.py --ensemble_model random_forest --dataset_name taiwan_bankrupt.csv --rounds 3 --num_trees 200 --n_prune 15 --baseline EPIC

## Parameters

| Argument  | Type  | Default                 | Description                                                                                                                                                                                                                                        |
| --------- | ----- | ----------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `ensemble_model` | `str` | `'random_forest'`       | Type of ensemble model to prune. Options: `'random_forest'`, `'bagging'`, `'xgboost'`.                                                                                                                                                             |
| `dataset_name` | `str` | `'taiwan_bankrupt.csv'` | Dataset to use. Options include: <ul><li>`taiwan_bankrupt.csv`</li><li>`diabetes.csv`</li><li>`magic04.data`</li><li>`HTRU_2.csv`</li><li>`connect-4.data`</li><li>`bank-additional.csv`</li><li>`diabetes1.csv`</li><li>`spambase.data`</li></ul> |
| `rounds`  | `int` | `3`                     | Number of rounds in 10-fold cross-validation.                                                                                                                                                                                                      |
| `num_trees` | `int` | `200`                   | Number of base learners in the ensemble.                                                                                                                                                                                                           |
| `n_prune` | `int` | `15`                    | Number of base learners to retain after pruning (selected from the top half).                                                                                                                                                                      |
| `baseline` | `str` | `'EPIC'`                | Baseline pruning algorithm. Options: `'EPIC'`, `'MDEP'`. Ignored when using OO in XGBoost.                                                                                                                                                         |

