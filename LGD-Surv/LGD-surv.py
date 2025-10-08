# LGD-Surv (Deephit part)
# Input data should have been normalized

import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from pycox.models import DeepHitSingle
from pycox.evaluation import EvalSurv
import torchtuples as tt
import scipy.integrate

if not hasattr(scipy.integrate, "simps"):
    def simps(y, x=None, **kwargs):
        return scipy.integrate.simpson(y, x=x, **kwargs)
    scipy.integrate.simps = simps

torch.manual_seed(12345)
np.random.seed(12345)

# Load data
file_path = "processed_augmented_data.csv"
df = pd.read_csv(file_path, skiprows=1)

E = df.iloc[:, -2].values.astype(int)
T = df.iloc[:, -1].values.astype(float)
X = df.iloc[:, :-2].values

time_bins = pd.qcut(T, q=4, duplicates='drop', labels=False)
stratify_labels = E * 10 + time_bins

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def to_tensor(data, dtype=torch.float32):
    return torch.tensor(data, dtype=dtype).to(device)

# parameters （should be decided by grid search）
num_durations = 
num_nodes = 
dropout = 
alpha = 
sigma = 
batch_size = 
epochs = 5000
patience = 20
lr = 0.01
weight_decay = 0.01
n_splits = 5

c_index_list = []
brier_list = []
ibs_list = []

# 5-fold cross-validation
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=12345)
for fold, (train_idx, test_idx) in enumerate(skf.split(X, stratify_labels)):
    print(f"\n--- Fold {fold+1}/{n_splits} ---")

    X_train, X_test = X[train_idx], X[test_idx]
    T_train, T_test = T[train_idx], T[test_idx]
    E_train, E_test = E[train_idx], E[test_idx]

    print("  Train event ratio:", np.bincount(E_train), "Time mean:", np.mean(T_train))
    print("  Test event ratio :", np.bincount(E_test), "Time mean:", np.mean(T_test))

    val_size = int(len(X_train) * 0.2)
    X_val, T_val, E_val = X_train[:val_size], T_train[:val_size], E_train[:val_size]
    X_tr, T_tr, E_tr = X_train[val_size:], T_train[val_size:], E_train[val_size:]

    labtrans = DeepHitSingle.label_transform(num_durations)
    y_train = labtrans.fit_transform(T_tr, E_tr)
    y_val = labtrans.transform(T_val, E_val)
    y_test = labtrans.transform(T_test, E_test)

    X_tr_t, X_val_t, X_test_t = map(to_tensor, [X_tr, X_val, X_test])
    train_data = (X_tr_t, (to_tensor(y_train[0], torch.long), to_tensor(y_train[1])))
    val_data = (X_val_t, (to_tensor(y_val[0], torch.long), to_tensor(y_val[1])))

    in_features = X.shape[1]
    net = tt.practical.MLPVanilla(
        in_features, num_nodes, labtrans.out_features,
        dropout=dropout, activation=torch.nn.ELU
    )
    model = DeepHitSingle(
        net, tt.optim.Adam(lr=lr, weight_decay=weight_decay),
        duration_index=labtrans.cuts, alpha=alpha, sigma=sigma
    )

    callbacks = [tt.cb.EarlyStopping(patience=patience)]
    log = model.fit(*train_data, batch_size=batch_size, epochs=epochs,
                    callbacks=callbacks, val_data=val_data, verbose=False)

    model.net.eval()
    surv_test = model.predict_surv_df(X_test_t)
    ev_test = EvalSurv(surv_test, T_test, E_test, censor_surv='km')

    c_index = ev_test.concordance_td()
    c_index_list.append(c_index)

    times = np.linspace(T_test.min(), T_test.max(), 100)
    median_time = np.median(T_test)
    brier_scores = ev_test.brier_score(times)
    median_idx = np.argmin(np.abs(times - median_time))
    brier_list.append(brier_scores.iloc[median_idx])

    ibs = ev_test.integrated_brier_score(times)
    ibs_list.append(ibs)

    print(f"Fold {fold+1} Test C-index: {c_index:.4f}")
    print(f"Fold {fold+1} Brier Score@median: {brier_list[-1]:.4f}")
    print(f"Fold {fold+1} IBS: {ibs:.4f}")

print("\n=== Final Results ===")
print(f"Average C-index: {np.mean(c_index_list):.4f} ± {np.std(c_index_list):.4f}")
print(f"Average Brier Score@median: {np.mean(brier_list):.4f} ± {np.std(brier_list):.4f}")
print(f"Average IBS: {np.mean(ibs_list):.4f} ± {np.std(ibs_list):.4f}")
