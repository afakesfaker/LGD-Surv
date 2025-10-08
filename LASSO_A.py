import pandas as pd
import numpy as np
from sklearn.linear_model import LassoCV
from sklearn.linear_model import lasso_path
import matplotlib.pyplot as plt
import warnings

# Input data preprocess
print("Reading input files...")
gene_file = 'p_genedata1.csv'
survival_file = 'survivaldata.csv'
gene_data = pd.read_csv(gene_file, index_col=0)
survival_data = pd.read_csv(survival_file, index_col=0)
print(f"Loaded gene data: {gene_data.shape[0]} genes, {gene_data.shape[1]} samples")
print(f"Loaded survival data: {survival_data.shape[0]} metrics, {survival_data.shape[1]} samples")

gene_data.index = gene_data.index.str.replace('.', '_', regex=False)
print("Replaced '.' with '_' in gene IDs")

survival_times = survival_data.iloc[1, :].values  # row index 1 = 2nd row
invalid_mask = survival_times == -1
valid_mask = ~invalid_mask

print(f"Found {np.sum(invalid_mask)} samples with invalid survival time")
if np.any(invalid_mask):
    gene_data = gene_data.loc[:, valid_mask]
    survival_data = survival_data.loc[:, valid_mask]
    print("Removed invalid samples from both datasets")

print(f"Post-cleaning: {gene_data.shape[1]} samples remain")

# Prepare LASSO data
print("Preparing data for LASSO analysis...")
survival_status = survival_data.iloc[0, :].values
survival_times = survival_data.iloc[1, :].values
X = gene_data.T.values
n_samples, n_features = X.shape
print(f"Ready for LASSO: {n_samples} samples, {n_features} features")

nan_mask = np.isnan(survival_times)
if np.any(nan_mask):
    print(f"Warning: Found {np.sum(nan_mask)} NaN survival times. Removing those samples.")
    X = X[~nan_mask, :]
    survival_times = survival_times[~nan_mask]
    survival_status = survival_status[~nan_mask]

# LASSO with CV
print("Performing LASSO with cross-validation...")
n_folds = 10
if n_samples < n_folds:
    n_folds = n_samples
    warnings.warn(f"Sample count < folds; using {n_folds}-fold CV")

lasso_cv = LassoCV(cv=n_folds, random_state=12345, max_iter=5000).fit(X, survival_times)

lambda_min = lasso_cv.alpha_
print(f"Selected lambda (alpha) using lambda_min: {lambda_min:.6f}")

coef_min = lasso_cv.coef_
nonzero_min = np.sum(coef_min != 0)
print(f"Number of selected features at lambda_min: {nonzero_min}")

mse_mean = lasso_cv.mse_path_.mean(axis=1)
mse_std = lasso_cv.mse_path_.std(axis=1)
idx_min = np.argmin(mse_mean)
lambda_1se = lasso_cv.alphas_[np.where(mse_mean <= mse_mean[idx_min] + mse_std[idx_min])[0][-1]]
print(f"lambda_1se: {lambda_1se:.6f}")

# results
gene_ids = gene_data.index.values
selected_indices = np.where(coef_min != 0)[0]
selected_names = gene_ids[selected_indices]
selected_coefs = coef_min[selected_indices]

result_df = pd.DataFrame({
    'GeneID': selected_names,
    'Coefficient': selected_coefs
})

result_df = result_df.reindex(result_df.Coefficient.abs().sort_values(ascending=False).index)
print("Selected features (lambda_min):")
print(result_df)

result_df.to_csv('lasso_selected_features_lambda_min.csv', index=False)
print("Results saved to 'lasso_selected_features_lambda_min.csv'")

alphas_lasso, coefs_lasso, _ = lasso_path(X, survival_times, alphas=lasso_cv.alphas_)

# Figures
plt.figure(figsize=(5.5, 5.5))
for i in range(coefs_lasso.shape[0]):
    plt.plot(np.log(alphas_lasso), coefs_lasso[i, :], linewidth=0.8)
plt.axvline(np.log(lambda_min), color='k', linestyle='--', linewidth=2, label='lambda_min')
plt.axvline(np.log(lambda_1se), color='r', linestyle='--', linewidth=2, label='lambda_1se')
plt.xlabel('Log Lambda')
plt.ylabel('Coefficients')
plt.title('LASSO Coefficient Path')
plt.legend()
plt.tight_layout()
plt.savefig('lasso_coefficient_path.png', dpi=1200)

plt.figure(figsize=(5.5, 5.5))
plt.semilogx(lasso_cv.alphas_, mse_mean, linewidth=2, label='CV MSE')
plt.fill_between(lasso_cv.alphas_, mse_mean - mse_std, mse_mean + mse_std, alpha=0.2)
plt.axvline(lambda_min, color='k', linestyle='--', linewidth=2, label='lambda_min')
plt.axvline(lambda_1se, color='r', linestyle='--', linewidth=2, label='lambda_1se')
plt.xlabel('Lambda')
plt.ylabel('Cross-Validated MSE')
plt.title('LASSO Cross-Validation Error')
plt.legend()
plt.tight_layout()
plt.savefig('lasso_cv_error.png', dpi=1200)

print("LASSO analysis complete.")
