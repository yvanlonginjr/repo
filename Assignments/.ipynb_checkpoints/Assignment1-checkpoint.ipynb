{
 "cells": [
  {
   "cell_type": "raw",
   "id": "08096616-7e52-493a-b724-6c88f561ab89",
   "metadata": {},
   "source": [
    "Part 1"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "fbd227fd-fd02-4684-88f6-88705432f28d",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "   normal_1  normal_2  uniform_1  uniform_2\n",
      "0  0.304717  5.352554   0.362763   2.302144\n",
      "1 -1.039984  6.798247   0.942406   2.944811\n",
      "2  0.750451  2.055459   0.255202  -0.083615\n",
      "3  0.940565  4.429404   0.834961  -0.239142\n",
      "4 -1.951035  6.658307   0.896856  -1.018130\n",
      "           normal_1      normal_2     uniform_1     uniform_2\n",
      "count  10000.000000  10000.000000  10000.000000  10000.000000\n",
      "mean      -0.010250      5.040703      0.502009      0.004288\n",
      "std        1.006336      2.005878      0.288007      1.735452\n",
      "min       -4.389115     -2.775609      0.000047     -2.999835\n",
      "25%       -0.677280      3.674732      0.253509     -1.516076\n",
      "50%       -0.013168      5.018178      0.501470      0.026644\n",
      "75%        0.649971      6.390794      0.754817      1.502683\n",
      "max        4.025824     13.302483      0.999982      2.999455\n"
     ]
    }
   ],
   "source": [
    "import numpy as np\n",
    "import pandas as pd\n",
    "\n",
    "# Reproducibility\n",
    "rng = np.random.default_rng(42)\n",
    "n = 10_000\n",
    "\n",
    "# Two normal, two uniform\n",
    "df = pd.DataFrame({\n",
    "    \"normal_1\": rng.normal(loc=0, scale=1, size=n),     # N(0,1)\n",
    "    \"normal_2\": rng.normal(loc=5, scale=2, size=n),     # N(5, 2^2)\n",
    "    \"uniform_1\": rng.uniform(low=0.0, high=1.0, size=n),# U[0,1]\n",
    "    \"uniform_2\": rng.uniform(low=-3.0, high=3.0, size=n)# U[-3,3]\n",
    "})\n",
    "\n",
    "# quick peek & summary\n",
    "print(df.head())\n",
    "print(df.describe())\n",
    "\n",
    "# save to disk\n",
    "df.to_csv(\"random_dataset.csv\", index=False)\n"
   ]
  },
  {
   "cell_type": "raw",
   "id": "129f21bc-8633-4515-ba83-a5abf6a424fc",
   "metadata": {},
   "source": [
    "Part 2"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "a8126692-097d-4aaf-a11b-60eee7fdeca4",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "   normal_1  normal_2  uniform_1  uniform_2         y\n",
      "0  0.304717  5.352554   0.362763   2.302144  0.742517\n",
      "1 -1.039984  6.798247   0.942406   2.944811  0.350154\n",
      "2  0.750451  2.055459   0.255202  -0.083615 -0.124345\n",
      "3  0.940565  4.429404   0.834961  -0.239142 -1.446170\n",
      "4 -1.951035  6.658307   0.896856  -1.018130 -6.026546\n",
      "           normal_1      normal_2     uniform_1     uniform_2             y\n",
      "count  10000.000000  10000.000000  10000.000000  10000.000000  10000.000000\n",
      "mean      -0.010250      5.040703      0.502009      0.004288     -2.678764\n",
      "std        1.006336      2.005878      0.288007      1.735452      3.762917\n",
      "min       -4.389115     -2.775609      0.000047     -2.999835    -13.022128\n",
      "25%       -0.677280      3.674732      0.253509     -1.516076     -5.556518\n",
      "50%       -0.013168      5.018178      0.501470      0.026644     -2.685474\n",
      "75%        0.649971      6.390794      0.754817      1.502683      0.077140\n",
      "max        4.025824     13.302483      0.999982      2.999455     14.992842\n"
     ]
    }
   ],
   "source": [
    "a1, a2, a3, a4, a5 = 1.2, -0.7, 0.5, 1.8, 0.6\n",
    "\n",
    "# Small, mean-zero noise\n",
    "noise_std = 0.2\n",
    "epsilon = rng.normal(loc=0.0, scale=noise_std, size=n)\n",
    "\n",
    "# y = a1*x1 + a2*x2 + a3*x3 + a4*x4 + a5*(x1^2) + epsilon\n",
    "df[\"y\"] = (\n",
    "    a1 * df[\"normal_1\"]\n",
    "    + a2 * df[\"normal_2\"]\n",
    "    + a3 * df[\"uniform_1\"]\n",
    "    + a4 * df[\"uniform_2\"]\n",
    "    + a5 * (df[\"normal_1\"] ** 2)  # squared term\n",
    "    + epsilon\n",
    ")\n",
    "\n",
    "# Quick check & save\n",
    "print(df.head())\n",
    "print(df.describe().loc[:, [\"normal_1\",\"normal_2\",\"uniform_1\",\"uniform_2\",\"y\"]])\n"
   ]
  },
  {
   "cell_type": "raw",
   "id": "eacd4db6-0549-4cea-a90a-08cc7d6b14f7",
   "metadata": {},
   "source": [
    "Part 3"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "e73a079d-edb6-43e7-bcc7-cfd9eedb0b57",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(7000, 5) (3000, 5)\n"
     ]
    }
   ],
   "source": [
    "import numpy as np\n",
    "\n",
    "rng = np.random.default_rng(42)\n",
    "idx = np.arange(len(df))\n",
    "rng.shuffle(idx)\n",
    "\n",
    "cut = int(0.7 * len(df))\n",
    "train_idx, test_idx = idx[:cut], idx[cut:]\n",
    "\n",
    "train_df = df.iloc[train_idx].reset_index(drop=True)\n",
    "test_df  = df.iloc[test_idx].reset_index(drop=True)\n",
    "\n",
    "print(train_df.shape, test_df.shape)\n"
   ]
  },
  {
   "cell_type": "raw",
   "id": "d1fe243d-0393-4f69-98e2-3fa6c24e372e",
   "metadata": {},
   "source": [
    "Part 4"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "a248f8fa-baa8-46d6-8eb8-a03d1a73cc6a",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Intercept: 0.5769065132960417\n",
      "Coefficients:\n",
      "normal_1     1.181553\n",
      "normal_2    -0.695526\n",
      "uniform_1    0.519359\n",
      "uniform_2    1.806909\n",
      "dtype: float64\n",
      "\n",
      "MSE (train): 0.814215\n",
      "MSE (test):  0.754439\n"
     ]
    }
   ],
   "source": [
    "import numpy as np\n",
    "import pandas as pd\n",
    "\n",
    "# --- Assumes you already have df with columns:\n",
    "# normal_1, normal_2, uniform_1, uniform_2, y\n",
    "\n",
    "# 1) 70/30 split (reproducible)\n",
    "features = [\"normal_1\", \"normal_2\", \"uniform_1\", \"uniform_2\"]\n",
    "rng = np.random.default_rng(42)\n",
    "idx = np.arange(len(df))\n",
    "rng.shuffle(idx)\n",
    "cut = int(0.7 * len(df))\n",
    "train_idx, test_idx = idx[:cut], idx[cut:]\n",
    "\n",
    "train_df = df.iloc[train_idx].reset_index(drop=True)\n",
    "test_df  = df.iloc[test_idx].reset_index(drop=True)\n",
    "\n",
    "X_train = train_df[features].to_numpy()\n",
    "y_train = train_df[\"y\"].to_numpy()\n",
    "X_test  = test_df[features].to_numpy()\n",
    "y_test  = test_df[\"y\"].to_numpy()\n",
    "\n",
    "# 2) Add intercept column\n",
    "X_train_aug = np.c_[np.ones(X_train.shape[0]), X_train]\n",
    "X_test_aug  = np.c_[np.ones(X_test.shape[0]),  X_test]\n",
    "\n",
    "# 3) OLS via least squares (more stable than explicit inverse)\n",
    "# beta = argmin ||X_train_aug * beta - y_train||_2\n",
    "beta, residuals, rank, s = np.linalg.lstsq(X_train_aug, y_train, rcond=None)\n",
    "\n",
    "intercept = beta[0]\n",
    "coefs = pd.Series(beta[1:], index=features)\n",
    "\n",
    "print(\"Intercept:\", intercept)\n",
    "print(\"Coefficients:\")\n",
    "print(coefs)\n",
    "\n",
    "# 4) Predictions & MSE\n",
    "yhat_train = X_train_aug @ beta\n",
    "yhat_test  = X_test_aug  @ beta\n",
    "\n",
    "mse_train = np.mean((y_train - yhat_train)**2)\n",
    "mse_test  = np.mean((y_test - yhat_test)**2)\n",
    "\n",
    "print(f\"\\nMSE (train): {mse_train:.6f}\")\n",
    "print(f\"MSE (test):  {mse_test:.6f}\")\n"
   ]
  },
  {
   "cell_type": "raw",
   "id": "9b1155d1-13ea-4512-aa4e-86d80d28bee4",
   "metadata": {},
   "source": [
    "Part 5"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "cefabc9c-48de-4195-9694-90ba733fade7",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "   normal_1  normal_2  uniform_1  uniform_2         y\n",
      "0 -0.053783  5.429782   0.809397   2.762215  1.511815\n",
      "1  1.059119  2.209087   0.226584   0.052329  0.500495\n",
      "2  0.172031  2.133818   0.941006   2.475254  3.791499\n",
      "3 -2.394260  4.193929   0.985119   2.191138  2.088677\n",
      "4 -1.206998  4.882123   0.945837  -2.762352 -8.434375\n"
     ]
    }
   ],
   "source": [
    "import numpy as np\n",
    "import pandas as pd\n",
    "\n",
    "# Base data to bootstrap from:\n",
    "# use df (all data) or train_df (training split) depending on your workflow\n",
    "base_df = df            # or: base_df = train_df\n",
    "n_boot = 10\n",
    "seed = 123\n",
    "\n",
    "def make_bootstrap_samples(dataframe, n_boot=10, seed=None):\n",
    "    rng = np.random.default_rng(seed)\n",
    "    n = len(dataframe)\n",
    "    samples = []\n",
    "    for b in range(n_boot):\n",
    "        idx = rng.integers(0, n, size=n)   # sample with replacement\n",
    "        boot_df = dataframe.iloc[idx].reset_index(drop=True)\n",
    "        samples.append(boot_df)\n",
    "    return samples\n",
    "\n",
    "boot_samples = make_bootstrap_samples(base_df, n_boot=n_boot, seed=seed)\n",
    "\n",
    "# Example: inspect the first bootstrap sample\n",
    "print(boot_samples[0].head())"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "db437fd9-11b2-43f9-8451-769a59cbcc0c",
   "metadata": {},
   "source": [
    "Part 6"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "id": "c76f375b-c87f-48a7-a507-edd6760b9e29",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "   replicate  intercept  normal_1  normal_2  uniform_1  uniform_2\n",
      "0          1   0.589363  1.164715 -0.700149   0.513322   1.800083\n",
      "1          2   0.530093  1.164982 -0.685513   0.500998   1.811133\n",
      "2          3   0.529094  1.211673 -0.688826   0.542098   1.800773\n",
      "3          4   0.586900  1.200677 -0.702407   0.537246   1.802607\n",
      "4          5   0.532613  1.180780 -0.692151   0.539696   1.798151\n",
      "5          6   0.545060  1.189225 -0.690864   0.511819   1.808773\n",
      "6          7   0.562282  1.157778 -0.693906   0.528728   1.793894\n",
      "7          8   0.571696  1.183093 -0.690911   0.469246   1.800420\n",
      "8          9   0.584451  1.197121 -0.695892   0.484817   1.801039\n",
      "9         10   0.551229  1.196622 -0.693327   0.518166   1.798855\n"
     ]
    }
   ],
   "source": [
    "features = [\"normal_1\", \"normal_2\", \"uniform_1\", \"uniform_2\"]\n",
    "target = \"y\"\n",
    "\n",
    "def fit_ols_return_beta(df_sample, features, target=\"y\"):\n",
    "    X = df_sample[features].to_numpy()\n",
    "    y = df_sample[target].to_numpy()\n",
    "    # add intercept column\n",
    "    X_aug = np.c_[np.ones(X.shape[0]), X]\n",
    "    # OLS via least squares\n",
    "    beta, residuals, rank, s = np.linalg.lstsq(X_aug, y, rcond=None)\n",
    "    return beta  # [intercept, coef1, coef2, coef3, coef4]\n",
    "\n",
    "# Fit per bootstrap sample\n",
    "rows = []\n",
    "for i, bdf in enumerate(boot_samples, start=1):\n",
    "    beta = fit_ols_return_beta(bdf, features, target)\n",
    "    row = {\"replicate\": i, \"intercept\": beta[0]}\n",
    "    row.update({feat: beta[j+1] for j, feat in enumerate(features)})\n",
    "    rows.append(row)\n",
    "\n",
    "coef_boot = pd.DataFrame(rows)\n",
    "print(coef_boot)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "6c9b2442-24f7-424d-8f1d-561acab6c5b6",
   "metadata": {},
   "source": [
    "Part 7"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "id": "ab6cd5f7-bbfa-4d2a-a939-fe18035ecd2b",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "   parameter      mean       std\n",
      "0  intercept  0.558278  0.024027\n",
      "1   normal_1  1.184666  0.017735\n",
      "2   normal_2 -0.693395  0.005061\n",
      "3  uniform_1  0.514614  0.024151\n",
      "4  uniform_2  1.801573  0.005021\n"
     ]
    }
   ],
   "source": [
    "import pandas as pd\n",
    "\n",
    "# coef_boot: columns like ['replicate','intercept','normal_1','normal_2','uniform_1','uniform_2', ...]\n",
    "params = [c for c in coef_boot.columns if c != \"replicate\"]\n",
    "\n",
    "boot_stats = (\n",
    "    coef_boot[params]\n",
    "    .agg(['mean', 'std'])\n",
    "    .T\n",
    "    .rename_axis('parameter')\n",
    "    .reset_index()\n",
    ")\n",
    "\n",
    "print(boot_stats)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "b1269636-1542-4bab-b814-1f3021c6ce4c",
   "metadata": {},
   "source": [
    "Part 8"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "061e5b3a-55d7-4d6e-844f-7a66c4e333c7",
   "metadata": {},
   "source": [
    "All coefficients are consistent with the bootstrap: four lie within 1 SD of the bootstrap mean (the fifth is ~1.06 SD away), and every estimate falls inside its 95% Confidence Interval. The bootstrap standard deviations are smal,especially for normal_2 and uniform_2 which implies precise estimates."
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python [conda env:machine_learning]",
   "language": "python",
   "name": "conda-env-machine_learning-py"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.13"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
