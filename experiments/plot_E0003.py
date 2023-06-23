import os
import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pd
import json
from functools import reduce
import numpy as np


def factors(n):
    return set(reduce(list.__add__,
                      ([i, n//i] for i in range(1, int(n**0.5) + 1) if n % i == 0)))


output_path = os.path.join("/data", "users2", "bbaker",
                           "projects", "cadasil_analysis", "figures")
experiment_path = os.path.join("/data", "users2", "bbaker", "projects",
                               "cadasil_analysis", "LSTM_BrainAge", "logs", "E0003_grid_search")

runs = 926

train_dfs = []
valid_dfs = []

for i in range(runs):
    run_path = os.path.join(experiment_path, "run_%d" % i)
    if not os.path.exists(run_path):
        continue
    train_path = os.path.join(run_path, "csv_logger", "train.csv")
    valid_path = os.path.join(run_path, "csv_logger", "valid.csv")
    param_path = os.path.join(run_path, "parameters.json")
    if not os.path.exists(train_path) or not os.path.exists(valid_path):
        continue
    train_df = pd.read_csv(train_path)
    valid_df = pd.read_csv(valid_path)
    params = json.load(open(param_path, "r"))
    for k, v in params.items():
        train_df[k] = v
        valid_df[k] = v
    train_dfs.append(train_df)
    valid_dfs.append(valid_df)

full_train_df = pd.concat(train_dfs).reset_index(drop=True)
full_valid_df = pd.concat(valid_dfs).reset_index(drop=True)

# Optim Kwargs, batch-size, lr
unique_hiddens = full_valid_df['model_kwargs'].unique()

fig, ax = plt.subplots(1, 1, figsize=(10, 10))

r = 0
c = 0
for i, unq in enumerate(unique_hiddens):
    if i == 5:
        continue
    if c > 1:
        c = 0
        r += 1
    print(i, r, c)
    #axi = ax[r, c]
    axi = ax
    unq_df = valid_df[valid_df['model_kwargs'] == unq]
    unq_bat = unq_df['batch_size'].unique()
    unq_lr = unq_df['lr'].unique()
    result = np.zeros((len(unq_bat), len(unq_lr)))
    stds = []
    for ii, bat in enumerate(unq_bat):
        match = unq_df[unq_df['batch_size'] == bat]
        stds.append([])
        for jj, lr in enumerate(unq_lr):
            match = match[match['lr'] == lr]
            std = []
            for k in match['k'].unique():
                match = match[match['k'] == k]
                best = np.max(match['correlation'])
                result[ii, jj] += best
                std.append(best)
            result[ii, jj] /= len(match['k'].unique())
            stds[-1].append("%f += %f" % (result[ii, jj], np.std(std)))
    print("Done with unq %s" % unq)
    print(stds)
    sb.heatmap(result, annot=stds, ax=axi,
               xticklabels=unq_bat, yticklabels=unq_lr)
    c += 1
    axi.set_title("%s" % unq)
plt.savefig(os.path.join(output_path, "grid_results.png"), bbox_inches="tight")
