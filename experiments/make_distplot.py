import os
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt


 #### Test Set
predictions = '/data/users2/mduda/scripts/brainAge/LSTM_BrainAge/experiments/testSet_phenotypes_predictions.csv'

pred_df = pd.read_csv(predictions)

pred_df = pred_df.sort_values(by=["age","BA_delta_corrected1" ]).reset_index()
pred_df['index'] = range(len(pred_df))

sb.set()

sb.set_style("whitegrid")
sb.set_context("paper")

fig, ax = plt.subplots(1, 1)

g = sb.jointplot(data=pred_df, x="PBA_corrected1",
                 y="index", ax=ax, hue="dataset", alpha = 0.2, palette = "Set2")
sb.scatterplot(data=pred_df, x="age", y="index", ax=g.ax_joint, color="gray", edgecolor=None, s = 5)
plt.ylim([-1, 5890])
plt.xlabel("Predicted Brain Age")
plt.ylabel("")
plt.savefig('/data/users2/mduda/scripts/brainAge/LSTM_BrainAge/logs/E0003_ukbhcp1200valid_inference/distplot_pba_testSet.png', bbox_inches="tight", dpi=300)



#### Validation set
predictions = '/data/users2/mduda/scripts/brainAge/LSTM_BrainAge/experiments/validationSet_phenotypes_predictions.csv'

pred_df = pd.read_csv(predictions)

pred_df = pred_df.sort_values(by=["age","BA_delta_corrected1" ]).reset_index()
pred_df['index'] = range(len(pred_df))

sb.set()
sb.set_style("whitegrid")
sb.set_context("paper")
fig, ax = plt.subplots(1, 1)

# CADASIL_Subject_Type


g = sb.jointplot(data=pred_df, x="PBA_corrected1",
                 y="index", ax=ax, hue="dataset", alpha = 0.2, palette = "Set2")
sb.scatterplot(data=pred_df, x="age", y="index", ax=g.ax_joint, color="gray", edgecolor=None, s = 5)
plt.ylim([-1, 10005])
ax.set(xlabel='Predicted Brain Age', ylabel='')

plt.savefig('/data/users2/mduda/scripts/brainAge/LSTM_BrainAge/logs/E0003_ukbhcp1200valid_inference/distplot_pba_validationSet.png', bbox_inches="tight", dpi=300)

print("Lorem ipsum")