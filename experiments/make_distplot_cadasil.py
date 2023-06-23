import os
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt

predictions = '/data/users2/mduda/scripts/brainAge/LSTM_BrainAge/logs/E0002_cadasil_inference/cadasil_withControls_brainAgePredictions_biasCorrected.csv'

pred_df = pd.read_csv(predictions)
pred_df['index'] = range(len(pred_df))
pred_df.CDR_group.fillna("Control", inplace=True)

sb.set()
fig, ax = plt.subplots(1, 1)

# CADASIL_Subject_Type


g = sb.jointplot(data=pred_df, x="PBA_corrected1",
                 y="index", ax=ax, hue="CDR_group")
sb.scatterplot(data=pred_df, x="age", y="index", ax=g.ax_joint, color="gray")
plt.ylim([-1, 22])

plt.savefig('/data/users2/mduda/scripts/brainAge/LSTM_BrainAge/logs/E0005_cadasil_withControl_inference/distplot_pba.png', bbox_inches="tight")

print("Lorem ipsum")