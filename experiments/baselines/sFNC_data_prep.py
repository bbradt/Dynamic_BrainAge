import scipy.io
import mat73
import numpy as np
import pandas as pd

triu = np.triu_indices(53,1)

# HCP
data = mat73.loadmat('/data/qneuromark/Results/SFNC/HCP/sFNC_HCP_REST1_LR.mat')
subjects = np.array([y[0].split("/")[7] for y in data['analysis_subjlist_finished']])
sFNC = np.array([y[triu] for y in data['sFNC']])

HCP_R1_LR = pd.DataFrame(sFNC)
HCP_R1_LR["Subject"] = subjects
HCP_R1_LR["dataset"] = np.array(["HCP"]*len(HCP_R1_LR))
HCP_R1_LR["session"] = np.array(["REST1_LR"]*len(HCP_R1_LR))

data = mat73.loadmat('/data/qneuromark/Results/SFNC/HCP/sFNC_HCP_REST1_RL.mat')
subjects = np.array([y[0].split("/")[7] for y in data['analysis_subjlist_finished']])
sFNC = np.array([y[triu] for y in data['sFNC']])

HCP_R1_RL = pd.DataFrame(sFNC)
HCP_R1_RL["Subject"] = subjects
HCP_R1_RL["dataset"] = np.array(["HCP"]*len(HCP_R1_RL))
HCP_R1_RL["session"] = np.array(["REST1_RL"]*len(HCP_R1_RL))

data = mat73.loadmat('/data/qneuromark/Results/SFNC/HCP/sFNC_HCP_REST2_LR.mat')
subjects = np.array([y[0].split("/")[7] for y in data['analysis_subjlist_finished']])
sFNC = np.array([y[triu] for y in data['sFNC']])

HCP_R2_LR = pd.DataFrame(sFNC)
HCP_R2_LR["Subject"] = subjects
HCP_R2_LR["dataset"] = np.array(["HCP"]*len(HCP_R2_LR))
HCP_R2_LR["session"] = np.array(["REST2_LR"]*len(HCP_R2_LR))

data = mat73.loadmat('/data/qneuromark/Results/SFNC/HCP/sFNC_HCP_REST2_RL.mat')
subjects = np.array([y[0].split("/")[7] for y in data['analysis_subjlist_finished']])
sFNC = np.array([y[triu] for y in data['sFNC']])

HCP_R2_RL = pd.DataFrame(sFNC)
HCP_R2_RL["Subject"] = subjects
HCP_R2_RL["dataset"] = np.array(["HCP"]*len(HCP_R2_RL))
HCP_R2_RL["session"] = np.array(["REST2_RL"]*len(HCP_R2_RL))

HCP = pd.concat([HCP_R1_LR, HCP_R1_RL, HCP_R2_LR, HCP_R2_RL])

del HCP_R1_LR
del HCP_R1_RL
del HCP_R2_LR
del HCP_R2_RL

# HCP Aging
data = mat73.loadmat('/data/qneuromark/Results/SFNC/HCP_aging/HCP_aging_SFNC.mat')
subjects = np.array([y[0].split("/")[7].split("_")[0] for y in data['analysis_subjlist_finished']])
sessions = np.array([y[0].split("/")[9] for y in data['analysis_subjlist_finished']])
sFNC = np.array([y[triu] for y in data['sFNC']])

del data

HCP_aging = pd.DataFrame(sFNC)
HCP_aging["Subject"] = subjects
HCP_aging["dataset"] = np.array(["HCP_Aging"]*len(HCP_aging))
HCP_aging["session"] = sessions

# UKBiobank
data = mat73.loadmat('/data/qneuromark/Results/SFNC/UKBioBank/UKB_SFNC.mat')
subjects = np.array([y[0].split("/")[8] for y in data['analysis_subjlist_finished']])
sessions = np.array([y[0].split("/")[9] for y in data['analysis_subjlist_finished']])
sFNC = np.array([y[triu] for y in data['sFNC']])

del data

UKB = pd.DataFrame(sFNC)
UKB["Subject"] = subjects
UKB["dataset"] = np.array(["UKB"]*len(UKB))
UKB["session"] = sessions

all_sFNC = pd.concat([HCP, HCP_aging, UKB])

all_data = pd.read_csv("/data/users2/mduda/scripts/brainAge/HCP_HCPA_UKB_age_filepaths_cogScores2.csv")

merged = pd.merge(all_data, all_sFNC, how = 'left', on = ['Subject', 'dataset', 'session'])

merged.to_pickle('sFNC_df.pkl')