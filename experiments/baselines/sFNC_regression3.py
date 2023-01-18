import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from joblib import dump, load
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.metrics import mean_absolute_error
from statsmodels.stats.multitest import fdrcorrection as fdr

# seed that was used during randomization of data in dataloader
sorting_seed = 319

#seed that was used during kfold CV
kfold_seed = 314159

# sizes of training and validation sets
train_data_size = 10000
valid_data_size = 5885

# sort  data to match dataloader randomization
sFNC_data = pd.read_pickle("sFNC_df.pkl")
np.random.seed(sorting_seed)
idxs = np.random.permutation(len(sFNC_data))

sFNC_data_sort = sFNC_data.loc[idxs]
sFNC_data_sort = sFNC_data_sort.reset_index(drop=True)


# Model training
np.random.seed(kfold_seed)
kfold = KFold(n_splits = 10, shuffle=True, random_state = kfold_seed)

#svr = SVR(kernel="rbf", C=100, gamma=0.1, epsilon=0.1)
svr = SVR(kernel="rbf", gamma=0.01, C = 100)

train_data = sFNC_data_sort.iloc[:train_data_size, :38]
train_data["CV_pred"] = 0

test_data = sFNC_data_sort.iloc[train_data_size:, :38]

X_test = sFNC_data_sort.iloc[train_data_size:, 38:]
y_test = sFNC_data_sort.loc[train_data_size:, "age"]

corrs_valid = []; corrs_test = []; mae_valid = []; mae_test = []
for k, (train_idx, valid_idx) in enumerate(kfold.split(np.arange(train_data_size))):
    label = "fold_%s"%k
    X_train = sFNC_data_sort.iloc[train_idx, 38:]
    X_valid = sFNC_data_sort.iloc[valid_idx, 38:]
    y_train = sFNC_data_sort.loc[train_idx, "age"]
    y_valid = sFNC_data_sort.loc[valid_idx, "age"]
    svr.fit(X_train, y_train)
    pred = svr.predict(X_valid)
    train_data.loc[valid_idx, "CV_pred"] = pred
    corrs_valid.append(np.corrcoef(pred, y_valid)[0,1])
    mae = mean_absolute_error(y_valid, pred)
    mae_valid.append(mae)
    test_pred = svr.predict(X_test)
    test_data[label] = test_pred
    corrs_test.append(np.corrcoef(test_pred, y_test)[0,1])
    mae = mean_absolute_error(y_test, test_pred)
    mae_test.append(mae)

dump(svr, 'svr_C100_gamma01.joblib') 

test_data["PBA_mean"] = np.mean(test_data.iloc[:,-10:],1)
test_data.to_csv("/data/users2/mduda/scripts/brainAge/LSTM_BrainAge/experiments/baselines/testSet_SVR_C100_gamma01_predictions.csv", index = False)

# print("Validation CV Corr:"); print(corrs_valid)
print("Validation CV Avg Corr: %s" %np.mean(corrs_valid))
# print("Validation CV MAE:"); print(mae_valid)
print("Validation CV Avg MAE: %s" %np.mean(mae_valid))

# print("Testing CV Corr:"); print(corrs_test)
print ("Testing Avg Corr: %s" %np.mean(corrs_test))
# print("Testing CV MAE:"); print(corrs_test)
print ("Testing Avg MAE: %s" %np.mean(mae_test))

# Bias Correction

from sklearn.linear_model import LinearRegression
X = np.array(test_data["age"]).reshape(-1,1)
y = np.array(test_data["PBA_mean"])
reg = LinearRegression().fit(X, y)
alpha = reg.coef_[0]
beta = reg.intercept_

# Beheshti et al. (2019), de Lange et al. (2019b)
test_data["PBA_corrected1"] = test_data["PBA_mean"] + (test_data["age"] - (alpha*test_data["age"] + beta))
test_data["BA_delta_corrected1"] = test_data["PBA_corrected1"] - test_data["age"]

# Cole et al. (2018)
test_data["PBA_corrected2"] = (test_data["PBA_mean"] - beta)/alpha 
test_data["BA_delta_corrected2"] = test_data["PBA_corrected2"] - test_data["age"]

pbacorr = test_data.corr().loc["age","PBA_corrected1"]
print(f"Corr(PBA_corrected1, age): {pbacorr:.3f}")

mae = mean_absolute_error(test_data["age"], test_data["PBA_mean"])
print(f"PBA MAE: {mae:.3f}")

mae = mean_absolute_error(test_data["age"], test_data["PBA_corrected1"])
print(f"PBA_corrected1 MAE: {mae:.3f}")

fields = [6,7]+list(np.arange(8,33,2))+[33,35,37]

cogLM_results = pd.DataFrame(test_data.columns[fields],columns = ['Measure'])
cogLM_results["pVal"] = 0; cogLM_results["tVal"] = 0; cogLM_results["coef"] = 0
cogLM_results["R2"] = 0; cogLM_results["R2adj"] = 0; cogLM_results["Fstat"] = 0; cogLM_results["Fstat_pVal"] = 0

for i, field in enumerate(fields):
    meas = cogLM_results.loc[i,"Measure"]
    results = smf.ols('%s ~ age + BA_delta_corrected1 + sex'%meas, data = test_data).fit()
    cogLM_results.loc[i, "pVal"] = results.pvalues['BA_delta_corrected1']
    cogLM_results.loc[i, "tVal"] = results.tvalues['BA_delta_corrected1']
    cogLM_results.loc[i, "coef"] = results.params['BA_delta_corrected1']
    cogLM_results.loc[i, "R2"] = results.rsquared
    cogLM_results.loc[i, "R2adj"] = results.rsquared_adj
    cogLM_results.loc[i, "Fstat"] = results.fvalue
    cogLM_results.loc[i, "Fstat_pVal"] = results.f_pvalue


rejected, pVal_adj = fdr(cogLM_results.pVal)

cogLM_results["significant"] = rejected
cogLM_results["pVal_FDR"] = pVal_adj


X_train = sFNC_data_sort.iloc[:train_data_size, 7:]
y_train = sFNC_data_sort.loc[:train_data_size-1, "age"]

X_test = sFNC_data_sort.iloc[train_data_size:, 7:]
y_test = sFNC_data_sort.loc[train_data_size:, "age"]

validation_data = sFNC_data_sort.iloc[train_data_size:,:]

svr.fit(X_train, y_train)
print(f"Best SVR with params: {svr.best_params_} and R2 score: {svr.best_score_:.3f}")
dump(svr, 'svr.joblib') 

test_pred = svr.predict(X_test)
validation_data["PBA"] = test_pred
test_corr = np.corrcoef(test_pred, y_test)[0,1]
mae = mean_absolute_error(y_test, test_pred)
print(f"Testing Corr: {test_corr:.3f}")
print(f"Testing MAE: {mae:.3f}")

# bias correction
from sklearn.linear_model import LinearRegression
X = np.array(validation_data["age"]).reshape(-1,1)
y = np.array(validation_data["PBA"])
reg = LinearRegression().fit(X, y)
alpha = reg.coef_[0]
beta = reg.intercept_

# Beheshti et al. (2019), de Lange et al. (2019b)
validation_data["PBA_corrected1"] = validation_data["PBA"] + (validation_data["age"] - (alpha*validation_data["age"] + beta))
validation_data["BA_delta_corrected1"] = validation_data["PBA_corrected1"] - validation_data["age"]

# Cole et al. (2018)
validation_data["PBA_corrected2"] = (validation_data["PBA"] - beta)/alpha 
validation_data["BA_delta_corrected2"] = validation_data["PBA_corrected2"] - validation_data["age"]

print(validation_data.loc[:,["age","fluidCog","fluidCog_scaled","PBA","PBA_corrected1","BA_delta_corrected1","PBA_corrected2","BA_delta_corrected2"]].corr())


# corrs_valid = []; corrs_test = []
# for k, (train_idx, valid_idx) in enumerate(kfold.split(np.arange(train_data_size))):
#     X_train = sFNC_data_sort.iloc[train_idx, 7:]
#     X_valid = sFNC_data_sort.iloc[valid_idx, 7:]
#     y_train = sFNC_data_sort.loc[train_idx, "age"]
#     y_valid = sFNC_data_sort.loc[valid_idx, "age"]
#     svr.fit(X_train, y_train)
#     pred = svr.predict(X_valid)
#     corrs_valid.append(np.corrcoef(pred, y_valid)[0,1])
#     test_pred = svr.predict(X_test)
#     corrs_test.append(np.corrcoef(test_pred, y_test)[0,1])

# print("Training CV Corr:"); print(corrs_valid)
# print("Training CV Avg Corr: %s" %np.mean(corrs_valid))

# print("Training CV Corr:"); print(corrs_test)
# print ("Testing Avg Corr: %s" %np.mean(corrs_test))