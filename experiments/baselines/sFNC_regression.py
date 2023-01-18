import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from joblib import dump, load
from sklearn.metrics import mean_absolute_error
import statsmodels.api as sm
import statsmodels.formula.api as smf

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
svr = GridSearchCV(
    SVR(kernel="rbf", gamma=0.1),
    param_grid={"C": [1e0, 1e1, 1e2, 1e3], "gamma": np.logspace(-2, 2, 5)},
    cv=kfold.split(np.arange(train_data_size))
)

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