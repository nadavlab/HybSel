import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.linear_model import ElasticNetCV, LassoCV
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import roc_curve, auc
from xgboost import XGBClassifier
import pickle
from datetime import datetime

#116823 #usecols=range(116000, 116825) #skiprows=2850 # 823
df = pd.read_csv('nalls_qc_cln_mls_3.csv')
X = df.iloc[:,1:116823]  # Features are all columns except the last one
y = df.iloc[:, -1]   # Labels are the last column
# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, stratify=y)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Lasso feature selection
lasso_cv = LassoCV(max_iter=100000, cv=5, random_state=42)
lasso_cv.fit(X_train_scaled, y_train)

feature_selector_lasso = SelectFromModel(lasso_cv, prefit=True)
X_train_selected_lasso = feature_selector_lasso.transform(X_train_scaled)
X_test_selected_lasso = feature_selector_lasso.transform(X_test_scaled)
selected_features_lasso = X.columns[feature_selector_lasso.get_support()]
selected_features_lasso_write=pd.DataFrame(selected_features_lasso)
selected_features_lasso_write.to_csv('selected_features_lasso_v5.csv', index=False)

# ElasticNet feature selection
alphas = [0.01]
enet_cv = ElasticNetCV(max_iter=1000000, random_state=42, l1_ratio=0.8, cv=5, tol=1e-2, alphas=alphas, selection='random')
enet_cv.fit(X_train_scaled, y_train)

feature_selector_enet = SelectFromModel(enet_cv, prefit=True)
X_train_selected_enet = feature_selector_enet.transform(X_train_scaled)
X_test_selected_enet = feature_selector_enet.transform(X_test_scaled)
selected_features_enet = X.columns[feature_selector_enet.get_support()]
selected_features_enet_write=pd.DataFrame(selected_features_enet)
selected_features_enet_write.to_csv('selected_features_enet_v5.csv', index=False)


#Hybsel
selected_columns = pd.read_csv('Hybsel_SNPs1.csv',header=None,lineterminator='\n')
selected_columns = selected_columns.iloc[:,0].str.strip()
selected_features_hybsel = pd.Index(selected_columns)

#X1 = df[selected_columns]
#y1 = df.iloc[:, -1] 
#X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, shuffle=False,test_size=0.2, stratify=y)
#X_train_scaled_hybsel = scaler.fit_transform(X1_train)
#X_test_scaled_hybsel = scaler.transform(X1_test)

X_train_scaled_hybsel = X_train[selected_columns]
X_test_scaled_hybsel = X_test[selected_columns]

selected_features_hybsel_write=pd.DataFrame(selected_features_hybsel)
selected_features_hybsel_write.to_csv('selected_features_hybsel_v5.csv', index=False)

#Hybsel_with_lasso
set_selected_features_lasso = set(selected_features_lasso)
set_selected_features_hybsel = set(selected_columns)
Hybsel_Lasso_columns = set_selected_features_lasso.union(set_selected_features_hybsel)
Hybsel_Lasso_features = list(Hybsel_Lasso_columns)
X_train_scaled_Hybsel_Lasso = X_train[Hybsel_Lasso_features]
X_test_scaled_Hybsel_Lasso = X_test[Hybsel_Lasso_features]
Hybsel_Lasso_features_write=pd.DataFrame(Hybsel_Lasso_features)
Hybsel_Lasso_features_write.to_csv('selected_features_hybsel_Lasso_v5.csv', index=False)


#Hybsel_n_lasso
set_selected_features_lasso = set(selected_features_lasso)
set_selected_features_hybsel = set(selected_columns)
Hybsel_n_lasso = set_selected_features_lasso.intersection(set_selected_features_hybsel)
Hybsel_n_lasso_features = list(Hybsel_n_lasso)
X_train_scaled_Hybsel_n_lasso = X_train[Hybsel_n_lasso_features]
X_test_scaled_Hybsel_n_lasso = X_test[Hybsel_n_lasso_features]
Hybsel_n_lasso_features_write=pd.DataFrame(Hybsel_n_lasso_features)
Hybsel_n_lasso_features_write.to_csv('selected_features_hybsel_n_Lasso_v5.csv', index=False)



params = {
    'objective': 'binary:logistic',  
    'colsample_bytree': 1.0, 
    'gamma': 0.1, 
    'learning_rate': 0.01, 
    'max_depth': 5, 
    'min_child_weight': 3, 
    'n_estimators': 200, 
    'subsample': 0.6
}


# XGBoost classifier # xgbClasssifier

boost_pre = XGBClassifier(**params)
boost_post_lasso = XGBClassifier(**params)
boost_post_enet = XGBClassifier(**params)
boost_post_hybsel = XGBClassifier(**params)
boost_post_hybsel_lasso = XGBClassifier(**params)
boost_post_hybsel_n_lasso = XGBClassifier(**params)

# Fit models
boost_pre.fit(X_train_scaled, y_train)
boost_post_lasso.fit(X_train_selected_lasso, y_train)
boost_post_enet.fit(X_train_selected_enet, y_train)
boost_post_hybsel.fit(X_train_scaled_hybsel, y_train)
boost_post_hybsel_lasso.fit(X_train_scaled_Hybsel_Lasso, y_train)
boost_post_hybsel_n_lasso.fit(X_train_scaled_Hybsel_n_lasso, y_train)


# Predict probabilities
y_proba_pre = boost_pre.predict_proba(X_test_scaled)[:, 1]
y_proba_post_lasso = boost_post_lasso.predict_proba(X_test_selected_lasso)[:, 1]
y_proba_post_enet = boost_post_enet.predict_proba(X_test_selected_enet)[:, 1]
y_proba_post_hybsel = boost_post_hybsel.predict_proba(X_test_scaled_hybsel)[:, 1]
y_proba_post_hybsel_lasso = boost_post_hybsel_lasso.predict_proba(X_test_scaled_Hybsel_Lasso)[:, 1]
y_proba_post_hybsel_n_lasso = boost_post_hybsel_n_lasso.predict_proba(X_test_scaled_Hybsel_n_lasso)[:, 1]


# Compute ROC curves and AUC scores
fpr_pre, tpr_pre, _ = roc_curve(y_test, y_proba_pre)
fpr_post_lasso, tpr_post_lasso, _ = roc_curve(y_test, y_proba_post_lasso)
fpr_post_enet, tpr_post_enet, _ = roc_curve(y_test, y_proba_post_enet)
fpr_post_hybsel, tpr_post_hybsel, _ = roc_curve(y_test, y_proba_post_hybsel)
fpr_post_hybsel_lasso, tpr_post_hybsel_lasso, _ = roc_curve(y_test, y_proba_post_hybsel_lasso)
fpr_post_hybsel_n_lasso, tpr_post_hybsel_n_lasso, _ = roc_curve(y_test, y_proba_post_hybsel_n_lasso)


roc_auc_pre = auc(fpr_pre, tpr_pre)
roc_auc_post_lasso = auc(fpr_post_lasso, tpr_post_lasso)
roc_auc_post_enet = auc(fpr_post_enet, tpr_post_enet)
roc_auc_post_hybsel = auc(fpr_post_hybsel, tpr_post_hybsel)
roc_auc_post_hybsel_lasso = auc(fpr_post_hybsel_lasso, tpr_post_hybsel_lasso)
roc_auc_post_hybsel_n_lasso = auc(fpr_post_hybsel_n_lasso, tpr_post_hybsel_n_lasso)


roc_pre = {'fpr': fpr_pre, 'tpr': tpr_pre, 'roc_auc': roc_auc_pre}
roc_lasso = {'fpr': fpr_post_lasso, 'tpr': tpr_post_lasso, 'roc_auc': roc_auc_post_lasso}
roc_enet = {'fpr': fpr_post_enet, 'tpr': tpr_post_enet, 'roc_auc': roc_auc_post_enet}
roc_hybsel = {'fpr': fpr_post_hybsel, 'tpr': tpr_post_hybsel, 'roc_auc': roc_auc_post_hybsel}
roc_hybsel_lasso = {'fpr': fpr_post_hybsel_lasso, 'tpr': tpr_post_hybsel_lasso, 'roc_auc': roc_auc_post_hybsel_lasso}
roc_hybsel_n_lasso = {'fpr': fpr_post_hybsel_n_lasso, 'tpr': tpr_post_hybsel_n_lasso, 'roc_auc': roc_auc_post_hybsel_n_lasso}


roc_results={'pre': roc_pre, 'lasso': roc_lasso, 'ent': roc_enet,'HybSel': roc_hybsel,'Hybsel_Lasso': roc_hybsel_lasso,'Hybsel_n_Lasso': roc_hybsel_n_lasso}

with open('roc_results_v5.pkl', 'wb') as f:
    pickle.dump(roc_results, f)

now = datetime.now()
timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
roc_image_name = f"roc_curve_{timestamp}.png"

# Plot ROC curves
plt.figure(figsize=(12, 8))
plt.plot(fpr_pre, tpr_pre, color='blue', lw=2, label=f'Pre-Feature Selection (AUC = {roc_auc_pre:.2f})')
plt.plot(fpr_post_lasso, tpr_post_lasso, color='red', lw=2, label=f'Post-Lasso  (AUC = {roc_auc_post_lasso:.2f})')
plt.plot(fpr_post_enet, tpr_post_enet, color='green', lw=2, label=f'Post-ElasticNet  (AUC = {roc_auc_post_enet:.2f})')
plt.plot(fpr_post_hybsel, tpr_post_hybsel, color='brown', lw=2, label=f'Post-HybSel  (AUC = {roc_auc_post_hybsel:.2f})')
plt.plot(fpr_post_hybsel, tpr_post_hybsel, color='orange', lw=2, label=f'Post-HybSel-Lasso  (AUC = {roc_auc_post_hybsel_lasso:.2f})')
plt.plot(fpr_post_hybsel, tpr_post_hybsel, color='purple', lw=2, label=f'Post-HybSel Intersect Lasso  (AUC = {roc_auc_post_hybsel_n_lasso:.2f})')

plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate',fontsize=18)
plt.ylabel('True Positive Rate',fontsize=18)
plt.tick_params(axis='x', labelsize=17)  
plt.tick_params(axis='y', labelsize=17) 
plt.legend(loc="lower right",fontsize=17)
plt.savefig(roc_image_name, dpi=600)
#plt.show()

# Print selected features and other metrics

selected_features_lasso = X.columns[feature_selector_lasso.get_support()]
selected_features_enet = X.columns[feature_selector_enet.get_support()]
selected_features_hybsel = pd.Index(selected_columns)
percentage_selected_lasso = (X_train_selected_lasso.shape[1] / X_train_scaled.shape[1]) * 100
percentage_selected_enet = (X_train_selected_enet.shape[1] / X_train_scaled.shape[1]) * 100
percentage_selected_hybsel = (X_train_scaled_hybsel.shape[1] / X_train_scaled.shape[1]) * 100
percentage_selected_hybsel_lasso = (X_train_scaled_Hybsel_Lasso.shape[1] / X_train_scaled.shape[1]) * 100
percentage_selected_hybsel_n_lasso = (X_train_scaled_Hybsel_n_lasso.shape[1] / X_train_scaled.shape[1]) * 100


print("Print Train Size")
print(X_train_scaled.shape)
print(X_train_selected_lasso.shape)
print(X_train_selected_enet.shape)
print(X_train_scaled_hybsel.shape)
print(X_train_scaled_Hybsel_Lasso.shape)
print(X_train_scaled_Hybsel_n_lasso.shape)


print("Print Test Size")
print(X_test_scaled.shape)
print(X_test_selected_lasso.shape)
print(X_test_selected_enet.shape)
print(X_test_scaled_hybsel.shape)
print(X_test_scaled_Hybsel_Lasso.shape)
print(X_test_scaled_Hybsel_n_lasso.shape)


print("Length of Selected Features (Lasso):", len(selected_features_lasso))
print("Length of Selected Features (ElasticNet):", len(selected_features_enet))
print("Length of Selected Features (HybSel):", len(selected_features_hybsel))
print("Length of Selected Features (HybSel(Sel_Col):", len(selected_columns))
print("Length of Selected Features (HybSel_lasso(Sle_Col):",len(Hybsel_Lasso_features))
print("Length of Selected Features (HybSel_lasso(Sle_Col):",len(Hybsel_n_lasso_features))

print(f"Percentage of Selected Features (Lasso): {percentage_selected_lasso:.2f}%")
print(f"Percentage of Selected Features (ElasticNet): {percentage_selected_enet:.2f}%")
print(f"Percentage of Selected Features (HybSel): {percentage_selected_hybsel:.2f}%")
print(f"Percentage of Selected Features (HybSel_Lasso): {percentage_selected_hybsel_lasso:.2f}%")
print(f"Percentage of Selected Features (HybSel_Lasso): {percentage_selected_hybsel_n_lasso:.2f}%")


print(f"Accuracy Improvement (Lasso): {accuracy_score(y_test, boost_post_lasso.predict(X_test_selected_lasso)) - accuracy_score(y_test, boost_pre.predict(X_test_scaled)):.4f}")
print(f"Accuracy Improvement (ElasticNet): {accuracy_score(y_test, boost_post_enet.predict(X_test_selected_enet)) - accuracy_score(y_test, boost_pre.predict(X_test_scaled)):.4f}")
print(f"Accuracy Improvement (Hybsel): {accuracy_score(y_test, boost_post_hybsel.predict(X_test_scaled_hybsel)) - accuracy_score(y_test, boost_pre.predict(X_test_scaled)):.4f}")
print(f"Accuracy Improvement (Hybsel_Lasso): {accuracy_score(y_test, boost_post_hybsel_lasso.predict(X_test_scaled_Hybsel_Lasso)) - accuracy_score(y_test, boost_pre.predict(X_test_scaled)):.4f}")
print(f"Accuracy Improvement (Hybsel_Lasso): {accuracy_score(y_test, boost_post_hybsel_n_lasso.predict(X_test_scaled_Hybsel_n_lasso)) - accuracy_score(y_test, boost_pre.predict(X_test_scaled)):.4f}")


