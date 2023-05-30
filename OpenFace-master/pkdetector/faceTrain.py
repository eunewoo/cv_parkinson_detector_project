# train_model.py
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from scipy.stats import mannwhitneyu
from sklearn.preprocessing import scale
from sklearn import svm
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score
import joblib

print('Now training Facial parkinson detection model...')

df = pd.read_csv("face_mimic_df.csv")

feats = ['AU_01_t12','AU_06_t12','AU_12_t12','AU_04_t13','AU_07_t13','AU_09_t13','AU_01_t14','AU_02_t14','AU_04_t14']
pca = PCA(n_components=3)
x_new = pca.fit_transform(df[feats])

X = df[feats].dropna()
y = df['diagnosed'].dropna()

X_resampled, y_resampled = SMOTE().fit_resample(X, y)

X = scale(X)
y = np.array(y)

best_auc = 0
best_f1 = 0
best_acc = 0
best_pre = 0
best_re = 0

for c in np.arange(0.1,10,2):
    for g in np.arange(0.1,2,0.05):
        clf = svm.SVC(kernel = 'rbf', C = c, gamma = g).fit(X_resampled,y_resampled)
        y_pred = cross_val_predict(clf, X_resampled, y_resampled, cv=10)
        acc = accuracy_score(y_resampled,y_pred)
        auc = roc_auc_score(y_resampled,y_pred)
        f1 = f1_score(y_resampled,y_pred)
        pre = precision_score(y_resampled,y_pred)
        re = recall_score(y_resampled,y_pred)
        if (auc>best_auc):
            best_auc = auc
            best_f1 = f1
            best_acc = acc
            best_pre = pre
            best_re = re

joblib.dump(clf, 'model.pkl')
print("Model is trained and saved.")
print('Accuracy of model is : ', best_auc)
