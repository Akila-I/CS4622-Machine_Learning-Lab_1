# %% [markdown]
# ### Import libraries

# %% [markdown]
# # Feature Engineering for Label 3

# %%
import pandas as pd
import numpy as np

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

from matplotlib import pyplot as plt
import seaborn as sns

# %% [markdown]
# ### Define string variables

# %%
FEATURES = []
for i in range(1, 257):
  FEATURES.append('feature_' + str(i))
  
print(FEATURES)

# %% [markdown]
# ### Model to predict the label 3

# %%
# k-Nearest Neighbors Classifier
knn_model = KNeighborsClassifier(n_neighbors=3)

# %% [markdown]
# ### Import data

# %%
df_train = pd.read_csv('train.csv')
df_valid = pd.read_csv('valid.csv')

# %%
df_train.head()

# %%
df_valid.head()

# %%
df_train.shape

# %%
df_train.describe()

# %%
df_valid.shape

# %%
df_valid.describe()

# %%
missing_values_sum =  df_train.isnull().sum()
missing_values_sum[missing_values_sum > 0]

# %% [markdown]
# Only label 2 has missing values. Does not affect for feature engineering for label 3

# %% [markdown]
# ### Split train and validate data for features and labels

# %%
x_train = df_train.values[:, 0:256]
y_train = df_train.values[:, 258]

x_valid = df_valid.values[:, 0:256]
y_valid = df_valid.values[:, 258]

# %% [markdown]
# ### Scale the data

# %%
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train) 
x_valid = scaler.transform(x_valid)

# %%
pd.DataFrame(x_train, columns=FEATURES).describe()

# %% [markdown]
# After scaling, mean -> 0, std -> 1

# %% [markdown]
# ### Accuracy before feature engineering

# %%
knn_model.fit(x_train, y_train)
knn_predictions = knn_model.predict(x_valid)
knn_accuracy = accuracy_score(y_valid, knn_predictions)
knn_conf_matrix = confusion_matrix(y_valid, knn_predictions)
print("KNN Accuracy:", knn_accuracy)

plt.figure(figsize=(15, 10))
sns.heatmap(knn_conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix from KNN")
plt.show()

# %% [markdown]
# ### K-Best Features selection method

# %%
from sklearn.feature_selection import SelectKBest, f_classif

k_best_selector = SelectKBest(f_classif, k=100)
x_train_kbest = k_best_selector.fit_transform(x_train, y_train)
x_train_kbest.shape

# %%
x_valid_kbest = k_best_selector.transform(x_valid)

knn_model.fit(x_train_kbest, y_train)
knn_predictions = knn_model.predict(x_valid_kbest)
knn_accuracy = accuracy_score(y_valid, knn_predictions)
knn_conf_matrix = confusion_matrix(y_valid, knn_predictions)
print("KNN Accuracy:", knn_accuracy)

plt.figure(figsize=(15, 10))
sns.heatmap(knn_conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix from KNN")
plt.show()

# %% [markdown]
# ### Best percentile Features selecetion method

# %%
from sklearn.feature_selection import SelectPercentile, f_classif

percentile_selector = SelectPercentile(f_classif, percentile=40)
x_train_percentile = percentile_selector.fit_transform(x_train, y_train)
x_train_percentile.shape

# %%
x_valid_percentile = percentile_selector.transform(x_valid)

knn_model.fit(x_train_percentile, y_train)
knn_predictions = knn_model.predict(x_valid_percentile)
knn_accuracy = accuracy_score(y_valid, knn_predictions)
knn_conf_matrix = confusion_matrix(y_valid, knn_predictions)
print("KNN Accuracy:", knn_accuracy)

plt.figure(figsize=(15, 10))
sns.heatmap(knn_conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix from KNN")
plt.show()

# %% [markdown]
# ### PCA Decomposition

# %%
from sklearn.decomposition import PCA

pca = PCA(n_components=0.97,svd_solver='full')
x_train_pca = pca.fit_transform(x_train)
x_train_pca.shape

# %%
x_valid_pca = pca.transform(x_valid)

knn_model.fit(x_train_pca, y_train)
knn_predictions = knn_model.predict(x_valid_pca)
knn_accuracy = accuracy_score(y_valid, knn_predictions)
knn_conf_matrix = confusion_matrix(y_valid, knn_predictions)
print("KNN Accuracy:", knn_accuracy)

plt.figure(figsize=(15, 10))
sns.heatmap(knn_conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix from KNN")
plt.show()

# %% [markdown]
# ### Combine K-Best and PCA

# %%
pca_for_kbest = PCA(n_components=0.95)
x_train_pca_with_kbest = pca_for_kbest.fit_transform(x_train_kbest)
x_train_pca_with_kbest.shape

# %%
x_valid_pca_with_kbest = pca_for_kbest.transform(x_valid_kbest)

knn_model.fit(x_train_pca_with_kbest, y_train)
knn_predictions = knn_model.predict(x_valid_pca_with_kbest)
knn_accuracy = accuracy_score(y_valid, knn_predictions)
knn_conf_matrix = confusion_matrix(y_valid, knn_predictions)
print("KNN Accuracy:", knn_accuracy)

plt.figure(figsize=(15, 10))
sns.heatmap(knn_conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix from KNN")
plt.show()

# %% [markdown]
# ### Add Percentile selection for K-best and PCA

# %%
percentile_selector_for_kbest_and_pca = SelectPercentile(f_classif, percentile=50)
x_train_percentile_for_pca_with_kbest = percentile_selector_for_kbest_and_pca.fit_transform(x_train_pca_with_kbest, y_train)
x_train_percentile_for_pca_with_kbest.shape

# %%
x_valid_percentile_for_pca_with_kbest = percentile_selector_for_kbest_and_pca.transform(x_valid_pca_with_kbest)

knn_model.fit(x_train_percentile_for_pca_with_kbest, y_train)
knn_predictions = knn_model.predict(x_valid_percentile_for_pca_with_kbest)
knn_accuracy = accuracy_score(y_valid, knn_predictions)
knn_conf_matrix = confusion_matrix(y_valid, knn_predictions)
print("KNN Accuracy:", knn_accuracy)

plt.figure(figsize=(15, 10))
sns.heatmap(knn_conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix from KNN")
plt.show()

# %% [markdown]
# ### Read the test.csv file, transform its features and save the transformed features in a new file

# %%
df_test = pd.read_csv('test.csv')
df_test.head()

# %%
missing_values_sum =  df_test.isnull().sum()
missing_values_sum[missing_values_sum > 0]

# %%
x_test = df_test.values[:, 0:256]
x_test = scaler.transform(x_test)

x_test_kbest = k_best_selector.transform(x_test)
x_test_pca_with_kbest = pca_for_kbest.transform(x_test_kbest)
x_test_percentile_for_pca_with_kbest = percentile_selector_for_kbest_and_pca.transform(x_test_pca_with_kbest)

# %% [markdown]
# ### Prediction before feature engineering

# %%
knn_model.fit(x_train, y_train)
knn_predictions_before = knn_model.predict(x_test)

knn_predictions_before

# %% [markdown]
# ### Prediction after feature engineering

# %%
knn_model.fit(x_train_percentile_for_pca_with_kbest, y_train)
knn_predictions_after = knn_model.predict(x_test_percentile_for_pca_with_kbest)

knn_predictions_after

# %%
df_for_csv = pd.DataFrame()
df_for_csv['Predicted labels before feature engineering'] = knn_predictions_before
df_for_csv['Predicted labels after feature engineering'] = knn_predictions_after
df_for_csv['No of new features'] = x_test_percentile_for_pca_with_kbest.shape[1]

for i in range(x_test.shape[1]):
    df_for_csv[f'new_feature_{i+1}'] = x_test_percentile_for_pca_with_kbest[:, i] if i < x_test_percentile_for_pca_with_kbest.shape[1] else pd.NA

df_for_csv.head()

# %%
df_for_csv.to_csv('190239A_label3.csv', index=False)


