import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB
np.random.seed(1)

df = pd.read_csv('card_transdata.csv', header = 0)

# Preprocessing scaled data
scaler = preprocessing.StandardScaler()
df_scaled = scaler.fit_transform(df[['distance_from_home', 'distance_from_last_transaction', 'ratio_to_median_purchase_price']])
df_scaled = pd.DataFrame(df_scaled, columns=df.columns[0:3])

# Preprocessing log transformed data
df_log_transformed = df[['distance_from_home', 'distance_from_last_transaction', 'ratio_to_median_purchase_price']].apply(np.log)

######### Training and testing KNN #########
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix

X = df_scaled
X2 = df[['repeat_retailer', 'used_chip', 'used_pin_number', 'online_order']]
X = pd.concat([X, X2], axis=1)

X = X.sample(frac=0.05, random_state=1) #using fraction of the dataset due to processing time
y = df['fraud'].iloc[X.index]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Hyperparameter tuning
k_range = np.arange(1, 20)
param_grid = {'n_neighbors': k_range}

knn = KNeighborsClassifier()
grid_search = GridSearchCV(estimator=knn, param_grid=param_grid, cv=5, n_jobs=-1)

grid_search.fit(X_train, y_train)
best_k = grid_search.best_params_['n_neighbors']

# Create a new KNN model with the best k
best_knn = KNeighborsClassifier(n_neighbors=best_k)
best_knn.fit(X_train, y_train)

y_pred = best_knn.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred)

print("K-NN:")
print("Best k:", best_k)
print("Accuracy (scaled):", accuracy)
print("Confusion Matrix (scaled):")
print(confusion)

#Test with log transformed data

# X = df_log_transformed
# X2 =df[['repeat_retailer','used_chip','used_pin_number','online_order']]
# X = pd.concat([X, X2], axis=1)
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
#
# knn = KNeighborsClassifier(n_neighbors=5)
# knn.fit(X_train, y_train)
#
# y_pred = knn.predict(X_test)
#
# accuracy = accuracy_score(y_test, y_pred)
# confusion = confusion_matrix(y_test, y_pred)
#
# print("Accuracy (log data):", accuracy)
# print("Confusion Matrix (log data):")
# print(confusion)

######### Training and testing Naive Bayes #########

# X = df_scaled
# X2 = df[['repeat_retailer', 'used_chip', 'used_pin_number', 'online_order']]
# X = pd.concat([X, X2], axis=1)
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
#
# gnb = GaussianNB()
# gnb.fit(X_train, y_train)
#
# y_pred = gnb.predict(X_test)
#
# accuracy = accuracy_score(y_test, y_pred)
# confusion = confusion_matrix(y_test, y_pred)
# print("Naive Bayes:")
# print("Accuracy (scaled):", accuracy)
# print("Confusion Matrix (scaled):")
# print(confusion)

#Test with log transformed data

# X = df_log_transformed
# X2 = df[['repeat_retailer', 'used_chip', 'used_pin_number', 'online_order']]
# X = pd.concat([X, X2], axis=1)
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
#
# gnb = GaussianNB()
# gnb.fit(X_train, y_train)
#
# y_pred = gnb.predict(X_test)
#
# accuracy = accuracy_score(y_test, y_pred)
# confusion = confusion_matrix(y_test, y_pred)
# print("Naive Bayes:")
# print("Accuracy (log):", accuracy)
# print("Confusion Matrix (log):")
# print(confusion)

######### Training and testing SVC #########

X = df_scaled
X2 = df[['repeat_retailer', 'used_chip', 'used_pin_number', 'online_order']]
X = pd.concat([X, X2], axis=1)

X = X.sample(frac=0.05) #using fraction of the dataset due to processing time
y = df['fraud'].iloc[X.index]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# gamma_range = np.logspace(-3, 3, 7)
# gamma_range = np.arange(10)
gamma_range = np.arange(0.4, 1.7, 0.1)

param_grid = {'gamma': gamma_range}

svm = SVC(kernel='rbf')
grid_search = GridSearchCV(estimator=svm, param_grid=param_grid, cv=5, n_jobs=-1)

grid_search.fit(X_train, y_train)
best_gamma = grid_search.best_params_['gamma']

# Create a new SVM model with the best gamma
best_svm = SVC(kernel='rbf', gamma=best_gamma)
best_svm.fit(X_train, y_train)

y_pred = best_svm.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred)

print ("SVC")
print("Best gamma:", best_gamma)
print("Accuracy (scaled):", accuracy)
print("Confusion Matrix (scaled):")
print(confusion)
#
# #Test with log transformed data

# X = df_scaled
# X2 = df[['repeat_retailer', 'used_chip', 'used_pin_number', 'online_order']]
# X = pd.concat([X, X2], axis=1)
#
# X = X.sample(frac=0.05) #using half of the dataset due to processing time
# y = df['fraud'].iloc[X.index]
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
# svm = SVC(kernel='rbf')
# svm.fit(X_train, y_train)
#
# y_pred = svm.predict(X_test)
#
# accuracy = accuracy_score(y_test, y_pred)
# confusion = confusion_matrix(y_test, y_pred)
#
# print ("SVC")
# print("Accuracy (log):", accuracy)
# print("Confusion Matrix (log):")
# print(confusion)
#
# ######### Training and testing Decision Tree #########
#
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, confusion_matrix
#
# X = df_scaled
# X2 = df[['repeat_retailer', 'used_chip', 'used_pin_number', 'online_order']]
# X = pd.concat([X, X2], axis=1)
# y = df['fraud']
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
#
# dt = DecisionTreeClassifier()
# dt.fit(X_train, y_train)
#
# y_pred = dt.predict(X_test)
#
# accuracy = accuracy_score(y_test, y_pred)
# confusion = confusion_matrix(y_test, y_pred)
#
# print ("Decision Tree")
# print("Accuracy (scaled):", accuracy)
# print("Confusion Matrix (scaled):")
# print(confusion)
#
# #Test with log transformed data
#
# X = df_log_transformed
# X2 = df[['repeat_retailer', 'used_chip', 'used_pin_number', 'online_order']]
# X = pd.concat([X, X2], axis=1)
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
#
# dt = DecisionTreeClassifier()
# dt.fit(X_train, y_train)
#
# y_pred = dt.predict(X_test)
#
# accuracy = accuracy_score(y_test, y_pred)
# confusion = confusion_matrix(y_test, y_pred)
#
# print ("Decision Tree")
# print("Accuracy (log):", accuracy)
# print("Confusion Matrix (log):")
# print(confusion)
#
# ######### Training and testing Random Forest #########
#
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, confusion_matrix
#
# X = df_scaled
# X2 = df[['repeat_retailer', 'used_chip', 'used_pin_number', 'online_order']]
# X = pd.concat([X, X2], axis=1)
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
#
# rf = RandomForestClassifier()
# rf.fit(X_train, y_train)
#
# y_pred = rf.predict(X_test)
#
# accuracy = accuracy_score(y_test, y_pred)
# confusion = confusion_matrix(y_test, y_pred)
#
# print ("Random Forest")
# print("Accuracy (scaled):", accuracy)
# print("Confusion Matrix (scaled):")
# print(confusion)
#
# #Test with log transformed data
#
# X = df_log_transformed
# X2 = df[['repeat_retailer', 'used_chip', 'used_pin_number', 'online_order']]
# X = pd.concat([X, X2], axis=1)
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
#
# rf = RandomForestClassifier()
# rf.fit(X_train, y_train)
#
# y_pred = rf.predict(X_test)
#
# accuracy = accuracy_score(y_test, y_pred)
# confusion = confusion_matrix(y_test, y_pred)
#
# print ("Random Forest")
# print("Accuracy (log):", accuracy)
# print("Confusion Matrix (log):")
# print(confusion)