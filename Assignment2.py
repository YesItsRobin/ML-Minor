import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB
from sklearn.experimental import enable_halving_search_cv as HalvingGridSearchCV
df = pd.read_csv('card_transdata.csv', header = 0)

# Preprocessing scaled data
scaler = preprocessing.StandardScaler()
df_scaled = scaler.fit_transform(df[['distance_from_home', 'distance_from_last_transaction', 'ratio_to_median_purchase_price']])
df_scaled = pd.DataFrame(df_scaled, columns=df.columns[0:3])

# Preprocessing log transformed data
df_log_transformed = df[['distance_from_home', 'distance_from_last_transaction', 'ratio_to_median_purchase_price']].apply(np.log)

######### Training and testing KNN #########
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

X = df_scaled
X2 =df[['repeat_retailer','used_chip','used_pin_number','online_order']]
X = pd.concat([X, X2], axis=1)
y = df['fraud']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred)

print("K-NN:")
print("Accuracy (scaled):", accuracy)
print("Confusion Matrix (scaled):")
print(confusion)

#Test with log transformed data

X = df_log_transformed
X2 =df[['repeat_retailer','used_chip','used_pin_number','online_order']]
X = pd.concat([X, X2], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred)

print("Accuracy (log data):", accuracy)
print("Confusion Matrix (log data):")
print(confusion)

######### Training and testing Naive Bayes #########

X = df_scaled
X2 = df[['repeat_retailer', 'used_chip', 'used_pin_number', 'online_order']]
X = pd.concat([X, X2], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

gnb = GaussianNB()
gnb.fit(X_train, y_train)

y_pred = gnb.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred)
print("Naive Bayes:")
print("Accuracy (scaled):", accuracy)
print("Confusion Matrix (scaled):")
print(confusion)

#Test with log transformed data

X = df_log_transformed
X2 = df[['repeat_retailer', 'used_chip', 'used_pin_number', 'online_order']]
X = pd.concat([X, X2], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

gnb = GaussianNB()
gnb.fit(X_train, y_train)

y_pred = gnb.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred)
print("Naive Bayes:")
print("Accuracy (log):", accuracy)
print("Confusion Matrix (log):")
print(confusion)

######### Training and testing SVC #########

X = df_scaled
X2 = df[['repeat_retailer', 'used_chip', 'used_pin_number', 'online_order']]
X = pd.concat([X, X2], axis=1)

X = X.sample(frac=0.5) #using half of the dataset due to processing time
y = df['fraud'].iloc[X.index]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Hyperparameter tuning gamma: float | Literal['scale', 'auto'] = "scale",
from sklearn.model_selection import HalvingGridSearchCV 
svm = SVC()
param_grid = {'gamma': np.arange(1, 10, 2)}

svm_gscv = HalvingGridSearchCV(svm, param_grid, cv=5)
svm_gscv.fit(X_train, y_train)
print("Best parameters:", svm_gscv.best_params_)
print("Best score:", svm_gscv.best_score_)
print("Best estimator:", svm_gscv.best_estimator_)
print("Best index:", svm_gscv.best_index_)
print("Scorer function:", svm_gscv.scorer_)
print("CV results:", svm_gscv.cv_results_)
print("Refit time:", svm_gscv.refit_time_)
print("Predictions:", svm_gscv.predict(X_test))
print("Score:", svm_gscv.score(X_test, y_test))

#Using best parameters from GridSearchCV
svm = SVC(kernel=svm_gscv.best_params_['kernel'])
svm.fit(X_train, y_train)

y_pred = svm.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred)

print ("SVC")
print("Accuracy (scaled):", accuracy)
print("Confusion Matrix (scaled):")
print(confusion)
#
# #Test with log transformed data

X = df_log_transformed
X2 = df[['repeat_retailer', 'used_chip', 'used_pin_number', 'online_order']]
X = pd.concat([X, X2], axis=1)

X = X.sample(frac=0.5) #using half of the dataset due to processing time
y = df['fraud'].iloc[X.index]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

svm = SVC(kernel='rbf')
svm.fit(X_train, y_train)

y_pred = svm.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred)

print ("SVC")
print("Accuracy (log):", accuracy)
print("Confusion Matrix (log):")
print(confusion)

######### Training and testing Decision Tree #########

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

X = df_scaled
X2 = df[['repeat_retailer', 'used_chip', 'used_pin_number', 'online_order']]
X = pd.concat([X, X2], axis=1)
y = df['fraud']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)

y_pred = dt.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred)

print ("Decision Tree")
print("Accuracy (scaled):", accuracy)
print("Confusion Matrix (scaled):")
print(confusion)

#Test with log transformed data

X = df_log_transformed
X2 = df[['repeat_retailer', 'used_chip', 'used_pin_number', 'online_order']]
X = pd.concat([X, X2], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)

y_pred = dt.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred)

print ("Decision Tree")
print("Accuracy (log):", accuracy)
print("Confusion Matrix (log):")
print(confusion)

######### Training and testing Random Forest #########

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

X = df_scaled
X2 = df[['repeat_retailer', 'used_chip', 'used_pin_number', 'online_order']]
X = pd.concat([X, X2], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

rf = RandomForestClassifier()
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred)

print ("Random Forest")
print("Accuracy (scaled):", accuracy)
print("Confusion Matrix (scaled):")
print(confusion)

#Test with log transformed data

X = df_log_transformed
X2 = df[['repeat_retailer', 'used_chip', 'used_pin_number', 'online_order']]
X = pd.concat([X, X2], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

rf = RandomForestClassifier()
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred)

print ("Random Forest")
print("Accuracy (log):", accuracy)
print("Confusion Matrix (log):")
print(confusion)