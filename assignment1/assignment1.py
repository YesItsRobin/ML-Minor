import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing

#1. Load the data set.
df = pd.read_csv('card_transdata.csv', header = 0)

#2. Thoroughly explore the data set.
print("DATA EXPLORATION")
print("Data sample:")
print(df[0:5])

print("\nFeatures:")
for c in df.columns:
    print("\t"+c)

print("\nNumber of samples: "+str(df.shape[0]))

plt.hist(df['distance_from_home'],8)
plt.xlabel('distance_from_home')
plt.ylabel('frequency')
#plt.show()

plt.hist(df['distance_from_last_transaction'],8)
plt.xlabel('distance_from_last_transaction')
plt.ylabel('frequency')
#plt.show()

plt.hist(df['ratio_to_median_purchase_price'],8)
plt.xlabel('ratio_to_median_purchase_price')
plt.ylabel('frequency')
#plt.show()

#draw boxplots of all features
#df.boxplot()
#plt.show()



#3. Based on the data exploration results, determine which preprocessing steps are needed.
'''
Distance from home is skewed to the right. So preprocessing used will be z-score normalization.
Distance from last transaction is skewed to the right. So preprocessing used will be z-score normalization.
Ratio to median purchase price is skewed to the right. So preprocessing used will be z-score normalization.
The other features are binary so no preprocessing is needed.
'''

#4. Preprocess the data as needed.
scaler = preprocessing.StandardScaler()
df_scaled = scaler.fit_transform(df[['distance_from_home', 'distance_from_last_transaction', 'ratio_to_median_purchase_price']])
df_scaled = pd.DataFrame(df_scaled, columns=df.columns[0:3])

print("\n transformed data:")
print(df_scaled)

print("\nMeans of original data:")
print(df.mean())
print("\nStandard deviations of original data:")
print(df.std())

print("\nMeans of transformed data:")
print(df_scaled.mean())
print("\nStandard deviations of transformed data:")
print(df_scaled.std())

#5. Build a k-nearest neighbor model for the given classification task.
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


X = df_scaled[['distance_from_home', 'distance_from_last_transaction', 'ratio_to_median_purchase_price']]
X2 =df[['repeat_retailer','used_chip','used_pin_number','online_order']]
X = pd.concat([X, X2], axis=1)
y = df['fraud']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)
print("Accuracy: "+str(accuracy_score(y_test, y_pred)))

#6. Build a naïve bayes model for the given classification task.
from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()
gnb.fit(X_train, y_train)

y_pred = gnb.predict(X_test)
print("Accuracy: "+str(accuracy_score(y_test, y_pred)))

#7. Apply the k-nearest neighbor model and naïve bayes model to classify a new example.