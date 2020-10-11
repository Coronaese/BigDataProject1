#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import time

# import data set
col_names = ["label", "firstBlood", "firstTower", "firstInhibitor", "firstBaron", "firstDragon", "firstRiftHerald", "t1_towerKills", "t1_inhibitorKills", "t1_baronKills", "t1_dragonKills", "t1_riftHeraldKills", "t2_towerKills", "t2_inhibitorKills", "t2_baronKills", "t2_dragonKills", "t2_riftHeraldKills"]
tar = pd.read_csv("F:\\Resource\\华工\\暂存\\大二\\工业大数据分析及应用\\Lab\\Project1\\new_data.csv", header = None, names = col_names)
tes = pd.read_csv("F:\\Resource\\华工\\暂存\\大二\\工业大数据分析及应用\\Lab\\Project1\\test_set.csv", header = None, names = col_names)
tar = tar.iloc[1:]
tar.head()

# construct variables
feature_cols = ["firstBlood", "firstTower", "firstInhibitor", "firstBaron", "firstDragon", "firstRiftHerald", "t1_towerKills", "t1_inhibitorKills", "t1_baronKills", "t1_dragonKills", "t1_riftHeraldKills", "t2_towerKills", "t2_inhibitorKills", "t2_baronKills", "t2_dragonKills", "t2_riftHeraldKills"]

tes[feature_cols] = tes[feature_cols].apply(pd.to_numeric, errors='coerce').fillna(0.0)
X_train = tar[feature_cols]
y_train = tar.label
X_test = tes[feature_cols]
y_test = tes.label

# Max_depth for DT
# depth = range(1, 20)
# d_scores = []
# for m in depth:
#     clf = DecisionTreeClassifier(max_depth=m)
#     clf = clf.fit(X_train, y_train)
#     y_pred = clf.predict(X_test)
#     d_scores.append(accuracy_score(y_test, y_pred))
#
# plt.plot(depth, d_scores)
# plt.xlabel('depth for Decision Tree')
# plt.ylabel('Accuracy on test set')
# plt.show()

# DT model
clf = DecisionTreeClassifier(max_depth=5)
time_start = time.time()
clf = clf.fit(X_train, y_train)
time_end = time.time()
y_pred = clf.predict(X_test)
# calculate the accuracy of DT model on test set
print("Decision Tree Accuracy:", accuracy_score(y_test, y_pred))
print("Decision Tree Training Time:", time_end - time_start)

# svm 's choosing kernel
# kernel = ['linear' ,'rbf', 'sigmoid']
# kernel_scores = []
# for m in kernel:
#     clf1 = SVC(kernel=m, gamma='auto')
#     clf1.fit(X_train, y_train)
#     y_pred = clf1.predict(X_test)
#     kernel_scores.append(accuracy_score(y_test, y_pred))
#
# plt.plot(kernel, kernel_scores)
# plt.xlabel('kernel for svm')
# plt.ylabel('Accuracy on test set')
# plt.show()

# svm model
clf1 = SVC(kernel='rbf', gamma='auto')
time_start1 = time.time()
clf1.fit(X_train, y_train)
time_end1 = time.time()
y_pred = clf1.predict(X_test)
# calculate the accuracy of SVM model on test set
print("SVM Accuracy:", accuracy_score(y_test, y_pred))
print("SVM Training Time:", time_end1 - time_start1)

# KNN 's choosing K value
# k_range = range(1, 31)
# k_scores = []
# for k in k_range:
#     clf2 = KNeighborsClassifier(n_neighbors=k)
#     clf2.fit(X_train, y_train)
#     y_pred = clf2.predict(X_test)
#     k_scores.append(accuracy_score(y_test, y_pred))
#
# plt.plot(k_range, k_scores)
# plt.xlabel('Value of K for KNN')
# plt.ylabel('Accuracy on test set')
# plt.show()

# KNN model
clf2 = KNeighborsClassifier(n_neighbors=11)
time_start2 = time.time()
clf2.fit(X_train, y_train)
time_end2 = time.time()
y_pred = clf2.predict(X_test)
# calculate the accuracy of KNN model on test set
print("KNN Accuracy:", accuracy_score(y_test, y_pred))
print("KNN Training Time:", time_end2 - time_start2)
