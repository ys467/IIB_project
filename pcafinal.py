#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

Created on Wed Jan 29 13:44:36 2020

@author: yungyu

"""

#import all necessary libraries and modules

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from mlxtend.plotting import plot_decision_regions
import matplotlib.gridspec as gridspec
import itertools

#read csv file constructed from gas sensor data
df = pd.read_csv(
        filepath_or_buffer = 'humidity.csv',
        names = ['Max S', 'Response Time',
                 'Area Ratio', 'Target'],
        sep = ',')
print(df)

#characterisation features selected from gas sensor data
features = ['Max S', 'Response Time', 'Area Ratio',]

#matrix truncation to define X and y
Xoriginal = df.iloc[:, 0:3].values
y = df.iloc[:, 3].values

#preprocessing - normalization to scale the data to a normal distribution
#mean = 0 and std. deviation = 1
X = StandardScaler().fit_transform(Xoriginal)

#set the number of PCA components to be equal to 2
pca = PCA(n_components = 2)

#PCA applied to the datasets
X = pca.fit_transform(X)

#prints the amount of information/variation held for each principal components
#after dimension reduction
print('Explained variation per principal component: {}'
      .format(pca.explained_variance_ratio_))

#store the values to be printed on PCA graph later
ratio = pca.explained_variance_ratio_
pca1 = round(ratio[0]*100,2)
pca2 = round(ratio[1]*100,2)

#initialization of different classifiers
logistic = LogisticRegression(random_state = 1,
                              solver = 'newton-cg',
                              multi_class = 'multinomial')
randomfor = RandomForestClassifier(random_state = 1,
                                   n_estimators = 100)
gnb = GaussianNB()
svm = SVC(gamma='auto')

#Figure settings
gs = gridspec.GridSpec(2,2)
fig = plt.figure(figsize = (12,12))
labels = ['(a) Logistic Regression', '(b) Random Forest',
          '(c) Gaussian Naive Bayes', '(d) SVM']

#loop to create subfigures with different classifiers
for classifier, label, grid in zip([logistic, randomfor, gnb, svm],
                                   labels,
                                   itertools.product([0,1], repeat = 2)):
    #data trained on classifier
    classifier.fit(X, y)
    ax = plt.subplot(gs[grid[0], grid[1]])
    fig = plot_decision_regions(X, y, clf = classifier, legend = 0)
    #customised legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles,
              ['Mixed gas dry', '10% humidity',
               '20% humidity', '30% humidity',],
              framealpha = 0.3,
              scatterpoints =1)
    plt.title(label)
    plt.xlabel('PC1: {}%'.format(pca1))
    plt.ylabel('PC2: {}%'.format(pca2))

plt.show()

#user input for novel data classification
usr1 = input("Enter your Max S value: ")
usr2 = input("Enter your Response Time value: ")
usr3 = input("Enter your Area Ratio value: ")

#PCA applied to the novel data based on previous PCA
usr_val = np.asarray([usr1, usr2, usr3])
modX = np.vstack([Xoriginal,usr_val])
modX = StandardScaler().fit_transform(modX)
modX = pca.fit_transform(modX)
usr_val_pca = modX[-1,:]

#Figure settings
gs = gridspec.GridSpec(2,2)
fig = plt.figure(figsize = (12,12))
#new set of labels for user figures with user input
labels_usr = ['(a) Logistic Regression', '(b) Random Forest',
              '(c) Gaussian Naive Bayes', '(d) SVM']

#loop to create subfigures with different classifiers
for classifier, label, grid in zip([logistic, randomfor, gnb, svm],
                                   labels_usr,
                                   itertools.product([0,1], repeat = 2)):
    #data trained on classifier
    classifier.fit(X, y)
    ax = plt.subplot(gs[grid[0], grid[1]])
    fig = plot_decision_regions(X, y, clf = classifier, legend = 0)
    #customised legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles,
              ['Mixed gas dry', '10% humidity',
               '20% humidity', '30% humidity'],
              framealpha = 0.3,
              scatterpoints =1)
    plt.title(label)
    plt.xlabel('PC1: {}%'.format(pca1))
    plt.ylabel('PC2: {}%'.format(pca2))
    #user input included asa black dot
    plt.scatter(usr_val_pca[0],usr_val_pca[1], c='black')

plt.show()
