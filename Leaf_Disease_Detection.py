# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import os

path = os.listdir('Dataset/')
classes = {'Pepper_bell_Bacterial_spot':0, 'Pepper_bell_healthy':1,'Potato_Early_blight':2,'Potato_healthy':3,'Tomato_mosaic_virus':4,'Tomato_healthy':5}

import cv2
X = []
Y = []
for cls in classes:
    pth = 'Dataset/'+cls
    for j in os.listdir(pth):
        img = cv2.imread(pth+'/'+j, 0)
        img = cv2.resize(img, (200,200))
        X.append(img)
        Y.append(classes[cls])
        
X = np.array(X)
Y = np.array(Y)

X_updated = X.reshape(len(X), -1)

np.unique(Y)

pd.Series(Y).value_counts()

(X.shape, X_updated.shape)

plt.imshow(X[0], cmap='gray')

xtrain, xtest, ytrain, ytest = train_test_split(X_updated, Y, random_state=10,
                                               test_size=.20)

xtrain.shape, xtest.shape

print(xtrain.max(), xtrain.min())
print(xtest.max(), xtest.min())
xtrain = xtrain/255
xtest = xtest/255
print(xtrain.max(), xtrain.min())
print(xtest.max(), xtest.min())

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

import warnings
warnings.filterwarnings('ignore')

lg = LogisticRegression(C=0.1)
lg.fit(xtrain, ytrain)

sv = SVC()
sv.fit(xtrain, ytrain)

print("Training Score:", lg.score(xtrain, ytrain))
print("Testing Score:", lg.score(xtest, ytest))

print("Training Score:", sv.score(xtrain, ytrain))
print("Testing Score:", sv.score(xtest, ytest))

pred = sv.predict(xtest)

misclassified=np.where(ytest!=pred)
misclassified

print("Total Misclassified Samples: ",len(misclassified[0]))
print(pred[3],ytest[3])

import pickle
with open('lgmodel.pkl', 'wb') as file:
    pickle.dump(lg, file)
with open('svmodel.pkl', 'wb') as file:
    pickle.dump(sv, file)


    
