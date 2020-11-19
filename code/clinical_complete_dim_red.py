# -*- coding: utf-8 -*-
"""
Created on Sun Nov  8 16:30:13 2020

@author: Chase
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler #used for 'Feature Scaling'

data_raw_clinical = pd.read_csv('C:\\Users\\Chase\\Google Drive\\College\\Graduate\\C3.ai Grand Challenge\\Dimensionality Reduction\\complete_timeseries_clinical.csv')
demo_data = np.genfromtxt('demographic_timeseries.csv', delimiter=',')
demo_data_ga = demo_data[1530:1682,1:]
data_clinical = data_raw_clinical.filter(regex='Georgia|California|NewYork|Florida')
data_np_clinical = data_clinical.to_numpy()

def gmm(data, components):
        # Fit a Gaussian mixture with EM using five components
        gmm = GaussianMixture(components, covariance_type='full').fit(data)
        labels = gmm.predict(data)
        
        pca = PCA(n_components=2)
        pca_data = pca.fit_transform(data)
        plt.scatter(pca_data[:,0], pca_data[:, 1], c=labels, s=40, cmap='viridis')
        plt.title('PCA #1 & #2 Scatter Plot - GMM - Georgia')
        plt.show()
        
        probs = gmm.predict_proba(data)
        #probs_binary = np.round(gmm.predict_proba(data),0)
        aic = gmm.aic(data)
        bic = gmm.bic(data)
        return probs, aic, bic

# clinical data GMM clustering, 1-10 clusters
components_vec = [2,3,4,5,6,7,8,9,10]
for i in range(len(components_vec)):
    probs, aic, bic = gmm(demo_data, components_vec[i])

# demographic data GMM clustering
data_raw_demo = pd.read_csv('C:\\Users\\Chase\\Google Drive\\College\\Graduate\\C3.ai Grand Challenge\\Dimensionality Reduction\\demographic_timeseries.csv')
data_np_demo = data_raw_demo.to_numpy()
