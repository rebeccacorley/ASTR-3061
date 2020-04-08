# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 20:03:39 2020

A simple MCMC code

@author: Lani Chastain
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st

''' Calculate the posterior '''

def target(lik, prior, param, theta):
    if theta < 0 or theta > 1: 
        return 0
    else: 
        return lik(param[0], theta).pmf(param[1])*prior.pdf(theta)
    

param = 14, 10 #our initial data; 14 trials and 10 heads
a = 1 #Params for the beta function
b = 1

lik = st.binom #our liklihood
prior = st.beta(a,b) #our prior
sigma = 0.1 #standard deviation for the gaussian proposal distribution

theta = 0.05 #our guess for p

niters = 5000 #how many iterations we want to do
samples = np.zeros(niters+1)
samples[0] = theta

for i in range(niters):
    theta_p = theta + st.norm(0, sigma).rvs() #new step
    rho = min(1, target(lik, prior, param, theta_p)/target(lik, prior, param, theta))
    u = np.random.uniform()
    ''' Metropolis-Hastings algorithm '''
    if u < rho: 
        theta = theta_p
    samples[i+1] = theta
nmcmc = len(samples)//2

'''chain genorated by the algorithm'''
plt.plot(samples, ':', label='Chain')
plt.xlabel("p")
plt.ylabel("P(p|D)")
plt.legend()
plt.show()