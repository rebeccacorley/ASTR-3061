#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 15:03:51 2020
@author: rebeccacorley
"""

# This is my first attempt in plotting redshift vs. log distance whoo.  

import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt('/Users/rebeccacorley/Documents/Observational Astro/DES-Data1.txt')
print(data)

redshift = data[:,4] #columns sart at 0!!
log_dist = data[:,2] #however many columns I want, the colon indicates all 
 

#The following lines play around with different graph properties
plt.plot(redshift, log_dist, '.', ms = 5, color = 'k', label = 'redshift data')
plt.xlabel('Z')
plt.ylabel('log(d)') 
plt.legend()
plt.show()

# Practice change
