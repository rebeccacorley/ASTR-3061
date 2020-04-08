# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 12:56:55 2020

Sample code to read in SNe data download from the net,
plot it, and compare with predictions from BB cosmology

@author: Lani Chastain
"""

import numpy as np
import matplotlib.pyplot as plt

# read in the table
data = np.genfromtxt('SCPUnion2.1_mu_vs_z.txt')
# pull out the redshift, magnitudes (mm) and magnitude estimated errors (dm)
redshift = data.T[1]
mm = data.T[2]
dm = data.T[3]
dlabel = 'SCP_2.1'

# mm is the "distance modulus" 
# from which we can compute the distance in parsecs (pc)
dpc = 10.**(mm/5.+1.)
# and in megaparsecs (Mpc)
dMpc = dpc / 10.**6
# and the error on that distance:
dMe = 10.**((mm+dm)/5.+1.-6.) - dMpc

# the Hubble relation :  d = c*z/H0 , with H0 in km/s/Mpc
c = 3.e5   # speed of light in km/s
# more accurately, the Hubble relation is d = c*z*sqrt(z)/H0
# so let's estimate H0 from the data; but only for small redshift in linear region:
indx = np.where(redshift < 0.05)
H0 = (c/dMpc[indx]*redshift[indx]).mean()
#H0 = (c/dMpc*redshift*np.sqrt(1+redshift)).mean()
# make a string to print on the plot:
sH0 = 'H0 = '+str(round(H0))+' km/s/Mpc'

# we will draw curves to "predict" where the data points should lie.
# first, make a vector of s with the numpy "arange" function.
dz = 0.001
zz = np.arange(dz,2.0,dz)
# the simplest (linear in z) Hubble relation curve:
ds1 = c/H0*zz
# a bit more accurate (and very similar to FRW with OmM = 0.43)
ds2 = c/H0*zz*np.sqrt(1.+zz)
# The general Hubble relation, Standard cosmological model, CDMLambda
OmM = 0.3  # fraction of matter, both ordinary matter (7%) and dark matter (23%)
OmL = 1. - OmM   # fraction of dark energy. They sum to 1 in our flat universe.
# This is the Friedmann-Robertson-Walker (FRW) integral for a flat universe:
ds3 = c/H0*np.cumsum(1./np.sqrt(OmM*(1+zz)**3+OmL))*dz*(1.+zz)
# in terms of magnitudes rather than distance:
m1 = (np.log10(ds1)+5.)*5. 
m2 = (np.log10(ds2)+5.)*5. 
m3 = (np.log10(ds3)+5.)*5. 

# compute the match between the data and the predictions,
#  using the chisq-per-degree-of-freedom:
# get the predictions at the same s as the data, by interpolating:
dp2 = np.interp(redshift,zz,ds2)
dp3 = np.interp(redshift,zz,ds3)
# compute the chisq per degree of freedom:
chisq2 = np.sum( ((dMpc-dp2)/dMe)**2 ) / dMpc.size
chisq3 = np.sum( ((dMpc-dp3)/dMe)**2 ) / dMpc.size
print( 'chisq2, chisq3 = ',chisq2,chisq3)

# draw a linear-linear plot with all the data and the three Hubble relation curves:
plt.figure()
plt.errorbar(redshift,dMpc,xerr=dz,yerr=dMe,fmt='+',label=dlabel)
plt.plot(zz,ds1,'m',label="d = c/H0*z, linear")
plt.plot(zz,ds2,'g',label="d = c/H0*z*sqrt(1+z)")
plt.plot(zz,ds3,'pink',label="d = c/H0*FRW(z)")
plt.xlim([0,1.5])
plt.ylim([0,14000])
plt.xlabel(' z')
plt.ylabel('Luminosity distance (Mpc)')
plt.text(0.1,9000,sH0,fontsize = 12)
plt.grid(b=True,which='both')
plt.legend(loc='upper left')
plt.show()

# zoom in on the lower left corner to see that all three curves
# agree with the data. The linear Hubble relation works fine for z < 0.1.

