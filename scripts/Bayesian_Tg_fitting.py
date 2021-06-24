# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 15:32:08 2021

@author: jmerri5
"""
from IPython.core.pylabtools import figsize
import numpy as np
from matplotlib import pyplot as plt
from tkinter import *
from tkinter import filedialog
# Create Tk root

import scipy.stats as stats
#Outputs a 2xNxM matrix where N is the number of files selected and M is the length of the data columns.
def import_data(isMultiple, T_max = 120):
    root = Tk()
    # Hide the main window
    root.withdraw()
    root.call('wm', 'attributes', '.', '-topmost', True)
    if isMultiple is False:
        infile = filedialog.askopenfilename(multiple=False)
        t, intensity = np.loadtxt(infile, skiprows=1, unpack=True) #load a single file and read all but the first row into variables
        T = T_max - (t/60.0)
        return T, intensity
    else:
        infile = filedialog.askopenfilename(multiple = True)
        T, intensity = np.empty((2, len(infile), len(infile[0])))
        for i in range(len(infile)):
            temp = np.loadtxt(infile[i], skiprows=1, unpack=True)
            T[i] = np.subtract(T_max, np.divide(temp[0], 60))
            intensity[i] = temp[1]
        return T, intensity
    


# plt.tight_layout()
is_fluorescence = True


if is_fluorescence is False:
    dat = np.loadtxt(r"C:\Users\jmerri5\OneDrive - Emory University\Polymer Drive\Jamie Merrill\Bayesian Analysis\example ellipsometry tg.txt", skiprows=1, unpack=True)
    count_data = dat[0]
    T = dat[1]
    target_accept = .93
else:
    dat = import_data(False, 120)
    count_data = dat[1]/min(dat[1])
    T = dat[0]
    target_accept = 0.95
    
figsize(12.5, 10)
#count_data = np.loadtxt(r"C:\Users\jmerri5\Desktop\txtdata.csv")
n_count_data = len(count_data)
plt.scatter(T, count_data, color="#348ABD", marker = "o")
plt.xlabel("Temperature (C)")
plt.ylabel("Counts")
# plt.title("Is there a transition?")
plt.xlim(int(min(T)-1), int(max(T)+1));
plt.figure()
x = T
y = count_data   
data = dict(x=x, y=y)


import pymc3 as pm
import theano.tensor as tt
import arviz as az

with pm.Model() as model:
    sigma = pm.HalfCauchy("Sigma", beta=10, testval=1.0)
    intercept_1 = pm.Normal("Intercept_1", 0, sigma=20)
    intercept_2 = pm.Normal("Intercept_2", 0, sigma=20)
    
    slope_1 = pm.Normal("Slope_1", 0, sigma=20)
    slope_2 = pm.Normal("Slope_2", 0, sigma=20)
    
    idx = np.arange(n_count_data) # Index
    tg = pm.DiscreteUniform("Tg", lower = 1, upper = n_count_data) # uniform probability over length of temperature array

    slope = pm.math.switch(tg > idx, slope_1, slope_2)
    intercept = pm.math.switch(tg > idx, intercept_1, intercept_2)
    
    regression = slope * idx + intercept
    y_obs = pm.Normal("y_obs", mu = regression,  sigma = sigma, observed=y)

    #MCMC to obtain posterior distribution
    # draw 10000 samples from final posterior distribution
    #
    # tune posterior 5000X (monte carlo on all variables, calculate loss, iterate and throw away those samples)
    # use no U-turn sampler (fancy hamiltonian markov chain monte carlo algo) to sample posterior distribution
    # 
    #
    trace = pm.sample(10000, tune=10000, init='jitter+adapt_diag', step=pm.NUTS(target_accept = target_accept), cores=1)
    az.plot_trace(trace)
    display(az.summary(trace, round_to = 4))
    pp = pm.sampling.fast_sample_posterior_predictive(trace)
    data_p = az.from_pymc3(trace, posterior_predictive=pp)
    az.plot_ppc(data_p, 'cumulative')
    
