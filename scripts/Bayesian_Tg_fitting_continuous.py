# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 16:14:21 2021

@author: jmerri5
"""
from IPython.core.pylabtools import figsize
import numpy as np
from matplotlib import pyplot as plt
from tkinter import *
from tkinter import filedialog
import pymc3 as pm
import theano.tensor as tt
import arviz as az
import scipy.stats as stats


##Data handling
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
    
#global variables
is_fluorescence = True 
plot_ppc = True

if is_fluorescence is False:
    dat = np.loadtxt(r"C:\Users\jmerri5\OneDrive - Emory University\Polymer Drive\Jamie Merrill\Bayesian Analysis\example ellipsometry tg.txt", skiprows=1, unpack=True)
    count_data = dat[0]
    T = dat[1]
    target_accept = 0.8 #same as default value for NUTS
else:
    dat = import_data(False, 120)
    count_data = dat[1] / min(dat[1])
    T = dat[0]
    target_accept = 0.9 #fluorescence data is noisier --> use a slightly higher acceptance target than default 80% (soft tuning param)
    
figsize(12.5, 10)
#count_data = np.loadtxt(r"C:\Users\jmerri5\Desktop\txtdata.csv")
n_count_data = len(count_data)
plt.scatter(count_data,T, color="#348ABD", marker = "o")
plt.xlabel("Temperature (C)")
plt.ylabel("Counts")
plt.xlim(int(min(T)-1), int(max(T)+1));
plt.figure()
x = T
y = count_data   
data = dict(x=x, y=y)



with pm.Model() as model:
    
    #define priors: initial guess for probability distributions of each variable in model
    
    sigma = pm.HalfCauchy("Sigma", beta=10, testval=1.0)
    intercept_1 = pm.Normal("Intercept_1", 0, sigma=20, testval = 1.0)
    intercept_2 = pm.Normal("Intercept_2", 0, sigma=20)
    slope_1 = pm.Normal("Slope_1", 0, sigma=20)
    slope_2 = pm.Normal("Slope_2", 0, sigma=20)
    w = pm.HalfNormal("Transition Width", sigma = 2.5)
    tg = pm.Uniform('Tg', lower = 0, upper = n_count_data) # uniform probability over length of temperature array
    
    #Derived quantities
    idx = np.arange(n_count_data) # Index of temp array
    
    weight = tt.nnet.sigmoid(w * (idx-tg)) #sigmoid = 1/(1+e^-(2 (idx-tg))) to approximate switchpoint instead of step change 
    slope = weight*slope_1 + (1-weight) * slope_2 
    intercept = weight*intercept_1 + (1-weight) * intercept_2
    regression = slope * idx + intercept 

    # full model: f(sigma, i1, i2, s1, s2, Tg) into one gaussian random variable, whose mean value = slope * T + intercept, w/variance sigma
    # pdf of this new variable y_obs is something like the normalized expansivity, while cdf ~ the normalized temp-dependent signal;
    y_obs = pm.Normal("y_obs", mu = regression, sigma = sigma, observed=y, testval = 1.0)
    ### MCMC to obtain posterior distribution
    # draw 10000 samples from posterior distribution after using 10000 samples to tune the step lengths of each variable (NUTS)
    #
    trace = pm.sample(10000, tune=10000, init = 'advi + adapt_diag_grad', step=pm.NUTS(target_accept = target_accept), cores=1, progressbar=True)
    az.plot_trace(trace)
    display(az.summary(trace, round_to = 4))
    if(plot_ppc is True):
        pp = pm.sampling.fast_sample_posterior_predictive(trace)
        data_p = az.from_pymc3(trace, posterior_predictive=pp)
        az.plot_ppc(data_p, 'cumulative', alpha = .5)
#plot predictive traces by sampling final posteriors
