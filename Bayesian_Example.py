# -*- coding: utf-8 -*-
"""
Created on Fri May 21 16:54:10 2021

@author: jmerri5
"""
from IPython.core.pylabtools import figsize
import numpy as np
from matplotlib import pyplot as plt
from tkinter import*
from tkinter import filedialog
# Create Tk root

root = Tk()
# Hide the main window
root.withdraw()
root.call('wm', 'attributes', '.', '-topmost', True)

import scipy.stats as stats
#Outputs a 2xNxM matrix where N is the number of files selected and M is the length of the data columns.
def import_data(isMultiple, T_max = 120):
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
    
# n_trials = [0, 1, 2, 3, 4, 5, 8, 15, 50, 500]
# data = stats.bernoulli.rvs(0.5, size=n_trials[-1])
# x = np.linspace(0, 1, 100)

# # For the already prepared, I'm using Binomial's conj. prior.
# for k, N in enumerate(n_trials):
#     sx = plt.subplot(len(n_trials) / 2, 2, k + 1)
#     plt.xlabel("$p$, probability of heads") \
#         if k in [0, len(n_trials) - 1] else None
#     plt.setp(sx.get_yticklabels(), visible=False)
#     heads = data[:N].sum()
#     y = dist.pdf(x, 1 + heads, 1 + N - heads)
#     plt.plot(x, y, label="observe %d tosses,\n %d heads" % (N, heads))
#     plt.fill_between(x, 0, y, color="#348ABD", alpha=0.4)
#     plt.vlines(0.5, 0, 4, color="k", linestyles="--", lw=1)

#     leg = plt.legend()
#     leg.get_frame().set_alpha(0.4)
#     plt.autoscale(tight=True)


# plt.suptitle("Bayesian updating of posterior probabilities",
#              y=1.02,
#              fontsize=14)

# plt.tight_layout()
#dat = import_data(False, 120)
is_fluorescence = False
if is_fluorescence is False:
    dat = np.loadtxt(r"C:\Users\jmerri5\OneDrive - Emory University\Polymer Drive\Jamie Merrill\Bayesian Analysis\example ellipsometry tg.txt", skiprows=1, unpack=True)
    count_data = dat[0] * 10000
    T = dat[1]
else:
    dat = import_data(False, 120)
    count_data = dat[1]
    T = dat[0]
figsize(12.5, 10)
#count_data = np.loadtxt(r"C:\Users\jmerri5\Desktop\txtdata.csv")
n_count_data = len(count_data)
plt.scatter(T, count_data, color="#348ABD", marker = "o")
plt.xlabel("Temperature (C)")
plt.ylabel("Counts")
plt.title("Is there a transition?")
plt.xlim(int(min(T)-1), int(max(T)+1));
plt.figure()   



import pymc3 as pm
import theano.tensor as tt
import arviz as az

with pm.Model() as model:
    alpha = 1.0/count_data.mean()  # normalize exponentials to count data
    lambda_1 = pm.Exponential("lambda_1", alpha) #slope 1 
    lambda_2 = pm.Exponential("lambda_2", alpha) #slope 2 
    tau = pm.Uniform("tau", lower=min(T), upper = max(T)) #switchpoint assigned to continuous uniform distributed random  over full T-range
    temperature = pm.Data("temp", T)
with model:
    idx = np.arange(n_count_data) # Index
    lambda_ = pm.math.switch(tau > idx, lambda_1, lambda_2)

with model:    
    observation = pm.Poisson("obs", lambda_, observed=count_data)
with model:
    trace = pm.sample(10000, tune=1000, init='jitter+adapt_diag', step = pm.NUTS(target_accept = 0.9), cores = 1)

### Mysterious code to be explained in Chapter 3.
lambda_1_samples = trace['lambda_1']
lambda_2_samples = trace['lambda_2']
tau_samples = trace['tau']


# figsize(12.5, 10)
# #histogram of the samples:

# ax = plt.subplot(311)

# plt.hist(lambda_1_samples, histtype='stepfilled', bins=30, alpha=0.85,
#          label="posterior of $\lambda_1$", color="#A60628", density=False)
# plt.legend(loc="upper left")
# plt.title(r"""Posterior distributions of the variables
#     $\lambda_1,\;\lambda_2,\;\tau$""")
# plt.xlabel("$\lambda_1$ value")

# ax = plt.subplot(312)
# plt.hist(lambda_2_samples, histtype='stepfilled', bins=30, alpha=0.85,
#          label="posterior of $\lambda_2$", color="#7A68A6", density=False,)
# plt.legend(loc="upper left")
# plt.xlabel("$\lambda_2$ value")

# ax = plt.subplot(313)
# w = 1.0 / tau_samples.shape[0] * np.ones_like(tau_samples)
# plt.hist(tau_samples, bins=30, alpha=1,
#          label=r"posterior of $\tau$",
#          color="green", weights=w, rwidth=2.)

# plt.legend(loc="upper left")
# plt.xlabel(r"$\tau$")
# plt.ylabel("probability");

plt.figure()
# tau_samples, lambda_1_samples, lambda_2_samples contain
# N samples from the corresponding posterior distribution
N = tau_samples.shape[0]
expected_texts_per_day = np.zeros(n_count_data)
for day in range(n_count_data):
    # ix is a bool index of all tau samples corresponding to
    # the switchpoint occurring prior to value of 'day'
    ix = day < tau_samples
    # Each posterior sample corresponds to a value for tau.
    # for each day, that value of tau indicates whether we're "before"
    # (in the lambda1 "regime") or
    #  "after" (in the lambda2 "regime") the switchpoint.
    # by taking the posterior sample of lambda1/2 accordingly, we can average
    # over all samples to get an expected value for lambda on that day.
    # As explained, the "message count" random variable is Poisson distributed,
    # and therefore lambda (the poisson parameter) is the expected value of
    # "message count".
    expected_texts_per_day[day] = ((lambda_1_samples[ix].sum()
                                    + lambda_2_samples[~ix].sum()) / N) 


plt.plot(T, expected_texts_per_day, lw=4, color="#E24A33",
          label="expected number of counts")
plt.xlabel("Temperature")
plt.ylabel("Expected # counts")
plt.title("Expected number of counts")
plt.scatter(T, count_data, color="#348ABD", alpha=.65,
        label="Fluorescence intensity")

plt.legend(loc="upper right");
with model:
    az.plot_trace(trace)
    display(az.summary(trace, round_to = 2))
