'''
set up varaibles
use loop of stan for x
check rhat values by printing summary
make graphs that are histograms, cdfs, plots of betas agai0nst eachother, 
make graphs of estimates of the relationships (add a random noise scaled to the sigma)
'''

#%%PACKAGES
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cmdstanpy
import numpy as np
import xarray as xr
import random 
import os
import datetime as dt
import scipy.stats as stats
filerun = dt.datetime.now()
os.chdir("C:\\Users\\pe1n24\\Desktop\\STAN")

cs = ["#e9d985","#b2bd7e","#749c75","#6a5d7b","#5d4a66", "#ED9390"]
# %%
#IMPORT AND COMPILE STAN CODE
mod = cmdstanpy.CmdStanModel(stan_file="C:\\Users\\pe1n24\\Desktop\\STAN\\BASE_POC_LR.stan")
# %%
#SET UP DATA 

poc = pd.read_csv("C:/Users/pe1n24/Desktop/poc_data_070525/flux_data_andvars_0805.csv")
poc = poc[poc["log_POC"]>= -2.5]
poc = poc[poc["Depth"] >= 100]
#poc = poc[poc["Ocean_Name"] == "Atlantic Ocean"]
#%%
#the stan code has the layout which needs x a dataframe of the intercept and all other independent variables.

x = np.column_stack([
    np.ones(poc.shape[0]),# intercept = column of 1s
    poc["SST"]/np.std(poc["SST"]),                      # SST
    poc["log_Chla"]/np.std(poc["log_Chla"]),       # chla
    poc["log_Depth"]/np.std(poc["log_Depth"])])   # depth

xvars = ["Intercept", "SST", "logChla", "logDepth"]

#set up the dataframe as a dict:
data = {"N": poc.shape[0],  #number of observations
        "p": x.shape[1],    #number of x variables
        "y": poc["log_POC"],#dependent variable
        "x": x}             #independent variables


# %%
#RUN THE STAN MODEL and save the output dataframes
mcmc = mod.sample(data = data, chains = 4, iter_sampling=2000)

#this will generate both a sigma value for each data point and an predicted y value for each data point 8000 times. 

#%%
sumstats =  mcmc.summary()
print(sumstats.head())

df = mcmc.draws_pd()
print(df.head())

#%% save
filerun.strftime('%d_%m')
sumstats.to_csv(f"output/summary_stats_stanrun_{filerun.strftime('%d_%m')}_25_{filerun.strftime('%M-%H')}.csv", index = False)
df.to_csv(f"output/all_data_stanrun_{filerun.strftime('%d_%m')}_25_{filerun.strftime('%M-%H')}.csv", index = False)
# %%
#CHECK RHAT VALUES
sns.boxplot(x = sumstats["R_hat"], color = cs[0])
plt.show()
print(np.nanmedian(sumstats["R_hat"]))

#if very close to 1, continue!


# %%
#EXTRACT BETA AND GAMMA DATA

betas = df.filter(like = "beta")
gammas = df.filter(like = "gamma")

#join together
posts = pd.concat([betas, gammas], axis=1)
cols = list(posts.columns.values)
# %%

fig, ax = plt.subplots(gammas.shape[1], 2, figsize = (10, gammas.shape[1]*2.5), dpi = 150, sharex = True)
fig.tight_layout()
for n in range(posts.shape[1]):
    if n >= gammas.shape[1]:
        xpos, ypos = n-4, 1
    else:
        xpos, ypos = n, 0
    col = cols[n]
    sns.scatterplot(x = np.arange(0, posts.shape[0]), y = posts[col], ax = ax[xpos, ypos],
                    s = 10, legend = False, color = cs[2])
    ax[xpos,ypos].set(ylabel = "", title = col)
plt.xlim([-50,8050])
plt.show()

#%% HISTOGRAM OF POSTERIORS
fig, ax = plt.subplots(gammas.shape[1], 2, figsize = (10, gammas.shape[1]*2.5), dpi = 150)
fig.tight_layout()
for n in range(posts.shape[1]):
    if n >= gammas.shape[1]:
        xpos, ypos = n-4, 1
    else:
        xpos, ypos = n, 0
    col = cols[n]
    print(col)
    sns.histplot(x = posts[col], ax = ax[xpos, ypos], color = cs[2], alpha = 0.8)
    ax[xpos, ypos].axvline(sumstats["50%"][col], color = "grey", linestyle = "-", label = "Median")
    ax[xpos, ypos].axvline(sumstats["Mean"][col], color = "black", linestyle = "-.", label = "Mean")
    ax[0,0].legend()
    ax[xpos, ypos].set(ylabel = "", xlabel = f" {col}: {xvars[xpos]}")
    ax[0,0].set_title("Beta Values (y ~)")
    ax[0,1].set_title("Gamma Values (sigma ~)")

#%% 
#POSTERIOR VS POSTERIOR

fig, ax = plt.subplots(posts.shape[1], posts.shape[1], figsize = (18,18), sharex='col', sharey='row')
for a in range(posts.shape[1]):
    acol = cols[a]
    for b in range(posts.shape[1]):
        bcol= cols[b]
        ano = acol[-2]
        bno = bcol[-2]
        ydata = posts[f"{acol}"]
        xdata = posts[f"{bcol}"]
        sns.scatterplot(x = xdata, y = ydata, ax = ax[a,b], size = 8, color = cs[2], legend = False)
        ax[a,b].set(xlabel = "", ylabel = "")
for a in range(posts.shape[1]):
    acol = cols[a]
    ax[a,0].set_ylabel(f"{acol}: {xvars[int(acol[-2])-1]}", fontsize = 10, weight = "bold")
    ax[0,a].set_title(f"{acol}: {xvars[int(acol[-2])-1]}", fontsize = 10, weight = "bold")       


# %% Y AGAINST Y

# %%
ysims = df.filter(like = "y_sim")
sigmas = df.filter(like = "sigma")
mus = df.filter(like = "mu")
# %%
fig, ax=  plt.subplots(2,1, figsize = (10, 10), dpi = 150, sharex = True)
fig.tight_layout()
sns.scatterplot(x = poc["log_POC"], y = ysims.median().tolist(), ax = ax[0], color = cs[1])
sns.scatterplot(x = poc["log_POC"], y = sigmas.median().tolist(), ax = ax[1], color = cs[1])
ax[0].set(ylabel = "Median Predicted logPOC")
#ax[0].set_ylim([-9, 9])
ax[1].set(ylabel = "Median Sigma", xlabel = "Actual logPOC")


# %%


#take mean mu and sigma for each value of x. take a random one from this normal distribution for each x
mus = df.filter(like = "mu")

m_mus = mus.mean().tolist()
m_sigmas = sigmas.mean().tolist()

y_new = []
for i in range(len(m_mus)):
    y_new.append(float(np.random.normal(m_mus[i], np.exp(m_sigmas[i]),1)))
#%%

y_sig0 = []
for i in range(len(m_mus)):
    y_sig0.append(float(np.random.normal(m_mus[i], 0,1)))
#%%
labels = ["Realised log_POC from model with sigma varying with predictors",
          "Realised log_POC from model with sigma = 0"]
fig, ax=  plt.subplots(2,1, figsize = (10, 8), dpi = 150, sharex = True)
sns.scatterplot(x = poc["log_POC"], y = y_new, color = cs[1], ax = ax[0])
sns.scatterplot(x = poc["log_POC"], y = y_sig0, color = cs[5], ax = ax[1])

for n in range(2):
    ax[n].axline((0,0), (1,1), color = "black", ls = "--")
    ax[n].set_ylabel(f"{labels[n]}")
ax[0].set_title(f"pearson correlation = {stats.pearsonr(poc['log_POC'], y = y_new)[0]}")
ax[1].set_title(f'pearson correlation = {stats.pearsonr(poc["log_POC"], y = y_sig0)[0]}')
#8000 guesses of mu for each of the poc measurements
#for n in range(poc):
#%%
#mu = b1 + b2*tn + b3*cn + b4*dn
simpoc = ysims.apply(lambda col: col.sample(n=1).iloc[0], axis=0).tolist()

sns.scatterplot(x = poc["log_POC"], y = simpoc, color = cs[1])
plt.axline((0,0), (1,1), color = "black", ls = "--")
plt.show()
#take average mu and sigma of 8000 guesses for each datapoint 

#randomly sample from that normal distribution

#realisation of eff*

#some of thie sgima i will get will be from measurement error other than other sources
#%%

fig, ax=  plt.subplots(2,1, figsize = (10, 8), dpi = 150)

for i in range(2):
    sns.histplot(x = poc["log_POC"], ax = ax[i], alpha = 0.4, color = "black")
    ax[i].set_xlabel(f"{labels[i]}")
sns.histplot(x = y_new, ax = ax[0], alpha = 0.7, color = cs[1])
sns.histplot(x = y_sig0, ax = ax[1], alpha = 0.7, color = cs[5])


#%%
#map of sigma over time and space

#%%

#how much of the bCP is due to fluxuations in the POC flux
#estimate total global export using my model
#do the same and set sigma to 0 - how much lower it is without the variability