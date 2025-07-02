"""
Created: 02.07.2025

Author: Pippa Edwards NOCS/UoS
Lots and lots of help from: Dr Greg Britten WHOI

Sources used: https://mc-stan.org/cmdstanpy/users-guide.html, ChatGPT

This code:
- loads stan model that has sigma that varies with x stan_sigvar.stan
- inputs POC dataset into stan model
- extracts output and saves as csv
- input stan code written by Pippa Edwards and Dr Greg Britten
"""
#%%PACKAGES
import pandas as pd
import matplotlib.pyplot as plt
import cmdstanpy
import numpy as np
import seaborn as sns
import datetime as dt

filerun = dt.datetime.now()
#%%
#IMPORT AND COMPILE STAN CODE
mod = cmdstanpy.CmdStanModel(stan_file="C:\\Users\\pe1n24\\Desktop\\STAN\\stan_sigvar.stan")
# %%
#SET UP DATA 

poc = pd.read_csv("C:/Users/pe1n24/Desktop/poc_data_070525/flux_data_andvars_0107.csv")
poc = poc[poc["log_POC"]>= -2.5]
poc = poc[poc["Depth"] >= 100]
poc = poc.dropna()
#poc = poc[poc["Ocean_Name"] == "Atlantic Ocean"]
#%%
#the stan code has the layout which needs x a dataframe of the intercept and all other independent variables.
x = np.column_stack([
     np.ones(poc.shape[0]),# intercept = column of 1s
     poc["SST"],                      # SST#
     poc["log_Chla"],       # chla
     poc["log_Depth"]])#,     #depth


# x = np.column_stack([
#     np.ones(poc.shape[0]),# intercept = column of 1s
#     (poc["SST"]-np.mean(poc["SST"]))/np.std(poc["SST"]),                      # SST
#     (poc["log_Chla"]-np.mean(poc["log_Chla"]))/np.std(poc["log_Chla"]),       # chla
#     (poc["log_Depth"]-np.mean(poc["log_Depth"]))/np.std(poc["log_Depth"])])#,     #depth
#     #poc['Si']/np.std(poc["Si"]),                   # Si conc
#     #poc['N']/np.std(poc["N"]),                    # N conc
#     #poc['P']/np.std(poc["P"]),                    # P conc
#     #poc['Eu']/np.std(poc["Eu"])])                 # eu depth

xvars = ["Intercept", "SST", "logChla", "logDepth"]#,'Si', 'N', 'P', 'Eu']

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
#%%
#output as dataframes
sumstats =  mcmc.summary()
print(sumstats.head())

df = mcmc.draws_pd()
print(df.head())
#%%
#check mcmc output
sns.boxplot(x = sumstats["R_hat"])
plt.show()
print(np.nanmedian(sumstats["R_hat"]))
#THIS NEEDS TO BE CLOSE TO 1
#%%
#save outputs
filerun.strftime('%d_%m')
sumstats.to_csv(f"output/summary_stats_varsigma_stanrun_{filerun.strftime('%d_%m')}_25_{filerun.strftime('%H-%M')}.csv", index = False)
df.to_csv(f"output/all_data_varsigma_stanrun_{filerun.strftime('%d_%m')}_25_{filerun.strftime('%H-%M')}.csv", index = False)
