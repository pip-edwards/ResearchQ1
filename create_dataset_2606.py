"""
Created: 26.06.2025

Author: Pippa Edwards NOCS/UoS

Sources used: Xarray user guide, ChatGPT

This code:
- loads in POC dataset and then matches it to SST and Chla
- transforms paramaters
- saves as csv
"""

#%%
#import packages
import pandas as pd
import numpy as np
import xarray as xr
import matplotlib
import matplotlib.pyplot as plt
import datetime as dt
matplotlib.rcParams['font.family'] = 'Arial'
import os

#%%

pocdf = pd.read_csv("C:\\Users\\pe1n24\\Desktop\\poc_data_070525\Global_POC_Database_2025-04-16.csv")
#print(pocdf.columns)
#%%

#remove those on land
pocdf = pocdf[pocdf["on_land"] == False]

#filter to the wanted columns (For now)
pocdf = pocdf[['depth', 'latitude', 'longitude','poc_converted', 'date_formatted',
                'year', 'month', 'season','ocean_name']]

pocdf = pocdf.dropna() #those lost are those that are missing a date or depth value - these could be used retrospectviely maybe?

#format date
pocdf["date"] = pd.to_datetime(pocdf["date_formatted"], format = "%Y-%m")

#the sst finishes in 2022 and the chla starts in 1997.
#bound the dataset to this.
pocdf = pocdf[pocdf["date"] >= dt.datetime(1997, 9, 1)]
pocdf = pocdf[pocdf["date"] < dt.datetime(2022, 7, 1)]


#load in sst dataset:
data_path = "C:/Users/pe1n24/OneDrive - University of Southampton/RQ1/Data"
os.chdir("C:/Users/pe1n24/Desktop/poc_data_070525/")

sst = xr.open_dataset(f"{data_path}/ALL_SST_0207.nc")["SST"]
#this has been converted into -180to180.
#print(sst["lon"])
#if it has not:
# lons = np.concatenate((np.arange(0.5, 180.5, 1), np.arange(-179.5, 0.5,1)))
# sst = sst.assign_coords(lon = lons)
# sst = sst.sortby("lon")

#%%
#add sst and chla to these data points:
ssts = []
chlas = []
#add if it is in the cariaco basin
carios = []
#cariaco lats and lons are only at these specific lons so will only remove when there
clons = [-64.67, -64.4]

#coast data
coasts = []
ctype = []

#load in coastal bathymetry map from CMEMS
coastdepths = xr.open_dataset("GLO-MFC_001_024_mask_bathy.nc")["deptho"]

#loop through
for i,r in pocdf.iterrows():
    if i % 1000 == 0:
        print(i)

    lat = r["latitude"]
    lon = r["longitude"]

    #set lat and lon for temperature files
    if lon < 0 :
        tlon = round(lon) - 0.5
    elif lon == 180:
        tlon = -0.5
    else:
        tlon = round(lon) + 0.5

    #set the day as the middate of the month to match to SST
    sday = dt.datetime(int(r["year"]), int(r["month"]), 15)

    #extract the sst value for this datapoint
    lattemp = sst.sel(lat =lat, method = "nearest")
    lontemp = lattemp.sel(lon = tlon)
    daytemp = lontemp.sel(time = sday, method = "nearest")
    ssts.append(np.nanmean(daytemp.values))

    #open the correct chla file
    chla = xr.open_dataset(f"oc-cci/occci_chla_{int(r['year'])}.nc")["chlor_a"]

    #set the date to match the chla days
    cday = dt.datetime(int(r["year"]), int(r["month"]), 1)
    
    #find mean oc-cci
    latchla = chla.sel(lat = slice(lat+ 0.5, lat- 0.5))
    lonchla = latchla.sel(lon = slice(lon- 0.5, lon+ 0.5))
    daychla = lonchla.sel(time = cday, method = "nearest")
    chlas.append(np.nanmean(daychla.values))

    #find if it is in the cariaco basin
    if lon in clons:
        carios.append("Cariaco")
    else:
        carios.append("Not")

    #find if it is within 200m shelf aka coastal
    dval = coastdepths.sel(latitude = lat, method = "nearest")
    dval = dval.sel(longitude = lon, method = "nearest")
    cdepth = np.nanmean(dval)
    if pd.isna(cdepth) == True:
        pocdflat = pocdf[pocdf["latitude"] == pocdf["latitude"][i]]
        pocdflon = pocdflat[pocdflat["longitude"] == pocdflat["longitude"][i]]
        cdepth= np.max(pocdflon["depth"])
    coasts.append(cdepth)
    if cdepth <= 200:
        ctype.append("Coastal Ocean")
    else:
        ctype.append("Open Ocean")

#add all to dataframe
pocdf["Coast_Depth"] = coasts
pocdf["Coast"] = ctype
pocdf["SST"] = ssts
pocdf["Chla"] = chlas
pocdf["Location"] = carios
    
# %%

#reformat dataframe
pocdf = pocdf[["date", "year", "month", "season", "latitude", "longitude", 
               "depth", "ocean_name", "Location", "Coast_Depth", "Coast", "SST", "Chla", "poc_converted"]]

pocdf= pocdf.rename(columns= {"date":"Date", "year" :"Year", "month":"Month",
                              "season":"Season", "latitude":"Latitude", "longitude":"Longitude", 
                              "depth":"Depth", "ocean_name":"Ocean_Name", "poc_converted":"POC"})

#make log constituents
pocdf["log_Depth"] = np.log(pocdf["Depth"])
pocdf["log_Chla"] = np.log(pocdf["Chla"])
pocdf["log_POC"] = np.log(pocdf["POC"])


print(pocdf.shape)
# %%

#save csv
pocdf.to_csv("flux_data_andvars_0107.csv", index=False)

# %%
