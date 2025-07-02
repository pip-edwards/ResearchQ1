"""
Created: 07.02.2025

Author: Pippa Edwards

Sources used: xarray documentation, ChatGPT

This code:
- Writes a new SST file that is a monthly climatology and overall climatology
- Writes a new regridded Chla to the same grid as SST
- Writes a new Chla file that is a monthly climatology and overall climatology
- Writes a new Z_Eu file that is an overall climatology
"""

#%%
#packages:
import xarray as xr
import pandas as pd
import numpy as np
import datetime as dt
import glob

#%%
#set up datapath and months
climdatapath = "C:/Users/pe1n24/Desktop/poc_data_070525/"
months = np.arange(1,13,1)

#%%
#SST

#open SST
sst_all = xr.open_dataset("ALL_SST_0207.nc")["SST"]
#limit to matched dataset time frame
sst_all = sst_all.sel(time = slice(dt.datetime(1997,9,1), dt.datetime(2022,7,1)))
print(np.min(sst_all["time"]).values,np.max(sst_all["time"]).values)

#make monthly climatology
for m in months:
     print(m)
     if m == 1:
         print(m)
         msst = sst_all.sel(time = sst_all.time.dt.month == m)
         msst = msst.mean(dim = "time", skipna= True)
         msst = msst.assign_coords({"month": m})
     else: 
         print(m)
         msst1 = sst_all.sel(time = sst_all.time.dt.month == m)
         msst1 = msst1.mean(dim = "time", skipna= True)
         msst1 = msst1.assign_coords({"month": m})
         print(np.min(msst1).values, np.max(msst1).values)
         msst = xr.concat([msst, msst1], dim = "month")
print(np.min(msst).values, np.max(msst).values)
msst.to_netcdf(f"{climdatapath}SST_monthly_clim_2606.nc")

#make overall climatology
sst = msst.mean(dim = "month")
print(np.min(sst).values, np.max(sst).values)
sst.to_netcdf(f"{climdatapath}SST_notimedim_clim_2606.nc")

#%%
#CHLA files

#make list of files as they are yearly
chlas = []
for file in glob.glob("C:/Users/pe1n24/Desktop/poc_data_070525/oc-cci/*.nc"):
     chlas.append(file)

#%%
#define slicing function

def get_slice(l):

    if l >= 1:
        l = round(l)
        return l-0.01, l-1
    elif l >= 0:
        return 0, 1
    else:
        l = round(l) + 1
        return l, l-0.99

#%%
#regrid and save new yearly regridded data

start_data = xr.open_dataset(chlas[0])["chlor_a"]

#get lat and lon values of sst to map to
slats = sst_all["lat"].values
slons = sst_all["lon"].values

for c in chlas[8:13]:
    start_data = xr.open_dataset(c)["chlor_a"]
    year = c[-7:-3]
    print(year)
    all_points = []
    for lat in slats:
        #print(lat)
        latu, latl = get_slice(lat)
        
        row_points = []
        for lon in slons:
            lonu, lonl = get_slice(lon)
            
            if lat == slats[0] and lon == slons[0]:
                point0 = start_data.sel(lat = slice(latu, latl))
                if point0.shape[1] == 0:
                    point0 = start_data.sel(lat = slice(latl, latu))
                point00 = point0.sel(lon = slice(lonu, lonl))
                if point00.shape[2] == 0:
                    point00 = point0.sel(lon = slice(lonl, lonu))
                #print(point00.shape)
                point00 = point00.mean(dim = ["lat", "lon"], skipna = True)
                point00 = point00.expand_dims({"lat": [lat], "lon": [lon]})
                row_points.append(point00)
            else:
                pointl = start_data.sel(lat = slice(latu, latl))
                if pointl.shape[1] == 0:
                    pointl = start_data.sel(lat = slice(latl, latu))
                pointll = pointl.sel(lon = slice(lonu, lonl))
                if pointll.shape[2] == 0:
                    pointll = pointl.sel(lon = slice(lonl, lonu))
                #print(pointll.shape)
                pointll = pointll.mean(dim = ["lat", "lon"], skipna = True)    
                pointll = pointll.expand_dims({"lat": [lat], "lon": [lon]})
                #point00 = xr.merge([point00, pointll])
                row_points.append(pointll)
                pointll = pointll.close()
                
        row = xr.concat(row_points, dim="lon")
        all_points.append(row)
    
    result = xr.concat(all_points, dim = "lat")
    result.to_netcdf(f"C:/Users/pe1n24/Desktop/poc_data_070525/new_regridded_occci_chla_{year}.nc")

#%%
#open new data

chlas = []
for file in glob.glob("C:/Users/pe1n24/Desktop/poc_data_070525/new_regridded*.nc"):
     chlas.append(file)
     print(file)

#%%

#join altogether in one dataframe
chlaxs = []
for c in chlas:
    print(c[-7:-3])
    chla1 = xr.open_dataset(c)["chlor_a"]
    chlaxs.append(chla1)
chla = xr.concat(chlaxs, dim = "time")

#filter to time
chla = chla.sel(time = slice(dt.datetime(1997,9,1), dt.datetime(2022,7,1)))

#make monthly climatology
mchlas = []
for m in np.arange(1,13,1):
    
    mchla = chla.sel(time = chla.time.dt.month == m)
    mchla = mchla.mean(dim = "time", skipna= True)
    mchla = mchla.assign_coords({"month": m})
    mchlas.append(mchla)
    print(m,np.nanmean(mchla))
chlam = xr.concat(mchlas, dim = "month")
print(np.nanmean(chlam))
chlam.to_netcdf("C:/Users/pe1n24/Desktop/poc_data_070525/occci_monthly_climatology_regridded.nc")

#make pverall climatology
chla = chla.mean(dim = "time")
chla.to_netcdf("C:/Users/pe1n24/Desktop/poc_data_070525/occci_notimedim_climatology_regridded.nc")
