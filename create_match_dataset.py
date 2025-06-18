#Analysisng and combining all the diffferent data

#%%
def get_letter(y):
    if y >= 60:
        return "A"
    elif y >= 30:
        return "B"
    elif y >= 0:
        return "C"
    elif y>= -30:
        return "D"
    elif y>= -60:
        return "E"
    elif y>= -90:
        return "F"
    else:
        return "no_match"
        
def get_number(x):
    if x >= 120:
        return "6"
    elif x >= 60:
        return "6"
    elif x >= 0:
        return "4"
    elif x>= -60:
        return "3"
    elif x>= -120:
        return "2"
    elif x>= -180:
        return "1"
    else:
        return "no_match"       
#%%

import pandas as pd
import numpy as np
import seaborn as sns
import xarray as xr
import matplotlib
import matplotlib.pyplot as plt
import datetime as dt
matplotlib.rcParams['font.family'] = 'Arial'

# %%

pocdf = pd.read_csv("Global_POC_Database_2025-04-16.csv")
print(pocdf.columns)
#%%
'''
#Index(['poc', 'poc_sd', 'timestamp', 'time_recovered', 'duration',
       'time_deployed', 'depth', 'latitude', 'longitude', 'time_mid', 'poc_qc',
       'pressure', 'poc_unit', 'Pangaea_dataset_id', 'poc_method',
       'poc_principal_investigator', 'instrument', 'poc_comment',
       'poc_short_name', 'poc_name', 'flux_tot', 'flux_tot_unit',
       'duration_unit', 'depth_unit', 'elevation', 'Reference',
       'poc_converted', 'converted_date', 'converted_date_deployment',
       'converted_date_recovery', 'date_formatted', 'date_num', 'year',
       'month', 'season', 'New_category', 'ocean_name', 'on_land', 'lat_grid',
       'lon_grid', 'depth_grid', 'is_duplicate'],
      dtype='object')
'''

#remove those on land
pocdf = pocdf[pocdf["on_land"] == False]

#filter to the wanted columns (For now)
pocdf = pocdf[['depth', 'latitude', 'longitude','poc_converted', 'date_formatted',
                'year', 'month', 'season','ocean_name']]

pocdf = pocdf.dropna() #those lost are those that are missing a date or depth value - these could be used retrospectviely maybe?

#chla data does not go before 1997 so filter this
pocdf["date"] = pd.to_datetime(pocdf["date_formatted"], format = "%Y-%m")

pocdf = pocdf[pocdf["date"] > dt.datetime(1997, 9, 4)]


# %% MATCH VARIABILES

#load in sst dataset:
data_path = "C:/Users/pe1n24/OneDrive - University of Southampton/RQ1/Data"
sst = xr.open_dataset(f"{data_path}/ALL_SST_2802.nc")["SST"]

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

coastdepths = xr.open_dataset("GLO-MFC_001_024_mask_bathy.nc")["deptho"]

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
    day = dt.datetime(int(r["year"]), int(r["month"]), 15)

    lattemp = sst.sel(lat =lat, method = "nearest")
    lontemp = lattemp.sel(lon = tlon)
    daytemp = lontemp.sel(time = day, method = "nearest")
    ssts.append(np.nanmean(daytemp.values))
    
    #open the correct chla file for the lat and lon
    let = get_letter(lat)
    num = get_number(lon)

    pos = let+ str(num)
    chladata = xr.open_dataset(f"F:\\phd\\RQ1\\OC-CCI\\{pos}.nc")["chlor_a"]

    #find the mean oc-cci
    latchla = chladata.sel(lat = slice(lat+ 0.5, lat- 0.5))
    lonchla = latchla.sel(lon = slice(lon- 0.5, lon+ 0.5))
    daychla = lonchla.sel(time = day, method = "nearest")
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

pocdf["Coast_Depth"] = coasts
pocdf["Coast"] = ctype
pocdf["SST"] = ssts
pocdf["Chla"] = chlas
pocdf["Location"] = carios

print("DONE!")
# %%
pocdf = pocdf.dropna().reset_index(drop = True)
# %%

pocdf = pocdf[["date", "year", "month", "season", "latitude", "longitude", 
               "depth", "ocean_name", "Location", "Coast_Depth", "Coast", "SST", "Chla", "poc_converted"]]

pocdf= pocdf.rename(columns= {"date":"Date", "year" :"Year", "month":"Month",
                              "season":"Season", "latitude":"Latitude", "longitude":"Longitude", 
                              "depth":"Depth", "ocean_name":"Ocean_Name", "poc_converted":"POC"})
pocdf["log_Depth"] = np.log(pocdf["Depth"])
pocdf["log_Chla"] = np.log(pocdf["Chla"])
pocdf["log_POC"] = np.log(pocdf["POC"])



# %%
pocdf.to_csv("flux_data_andvars_0805.csv", index=False)
# %%
