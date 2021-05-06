"""
Script for calculating pattern correlations between models and observations

Author     : Zachary M. Labe
Date       : 6 May 2021
Version    : 1
"""

### Import packages
import sys
import matplotlib.pyplot as plt
import numpy as np
import calc_Utilities as UT
import calc_dataFunctions as df
import calc_Stats as dSS
import scipy.stats as sts
import calc_DetrendData as DET
from netCDF4 import Dataset

### Plotting defaults 
plt.rc('text',usetex=True)
plt.rc('font',**{'family':'sans-serif','sans-serif':['Avant Garde']}) 

###############################################################################
###############################################################################
###############################################################################
### Data preliminaries 
modelGCMs = ['CanESM2','MPI','CSIRO-MK3.6','KNMI-ecearth','GFDL-CM3','GFDL-ESM2M','LENS']
letters = ["a","b","c","d","e","f","g","h","i","j","k","l","m"]
datasetsingle = ['SMILE']
dataset_obs = 'ERA5BE'
monthlychoiceq = ['annual','JFM','AMJ','JAS','OND']
variables = ['T2M','P','SLP']
reg_name = 'SMILEGlobe'
level = 'surface'
timeper = 'historical'
###############################################################################
###############################################################################
land_only = False
ocean_only = False
###############################################################################
###############################################################################
baseline = np.arange(1951,1980+1,1)
###############################################################################
###############################################################################
window = 0
yearsall = np.arange(1950+window,2019+1,1)
###############################################################################
###############################################################################
numOfEns = 16
lentime = len(yearsall)
###############################################################################
###############################################################################
dataset = datasetsingle[0]
lat_bounds,lon_bounds = UT.regions(reg_name)
###############################################################################
###############################################################################
ravelyearsbinary = False
ravelbinary = False
lensalso = True
randomalso = False
shuffletype = 'none'
###############################################################################
###############################################################################
###############################################################################
###############################################################################
### Read in model data
def read_primary_dataset(variq,dataset,monthlychoice,numOfEns,lensalso,randomalso,ravelyearsbinary,ravelbinary,shuffletype,timeper,lat_bounds=lat_bounds,lon_bounds=lon_bounds):
    data,lats,lons = df.readFiles(variq,dataset,monthlychoice,numOfEns,lensalso,randomalso,ravelyearsbinary,ravelbinary,shuffletype,timeper)
    datar,lats,lons = df.getRegion(data,lats,lons,lat_bounds,lon_bounds)
    print('\nOur dataset: ',dataset,' is shaped',data.shape)
    return datar,lats,lons
def read_obs_dataset(variq,dataset_obs,numOfEns,lensalso,randomalso,ravelyearsbinary,ravelbinary,shuffletype,lat_bounds=lat_bounds,lon_bounds=lon_bounds):
    data_obs,lats_obs,lons_obs = df.readFiles(variq,dataset_obs,monthlychoice,numOfEns,lensalso,randomalso,ravelyearsbinary,ravelbinary,shuffletype,timeper)
    data_obs,lats_obs,lons_obs = df.getRegion(data_obs,lats_obs,lons_obs,
                                            lat_bounds,lon_bounds)    
    print('our OBS dataset: ',dataset_obs,' is shaped',data_obs.shape)
    return data_obs,lats_obs,lons_obs
def calcTrend(data):
    slopes = np.empty((data.shape[1],data.shape[2]))
    x = np.arange(data.shape[0])
    for i in range(data.shape[1]):
        for j in range(data.shape[2]):
            mask = np.isfinite(data[:,i,j])
            y = data[:,i,j]
            
            if np.sum(mask) == y.shape[0]:
                xx = x
                yy = y
            else:
                xx = x[mask]
                yy = y[mask]      
            if np.isfinite(np.nanmean(yy)):
                slopes[i,j],intercepts, \
                r_value,p_value,std_err = sts.linregress(xx,yy)
            else:
                slopes[i,j] = np.nan
    
    dectrend = slopes * 10.   
    print('Completed: Finished calculating trends!')      
    return dectrend

def readControl(monthlychoice,lat_bounds,lon_bounds):
    directorydata2 = '/Users/zlabe/Data/CMIP6/CESM2/picontrol/monthly/'
    data = Dataset(directorydata2 + 'T2M_000101-120012.nc')
    tempcc = data.variables['T2M'][:12000,:,:] # 1000 year control
    latc = data.variables['latitude'][:]
    lonc = data.variables['longitude'][:]
    data.close()
    
    tempc,latc,lonc = df.getRegion(tempcc,latc,lonc,lat_bounds,lon_bounds)
    datan = np.reshape(tempc,(tempc.shape[0]//12,12,tempc.shape[1],tempc.shape[2]))
    if monthlychoice == 'annual':
        datanq = np.nanmean(datan[:,:,:,:],axis=1)
    else:
        datanq = datan
    return datanq

###############################################################################
###############################################################################
###############################################################################
###############################################################################
### Call functions
# for vv in range(len(variables)):
#     for mo in range(len(monthlychoiceq)):
for vv in range(1):
    for mo in range(1):
        variq = variables[vv]
        monthlychoice = monthlychoiceq[mo]
        directorydata = '/Users/zlabe/Documents/Research/ModelComparison/Data/Climatologies/'
        directoryfigure = '/Users/zlabe/Desktop/ModelComparison_v1/Climatologies/patternCorr/%s/' % variq
        saveData =  monthlychoice + '_' + variq + '_' + reg_name + '_' + timeper
        print('*Filename == < %s >' % saveData) 
        
        ### Read data
        models,lats,lons = read_primary_dataset(variq,dataset,monthlychoice,numOfEns,
                                                lensalso,randomalso,ravelyearsbinary,
                                                ravelbinary,shuffletype,timeper,
                                                lat_bounds,lon_bounds)
        data_obs,lats_obs,lons_obs = read_obs_dataset(variq,dataset_obs,numOfEns,
                                                      lensalso,randomalso,ravelyearsbinary,
                                                      ravelbinary,shuffletype,lat_bounds,
                                                      lon_bounds)
        
        ### Only 70 years so through 2090 if future (2020-2089)
        if timeper == 'future':
            models = models[:,:,:70,:,:]
            yearsall = np.arange(2020,2089+1,1)
            baseline = np.arange(2021,2050+1,1)
        
        ### Add on additional "model" which is a multi-model mean
        modelmean = np.nanmean(models,axis=0)[np.newaxis,:,:,:,:]
        modelsall = np.append(models,modelmean,axis=0)
        
        ### Meshgrid of lat/lon
        lon2,lat2 = np.meshgrid(lons,lats)
        
        ### Read in control
        con = readControl(monthlychoice,lat_bounds,lon_bounds)
        trendcon = np.empty((len(con)//30,con.shape[1],con.shape[2]))
        for count,i in enumerate(range(0,len(con)-30,30)):
            trendcon[count,:,:,] = calcTrend(con[i:i+30,:,:])

        ##############################################################################
        ##############################################################################
        ##############################################################################
        ### Process trends
        obstrend = calcTrend(data_obs)
        modeltrends = np.empty((modelsall.shape[0],modelsall.shape[1],modelsall.shape[3],modelsall.shape[4]))
        for i in range(modeltrends.shape[0]):
            for e in range(modeltrends.shape[1]):
                modeltrends[i,e,:,:] = calcTrend(modelsall[i,e,:,:,:])
                
        ## Calculate SNR
        constd = np.nanstd(trendcon,axis=0)
        
        SNRobs = obstrend/constd
        SNRmodels = modeltrends/constd
                
        ### Begin function to correlate observations with model, ensemble
        corrsnr = np.empty((SNRmodels.shape[0],SNRmodels.shape[1]))
        for mo in range(SNRmodels.shape[0]):
            for ens in range(SNRmodels.shape[1]):
                varxsnr = SNRobs[:,:]
                varysnr = SNRmodels[mo,ens,:,:]
                corrsnr[mo,ens] = UT.calc_spatialCorr(varxsnr,varysnr,lats,lons,'yes')
        
        ### Average correlations across ensemble members
        meancorrsnr= np.nanmean(corrsnr,axis=1)      
        
        ##############################################################################
        ##############################################################################
        ##############################################################################
        ### Save correlations
        np.savez(directorydata + saveData + '_corrsSNR.npz',corrsnr)