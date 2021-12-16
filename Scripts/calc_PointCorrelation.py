"""
Script for calculating point by point correlations between models and observations

Author     : Zachary M. Labe
Date       : 4 May 2021
Version    : 1
"""

### Import packages
import sys
import matplotlib.pyplot as plt
import cmocean
import numpy as np
import calc_Utilities as UT
import calc_dataFunctions as df
import calc_Stats as dSS
import scipy.stats as sts

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
monthlychoiceq = ['annual']
variables = ['T2M']
reg_name = 'Arctic'
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

        ###############################################################################
        ###############################################################################
        ###############################################################################
        ### Begin point by point correlations
        
        ### Begin function to correlate observations with model, ensemble, year
        corr = np.empty((modelsall.shape[0],modelsall.shape[1],modelsall.shape[3],modelsall.shape[4]))
        for mo in range(modelsall.shape[0]):
            for ens in range(modelsall.shape[1]):
                for i in range(modelsall.shape[3]):
                    for j in range(modelsall.shape[4]):
                        varx = data_obs[:,i,j]
                        vary = modelsall[mo,ens,:,i,j]
                        corr[mo,ens,i,j] = sts.pearsonr(varx,vary)[0]
            print('Model #%s done!' % (mo+1))
                    
        ### Average correlations across ensemble members
        meancorr = np.nanmean(corr,axis=1)
        
        ##############################################################################
        ##############################################################################
        ##############################################################################
        ### Save correlations
        np.savez(directorydata + saveData + '_PointByPoint_corrs.npz',meancorr)
        np.savez(directorydata + saveData + '_PointByPoint_lats.npz',lats)
        np.savez(directorydata + saveData + '_PointByPoint_lons.npz',lons)
        
        ###############################################################################
        ###############################################################################
        ###############################################################################
        ### Begin point by point correlations after removing annual mean from each map
        modelsallglo, data_obsglo = dSS.remove_annual_mean(modelsall,data_obs,lats,lons,lats_obs,lons_obs)
        
        ### Begin function to correlate observations with model, ensemble, year
        corrglo = np.empty((modelsallglo.shape[0],modelsallglo.shape[1],modelsallglo.shape[3],modelsallglo.shape[4]))
        for mo in range(modelsallglo.shape[0]):
            for ens in range(modelsallglo.shape[1]):
                for i in range(modelsallglo.shape[3]):
                    for j in range(modelsallglo.shape[4]):
                        varxglo = data_obsglo[:,i,j]
                        varyglo = modelsallglo[mo,ens,:,i,j]
                        corrglo[mo,ens,i,j] = sts.pearsonr(varxglo,varyglo)[0]
            print('Model #%s done!' % (mo+1))
                    
        ### Average correlations across ensemble members
        meancorrglo = np.nanmean(corrglo,axis=1)
        
        ##############################################################################
        ##############################################################################
        ##############################################################################
        ### Save correlations
        np.savez(directorydata + saveData + '_PointByPoint_corrsGLO.npz',meancorrglo)