"""
Script for calculating RMSE between models and observations after 
standardizing data

Author     : Zachary M. Labe
Date       : 26 July 2021
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
monthlychoiceq = ['annual']
variables = ['T2M']
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
        if land_only == True:
            saveData =  monthlychoice + '_LAND_' + variq + '_' + reg_name + '_' + timeper
            typemask = 'LAND'
        elif ocean_only == True:
            saveData =  monthlychoice + '_OCEAN_' + variq + '_' + reg_name + '_' + timeper
            typemask = 'OCEAN'
        else:
            typemask = 'LAND/OCEAN'
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
        
###############################################################################        
###############################################################################                        
        if land_only == True:
            modelsall, data_obs = dSS.remove_ocean(modelsall,data_obs,
                                              lat_bounds,
                                              lon_bounds) 
            print('\n*Removed ocean data*')
###############################################################################
###############################################################################
        if ocean_only == True:
            modelsall, data_obs = dSS.remove_land(modelsall,data_obs,
                                              lat_bounds,
                                              lon_bounds) 
            print('\n*Removed land data*')  
###############################################################################
###############################################################################        
        ### Change 0s for mask to nans
        modelsall[modelsall == 0.] = np.nan
        data_obs[data_obs == 0.] = np.nan
        
        ### Meshgrid of lat/lon
        lon2,lat2 = np.meshgrid(lons,lats)
        
        ### Standardize all data
        def standardize(modelsall,data_obs):
            if modelsall.ndim == 5:
                modelsall_flatyrens = modelsall.reshape(modelsall.shape[0],
                                                        modelsall.shape[1]*modelsall.shape[2],
                                                        lats.shape[0],lons.shape[0])
                xmean = np.nanmean(modelsall_flatyrens,axis=1)[:,np.newaxis,np.newaxis,:,:]
                xstd = np.nanstd(modelsall_flatyrens,axis=1)[:,np.newaxis,np.newaxis,:,:]
                
                obsmean = np.nanmean(data_obs,axis=0)
                obstd = np.nanstd(data_obs,axis=0)
                
                modelsall_std = ((modelsall - xmean)/xstd).reshape(modelsall.shape[0],
                                                        modelsall.shape[1],modelsall.shape[2],
                                                        lats.shape[0],lons.shape[0])            
                obs_std = (data_obs - obsmean)/obstd
            else:
                print(ValueError('\nCheck dimensions!\n'))
                sys.exit()
            print('\n------STANDARDIZE EACH MODEL SEPARATELY------\n')
            return modelsall_std,obs_std
        modelsallstd,data_obsstd, = standardize(modelsall,data_obs)
        
        ### Calculate rmse per year
        rmsd = np.empty((modelsallstd.shape[0],modelsallstd.shape[1],modelsallstd.shape[2]))
        for mo in range(modelsallstd.shape[0]):
            for ens in range(modelsallstd.shape[1]):
                for yr in range(modelsallstd.shape[2]):
                    varxrm = data_obsstd[yr,:,:]
                    varyrm = modelsallstd[mo,ens,yr,:,:]
                    if any([land_only==True,ocean_only==True]):
                        rmsd[mo,ens,yr] = UT.calc_RMSE(varxrm,varyrm,lats,lons,'yesnan')
                        print('------USING MASKS FOR NANS!------')
                    else:
                        rmsd[mo,ens,yr] = UT.calc_RMSE(varxrm,varyrm,lats,lons,'yes')
        
        ### Average RMSE across ensemble members
        meanrmsd = np.nanmean(rmsd,axis=1)     
        
        ### Average RMSE across years
        meanrmsdyr = np.nanmean(meanrmsd,axis=1)

        ##############################################################################
        ##############################################################################
        ##############################################################################
        ### Remove annual mean from each map
        modelsallglo, data_obsglo = dSS.remove_annual_mean(modelsall,data_obs,lats,lons,lats_obs,lons_obs)
        modelsallstdglo,data_obsstdglo, = standardize(modelsallglo,data_obsglo)
        
        ### Calculate rmse per year
        rmsdglo = np.empty((modelsallstdglo.shape[0],modelsallstdglo.shape[1],modelsallstdglo.shape[2]))
        for mo in range(modelsallstdglo.shape[0]):
            for ens in range(modelsallstdglo.shape[1]):
                for yr in range(modelsallstdglo.shape[2]):
                    varxrmglo = data_obsstdglo[yr,:,:]
                    varyrmglo = modelsallstdglo[mo,ens,yr,:,:]
                    if any([land_only==True,ocean_only==True]):
                        rmsdglo[mo,ens,yr] = UT.calc_RMSE(varxrmglo,varyrmglo,lats,lons,'yesnan')
                        print('------USING MASKS FOR NANS!------')
                    else:
                        rmsdglo[mo,ens,yr] = UT.calc_RMSE(varxrmglo,varyrmglo,lats,lons,'yes')
        
        ### Average RMSE across ensemble members
        meanrmsdglo = np.nanmean(rmsdglo,axis=1)     
        
        ### Average RMSE across years
        meanrmsdyrglo = np.nanmean(meanrmsdglo,axis=1)

        ##############################################################################
        ##############################################################################
        ##############################################################################
        ### Save correlations
        np.savez(directorydata + saveData + '_RMSE_standardizedData.npz',rmsd)
        np.savez(directorydata + saveData + '_RMSEglo_standardizedData.npz',rmsdglo)