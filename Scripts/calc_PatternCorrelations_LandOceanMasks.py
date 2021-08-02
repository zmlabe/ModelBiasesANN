"""
Script for calculating pattern correlations between models and observations
for only land and ocean regions

Author     : Zachary M. Labe
Date       : 10 May 2021
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
for vv in range(len(variables)):
    for mo in range(len(monthlychoiceq)):
# for vv in range(1):
#     for mo in range(1):
        variq = variables[vv]
        monthlychoice = monthlychoiceq[mo]
        directorydata = '/Users/zlabe/Documents/Research/ModelComparison/Data/Climatologies/'
        directoryfigure = '/Users/zlabe/Desktop/ModelComparison_v1/Climatologies/patternCorr/%s/' % variq
        if land_only == True:
            saveData =  monthlychoice + '_LAND_' + variq + '_' + reg_name + '_' + timeper
        elif ocean_only == True:
            saveData =  monthlychoice + '_OCEAN_' + variq + '_' + reg_name + '_' + timeper
        else:
            sys.exit()
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

        ###############################################################################
        ###############################################################################
        ###############################################################################
        ### Begin spatial correlations
        
        ### Begin function to correlate observations with model, ensemble, year
        corr = np.empty((modelsall.shape[0],modelsall.shape[1],modelsall.shape[2]))
        for mo in range(modelsall.shape[0]):
            for ens in range(modelsall.shape[1]):
                for yr in range(modelsall.shape[2]):
                    varx = data_obs[yr,:,:]
                    vary = modelsall[mo,ens,yr,:,:]
                    corr[mo,ens,yr] = UT.calc_spatialCorr(varx,vary,lats,lons,'yesnan')
                    
        ### Average correlations across ensemble members
        meancorr = np.nanmean(corr,axis=1)
        
        ### Average across all years for each model
        meanallcorr = np.nanmean(meancorr,axis=1)
        
        ### Check highest member
        maxmodelcorr = np.argmax(meancorr,axis=0)
        uniquecorr,countcorr = np.unique(maxmodelcorr,return_counts=True)

        ###############################################################################
        ###############################################################################
        ###############################################################################
        ### Detrend data for correlations
        data_obsdt = DET.detrendDataR(data_obs,'surface','monthly')
        modelsalldt = DET.detrendData(modelsall,'surface','monthly')
        
        ### Begin function to correlate observations with model, ensemble, year
        corrdt = np.empty((modelsalldt.shape[0],modelsalldt.shape[1],modelsalldt.shape[2]))
        for mo in range(modelsalldt.shape[0]):
            for ens in range(modelsalldt.shape[1]):
                for yr in range(modelsalldt.shape[2]):
                    varxdt = data_obsdt[yr,:,:]
                    varydt = modelsalldt[mo,ens,yr,:,:]
                    corrdt[mo,ens,yr] = UT.calc_spatialCorr(varxdt,varydt,lats,lons,'yesnan')
                    
        ### Average correlations across ensemble members
        meancorrdt = np.nanmean(corrdt,axis=1)
        
        ### Average across all years for each model
        meanallcorrdt = np.nanmean(meancorrdt,axis=1)
        
        ### Check highest member
        maxmodelcorrdt = np.argmax(meancorrdt,axis=0)
        uniquecorrdt,countcorrdt = np.unique(maxmodelcorrdt,return_counts=True)

        ###############################################################################
        ###############################################################################
        ###############################################################################
        ### Remove annual mean
        modelsallglo, data_obsglo = dSS.remove_annual_mean(modelsall,data_obs,lats,lons,lats_obs,lons_obs)
        
        ### Begin function to correlate observations with model, ensemble, year
        corrglo = np.empty((modelsallglo.shape[0],modelsallglo.shape[1],modelsallglo.shape[2]))
        for mo in range(modelsallglo.shape[0]):
            for ens in range(modelsallglo.shape[1]):
                for yr in range(modelsallglo.shape[2]):
                    varxglo = data_obsglo[yr,:,:]
                    varyglo = modelsallglo[mo,ens,yr,:,:]
                    corrglo[mo,ens,yr] = UT.calc_spatialCorr(varxglo,varyglo,lats,lons,'yesnan')
                    
        ### Average correlations across ensemble members
        meancorrglo = np.nanmean(corrglo,axis=1)
        
        ### Average across all years for each model
        meanallcorrglo = np.nanmean(meancorrglo,axis=1)
        
        ### Check highest member
        maxmodelcorrglo = np.argmax(meancorrglo,axis=0)
        uniquecorrglo,countcorrglo = np.unique(maxmodelcorrglo,return_counts=True)
        
        ##############################################################################
        ##############################################################################
        ##############################################################################
        ## Process trends
        obstrend = calcTrend(data_obs)
        modeltrends = np.empty((modelsall.shape[0],modelsall.shape[1],modelsall.shape[3],modelsall.shape[4]))
        for i in range(modeltrends.shape[0]):
            for e in range(modeltrends.shape[1]):
                modeltrends[i,e,:,:] = calcTrend(modelsall[i,e,:,:,:])
                
        ### Begin function to correlate observations with model, ensemble
        corrtrends = np.empty((modeltrends.shape[0],modeltrends.shape[1]))
        for mo in range(modeltrends.shape[0]):
            for ens in range(modeltrends.shape[1]):
                varxtrends = obstrend[:,:]
                varytrends = modeltrends[mo,ens,:,:]
                corrtrends[mo,ens] = UT.calc_spatialCorr(varxtrends,varytrends,lats,lons,'yesnan')
        
        ### Average correlations across ensemble members
        meancorrtrends = np.nanmean(corrtrends,axis=1)      
        
        ##############################################################################
        ##############################################################################
        ##############################################################################
        ### Save correlations
        np.savez(directorydata + saveData + '_corrs.npz',corr)
        np.savez(directorydata + saveData + '_corrsdt.npz',corrdt)
        np.savez(directorydata + saveData + '_corrsglo.npz',corrglo)
        np.savez(directorydata + saveData + '_corrstrends.npz',corrtrends)