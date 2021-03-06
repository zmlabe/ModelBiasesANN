"""
Functions are useful statistical untilities for data processing in the ANN
 
Notes
-----
    Author : Zachary Labe
    Date   : 15 July 2020
    
Usage
-----
    [1] rmse(a,b)
    [2] pickSmileModels(data,modelGCMs,pickSMILE)
    [3] remove_annual_mean(data,data_obs,lats,lons,lats_obs,lons_obs)
    [4] remove_merid_mean(data,data_obs)
    [5] remove_observations_mean(data,data_obs,lats,lons)
    [6] calculate_anomalies(data,data_obs,lats,lons,baseline,yearsall)
    [7] remove_ensemble_mean(data,ravel_modelens,ravelmodeltime,rm_standard_dev,numOfEns)
    [8] remove_ocean(data,data_obs)
    [9] remove_land(data,data_obs)
    [10] standardize_data(Xtrain,Xtest)
    [11] rm_standard_dev(var,window,ravelmodeltime,numOfEns)
    [12] rm_variance_dev(var,window)
"""

def rmse(a,b):
    """
    Calculates the root mean squared error
    takes two variables, a and b, and returns value
    """
    
    ### Import modules
    import numpy as np
    
    ### Calculate RMSE
    rmse_stat = np.sqrt(np.mean((a - b)**2))
    
    return rmse_stat

###############################################################################
    
def pickSmileModels(data,modelGCMs,pickSMILE):
    """
    Select models to analyze if using a subset
    """
    
    ### Pick return indices of models
    lenOfPicks = len(pickSMILE)
    indModels = [i for i, item in enumerate(modelGCMs) if item in pickSMILE]
    
    ### Slice data
    if data.shape[0] == len(modelGCMs):
        if len(indModels) == lenOfPicks:
            modelSelected = data[indModels]
        else:
            print(ValueError('Something is wrong with the indexing of the models!'))
    else:
        print(ValueError('Something is wrong with the order of the data!'))
    
    return modelSelected

###############################################################################

def remove_annual_mean(data,data_obs,lats,lons,lats_obs,lons_obs):
    """
    Removes annual mean from data set
    """
    
    ### Import modulates
    import numpy as np
    import calc_Utilities as UT
    
    ### Create 2d grid
    lons2,lats2 = np.meshgrid(lons,lats)
    lons2_obs,lats2_obs = np.meshgrid(lons_obs,lats_obs)
    
    ### Calculate weighted average and remove mean
    data = data - UT.calc_weightedAve(data,lats2)[:,:,:,np.newaxis,np.newaxis]
    data_obs = data_obs - UT.calc_weightedAve(data_obs,lats2_obs)[:,np.newaxis,np.newaxis]
    
    return data,data_obs

###############################################################################

def remove_merid_mean(data,data_obs,lats,lons,lats_obs,lons_obs):
    """
    Removes meridional mean from data set
    """
    
    ### Import modules
    import numpy as np
    
    ### Remove mean of latitude
    data = data - np.nanmean(data,axis=3)[:,:,:,np.newaxis,:]
    data_obs = data_obs - np.nanmean(data_obs,axis=1)[:,np.newaxis,:]

    return data,data_obs

###############################################################################

def remove_observations_mean(data,data_obs,lats,lons):
    """
    Removes observations to calculate model biases
    """
    
    ### Import modules
    import numpy as np
    
    ### Remove observational data
    databias = data - data_obs[np.newaxis,np.newaxis,:,:,:]

    return databias

###############################################################################

def calculate_anomalies(data,data_obs,lats,lons,baseline,yearsall):
    """
    Calculates anomalies for each model and observational data set. Note that
    it assumes the years at the moment
    """
    
    ### Import modules
    import numpy as np
    
    ### Select years to slice
    minyr = baseline.min()
    maxyr = baseline.max()
    yearq = np.where((yearsall >= minyr) & (yearsall <= maxyr))[0]
    
    if data.ndim == 5:
        
        ### Slice years
        modelnew = data[:,:,yearq,:,:]
        obsnew = data_obs[yearq,:,:]
        
        ### Average climatology
        meanmodel = np.nanmean(modelnew[:,:,:,:,:],axis=2)
        meanobs = np.nanmean(obsnew,axis=0)
        
        ### Calculate anomalies
        modelanom = data[:,:,:,:,:] - meanmodel[:,:,np.newaxis,:,:]
        obsanom = data_obs[:,:,:] - meanobs[:,:]

    return modelanom,obsanom

###############################################################################

def remove_ensemble_mean(data,ravel_modelens,ravelmodeltime,rm_standard_dev,numOfEns):
    """
    Removes ensemble mean
    """
    
    ### Import modulates
    import numpy as np
    
    ### Remove ensemble mean
    if data.ndim == 4:
        datameangoneq = data - np.nanmean(data,axis=0)
    elif data.ndim == 5:
        ensmeanmodel = np.nanmean(data,axis=1)
        datameangoneq = np.empty((data.shape))
        for i in range(data.shape[0]):
            datameangoneq[i,:,:,:,:] = data[i,:,:,:,:] - ensmeanmodel[i,:,:,:]
            print('Completed: Ensemble mean removed for model %s!' % (i+1))
    
    if ravel_modelens == True:
        datameangone = np.reshape(datameangoneq,(datameangoneq.shape[0]*datameangoneq.shape[1],
                                                 datameangoneq.shape[2],
                                                 datameangoneq.shape[3],
                                                 datameangoneq.shape[4]))
    else: 
        datameangone = datameangoneq
    if rm_standard_dev == False:
        if ravelmodeltime == True:
            datameangone = np.reshape(datameangoneq,(datameangoneq.shape[0]*datameangoneq.shape[1]*datameangoneq.shape[2],
                                                      datameangoneq.shape[3],
                                                      datameangoneq.shape[4]))
        else: 
            datameangone = datameangoneq
    
    return datameangone

###############################################################################

def remove_ocean(data,data_obs,lat_bounds,lon_bounds):
    """
    Masks out the ocean for land_only == True
    """
    
    ### Import modules
    import numpy as np
    from netCDF4 import Dataset
    import calc_dataFunctions as df
    
    ### Read in land mask
    directorydata = '/Users/zlabe/Data/masks/'
    filename = 'lsmask_19x25.nc'
    datafile = Dataset(directorydata + filename)
    maskq = datafile.variables['nmask'][:]
    lats = datafile.variables['latitude'][:]
    lons = datafile.variables['longitude'][:]
    datafile.close()
    
    mask,lats,lons = df.getRegion(maskq,lats,lons,lat_bounds,lon_bounds)
    
    ### Mask out model and observations
    datamask = data * mask
    data_obsmask = data_obs * mask
    
    return datamask, data_obsmask

###############################################################################

def remove_land(data,data_obs,lat_bounds,lon_bounds):
    """
    Masks out the ocean for ocean_only == True
    """
    
    ### Import modules
    import numpy as np
    from netCDF4 import Dataset
    import calc_dataFunctions as df
    
    ### Read in ocean mask
    directorydata = '/Users/zlabe/Data/masks/'
    filename = 'ocmask_19x25.nc'
    datafile = Dataset(directorydata + filename)
    maskq = datafile.variables['nmask'][:]
    lats = datafile.variables['latitude'][:]
    lons = datafile.variables['longitude'][:]
    datafile.close()
    
    mask,lats,lons = df.getRegion(maskq,lats,lons,lat_bounds,lon_bounds)
    
    ### Mask out model and observations
    datamask = data * mask
    data_obsmask = data_obs * mask
    
    return datamask, data_obsmask

###############################################################################

def standardize_data(Xtrain,Xtest):
    """
    Standardizes training and testing data
    """
    
    ### Import modulates
    import numpy as np

    Xmean = np.nanmean(Xtrain,axis=0)
    Xstd = np.nanstd(Xtrain,axis=0)
    Xtest = (Xtest - Xmean)/Xstd
    Xtrain = (Xtrain - Xmean)/Xstd
    
    stdVals = (Xmean,Xstd)
    stdVals = stdVals[:]
    
    return Xtrain,Xtest,stdVals

###############################################################################
    
def rm_standard_dev(var,window,ravelmodeltime,numOfEns):
    """
    Smoothed standard deviation
    """
    import pandas as pd
    import numpy as np
    
    print('\n\n-----------STARTED: Rolling std!\n\n')
    
    
    if var.ndim == 3:
        rollingstd = np.empty((var.shape))
        for i in range(var.shape[1]):
            for j in range(var.shape[2]):
                series = pd.Series(var[:,i,j])
                rollingstd[:,i,j] = series.rolling(window).std().to_numpy()
    elif var.ndim == 4:
        rollingstd = np.empty((var.shape))
        for ens in range(var.shape[0]):
            for i in range(var.shape[2]):
                for j in range(var.shape[3]):
                    series = pd.Series(var[ens,:,i,j])
                    rollingstd[ens,:,i,j] = series.rolling(window).std().to_numpy()
    elif var.ndim == 5:
        varn = np.reshape(var,(var.shape[0]*var.shape[1],var.shape[2],var.shape[3],var.shape[4]))
        rollingstd = np.empty((varn.shape))
        for ens in range(varn.shape[0]):
            for i in range(varn.shape[2]):
                for j in range(varn.shape[3]):
                    series = pd.Series(varn[ens,:,i,j])
                    rollingstd[ens,:,i,j] = series.rolling(window).std().to_numpy()
    
    newdataq = rollingstd[:,window:,:,:] 
    
    if ravelmodeltime == True:
        newdata = np.reshape(newdataq,(newdataq.shape[0]*newdataq.shape[1],
                                       newdataq.shape[2],newdataq.shape[3]))
    else:
        newdata = np.reshape(newdataq,(newdataq.shape[0]//numOfEns,numOfEns,newdataq.shape[1],
                                       newdataq.shape[2],newdataq.shape[3]))
    print('-----------COMPLETED: Rolling std!\n\n')     
    return newdata 

###############################################################################
    
def rm_variance_dev(var,window,ravelmodeltime):
    """
    Smoothed variance
    """
    import pandas as pd
    import numpy as np
    
    print('\n\n-----------STARTED: Rolling vari!\n\n')
    
    rollingvar = np.empty((var.shape))
    for ens in range(var.shape[0]):
        for i in range(var.shape[2]):
            for j in range(var.shape[3]):
                series = pd.Series(var[ens,:,i,j])
                rollingvar[ens,:,i,j] = series.rolling(window).var().to_numpy()
    
    newdataq = rollingvar[:,window:,:,:] 
    
    if ravelmodeltime == True:
        newdata = np.reshape(newdataq,(newdataq.shape[0]*newdataq.shape[1],
                                       newdataq.shape[2],newdataq.shape[3]))
    else:
        newdata = newdataq
    print('-----------COMPLETED: Rolling vari!\n\n')     
    return newdata 

###############################################################################
###############################################################################
###############################################################################