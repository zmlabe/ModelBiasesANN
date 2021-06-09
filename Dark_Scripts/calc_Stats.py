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
    [13] addNoiseTwinSingle(data,integer,sizeOfTwin,random_segment_seed,maskNoiseClass,lat_bounds,lon_bounds)
    [14] smoothedEnsembles(data,lat_bounds,lon_bounds)
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
    else:
        obsnew = data_obs[yearq,:,:]
        
        ### Average climatology
        meanobs = np.nanmean(obsnew,axis=0)
        
        ### Calculate anomalies
        obsanom = data_obs[:,:,:] - meanobs[:,:]
        modelanom = np.nan
        print('NO MODEL ANOMALIES DUE TO SHAPE SIZE!!!')

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
    
    ### Check for floats
    datamask[np.where(datamask==0.)] = 0
    data_obsmask[np.where(data_obsmask==0.)] = 0
    
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
    
    ### Check for floats
    datamask[np.where(datamask==0.)] = 0
    data_obsmask[np.where(data_obsmask==0.)] = 0
    
    return datamask, data_obsmask

###############################################################################

def standardize_data(Xtrain,Xtest):
    """
    Standardizes training and testing data
    """
    
    ### Import modulates
    import numpy as np

    Xmean = np.mean(Xtrain,axis=0)
    Xstd = np.std(Xtrain,axis=0)
    Xtest = (Xtest - Xmean)/Xstd
    Xtrain = (Xtrain - Xmean)/Xstd
    
    stdVals = (Xmean,Xstd)
    stdVals = stdVals[:]
    
    ### If there is a nan (like for land/ocean masks)
    if np.isnan(np.min(Xtrain)) == True:
        Xtrain[np.isnan(Xtrain)] = 0
        Xtest[np.isnan(Xtest)] = 0
        print('--THERE WAS A NAN IN THE STANDARDIZED DATA!--')
    
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

def addNoiseTwinSingle(data,data_obs,integer,sizeOfTwin,random_segment_seed,maskNoiseClass,lat_bounds,lon_bounds):
    """
    Calculate an additional class of noise added to the original data
    """
    import numpy as np
    import sys
    print('\n----------- USING EXPERIMENT CLASS #%s -----------' % sizeOfTwin)
    
    if sizeOfTwin == 1: 
        """
        Adds random noise to each grid point
        """
        newmodels = data.copy()
        if newmodels.shape[0] > 7:
            newmodels = newmodels[:7,:,:,:,:]
        
        dataRandNoise = np.random.randint(low=-integer,high=integer+1,size=newmodels.shape) 
        # dataRandNoise = np.random.uniform(low=-integer,high=integer,size=newmodels.shape) 
        randomNoiseTwinq = newmodels + dataRandNoise
        randomNoiseTwin = randomNoiseTwinq.reshape(randomNoiseTwinq.shape[0]*randomNoiseTwinq.shape[1],
                                                  randomNoiseTwinq.shape[2],randomNoiseTwinq.shape[3],
                                                  randomNoiseTwinq.shape[4])
        print('--Size of noise twin --->',randomNoiseTwin.shape)
        print('<<Added noise of +-%s at every grid point for twin!>>' % integer)
        
        ### Calculating random subsample
        if random_segment_seed == None:
            random_segment_seed = int(int(np.random.randint(1, 100000)))
        np.random.seed(random_segment_seed)
        
        nrows = randomNoiseTwin.shape[0]
        nens = randomNoiseTwinq.shape[1]
    
        ### Picking out random ensembles
        i = 0
        newIndices = list()
        while i < nens:
            line = np.random.randint(0, nrows)
            if line not in newIndices:
                newIndices.append(line)
                i += 1
            else:
                pass
        print('<<Subsampling noise on %s model-ensembles>>' % newIndices)
        
        ### Subsample the noisy data
        noiseModel = randomNoiseTwin[newIndices,:,:,:]
        noiseModelClass = noiseModel[np.newaxis,:,:,:,:]
        
        ### Mask land or ocean if necessary
        if maskNoiseClass != 'none':
            if maskNoiseClass == 'land':
                emptyobs = np.full((noiseModelClass.shape[2],noiseModelClass.shape[3],noiseModelClass.shape[4]),np.nan)
                noiseModelClass,wrong_obs = remove_ocean(noiseModelClass,emptyobs,lat_bounds,lon_bounds) 
                print('\n*Removed land data - OCEAN TWIN*')
            elif maskNoiseClass == 'ocean':
                emptyobs = np.full((noiseModelClass.shape[2],noiseModelClass.shape[3],noiseModelClass.shape[4]),np.nan)
                noiseModelClass,wrong_obs = remove_land(noiseModelClass,emptyobs,lat_bounds,lon_bounds)                 
                print('\n*Removed land data - NOISE TWIN*')  
            else:
                print(ValueError('SOMETHING IS WRONG WITH MASKING NOISE TWIN!'))
                sys.exit()
        
        ### Make new class of noisy twin subsample
        dataclass = np.append(data,noiseModelClass,axis=0)
        
    elif sizeOfTwin == 2: 
        """
        Adds multimodel bias to each model
        """
        newmodels = data.copy()
        multimodelmean = np.nanmean(newmodels,axis=0) # model mean      
        uniquemodelbias = newmodels - multimodelmean # difference
        
        aveensbias = np.nanmean(uniquemodelbias,axis=1) # ensemble mean
        avebias = np.nanmean(aveensbias,axis=1) # time mean

        randomNoiseTwinq = newmodels + avebias[:,np.newaxis,np.newaxis,:,:] 
        randomNoiseTwin = randomNoiseTwinq.reshape(randomNoiseTwinq.shape[0]*randomNoiseTwinq.shape[1],
                                                  randomNoiseTwinq.shape[2],randomNoiseTwinq.shape[3],
                                                  randomNoiseTwinq.shape[4])
        print('--Size of noise twin --->',randomNoiseTwin.shape)
        print('<<Added noise of multimodel bias>>')
        
        ### Calculating random subsample
        if random_segment_seed == None:
            random_segment_seed = int(int(np.random.randint(1, 100000)))
        np.random.seed(random_segment_seed)
        
        nrows = randomNoiseTwin.shape[0]
        nens = randomNoiseTwinq.shape[1]
    
        ### Picking out random ensembles
        i = 0
        newIndices = list()
        while i < nens:
            line = np.random.randint(0, nrows)
            if line not in newIndices:
                newIndices.append(line)
                i += 1
            else:
                pass
        print('<<Subsampling noise on %s model-ensembels>>' % newIndices)
        
        ### Subsample the noisy data
        noiseModel = randomNoiseTwin[newIndices,:,:,:]
        noiseModelClass = noiseModel[np.newaxis,:,:,:,:]
        
        ### Mask land or ocean if necessary
        if maskNoiseClass != 'none':
            if maskNoiseClass == 'land':
                emptyobs = np.full((noiseModelClass.shape[2],noiseModelClass.shape[3],noiseModelClass.shape[4]),np.nan)
                noiseModelClass,wrong_obs = remove_ocean(noiseModelClass,emptyobs,lat_bounds,lon_bounds) 
                print('\n*Removed land data - OCEAN TWIN*')
            elif maskNoiseClass == 'ocean':
                emptyobs = np.full((noiseModelClass.shape[2],noiseModelClass.shape[3],noiseModelClass.shape[4]),np.nan)
                noiseModelClass,wrong_obs = remove_land(noiseModelClass,emptyobs,lat_bounds,lon_bounds)                 
                print('\n*Removed land data - NOISE TWIN*')  
            else:
                print(ValueError('SOMETHING IS WRONG WITH MASKING NOISE TWIN!'))
                sys.exit()
        
        ### Make new class of noisy twin subsample
        dataclass = np.append(data,noiseModelClass,axis=0)
        
    elif sizeOfTwin == 3: 
        """
        Adds bias from observations to each model
        """        
        newmodels = data.copy()
        diffobs = newmodels - data_obs[np.newaxis,np.newaxis,:,:,:]
        ensmean = np.nanmean(diffobs,axis=1) # ensemble mean
        avebias = np.nanmean(ensmean,axis=1) # years mean

        randomNoiseTwinq = newmodels + avebias[:,np.newaxis,np.newaxis,:,:] 
        randomNoiseTwin = randomNoiseTwinq.reshape(randomNoiseTwinq.shape[0]*randomNoiseTwinq.shape[1],
                                                  randomNoiseTwinq.shape[2],randomNoiseTwinq.shape[3],
                                                  randomNoiseTwinq.shape[4])
        print('--Size of noise twin --->',randomNoiseTwin.shape)
        print('<<Added noise of observational bias>>')
        
        ### Calculating random subsample
        if random_segment_seed == None:
            random_segment_seed = int(int(np.random.randint(1, 100000)))
        np.random.seed(random_segment_seed)
        
        nrows = randomNoiseTwin.shape[0]
        nens = randomNoiseTwinq.shape[1]
    
        ### Picking out random ensembles
        i = 0
        newIndices = list()
        while i < nens:
            line = np.random.randint(0, nrows)
            if line not in newIndices:
                newIndices.append(line)
                i += 1
            else:
                pass
        print('<<Subsampling noise on %s model-ensembels>>' % newIndices)
        
        ### Subsample the noisy data
        noiseModel = randomNoiseTwin[newIndices,:,:,:]
        noiseModelClass = noiseModel[np.newaxis,:,:,:,:]
        
        ### Mask land or ocean if necessary
        if maskNoiseClass != 'none':
            if maskNoiseClass == 'land':
                emptyobs = np.full((noiseModelClass.shape[2],noiseModelClass.shape[3],noiseModelClass.shape[4]),np.nan)
                noiseModelClass,wrong_obs = remove_ocean(noiseModelClass,emptyobs,lat_bounds,lon_bounds) 
                print('\n*Removed land data - OCEAN TWIN*')
            elif maskNoiseClass == 'ocean':
                emptyobs = np.full((noiseModelClass.shape[2],noiseModelClass.shape[3],noiseModelClass.shape[4]),np.nan)
                noiseModelClass,wrong_obs = remove_land(noiseModelClass,emptyobs,lat_bounds,lon_bounds)                 
                print('\n*Removed land data - NOISE TWIN*')  
            else:
                print(ValueError('SOMETHING IS WRONG WITH MASKING NOISE TWIN!'))
                sys.exit()
        
        ### Make new class of noisy twin subsample
        dataclass = np.append(data,noiseModelClass,axis=0)
        
    elif sizeOfTwin == 4: 
        """
        Adds multimodel mean class
        """        
        newmodels = data.copy()
        multimodelmean = np.nanmean(newmodels,axis=0)
        randomNoiseTwin = multimodelmean
        print('--Size of noise twin --->',randomNoiseTwin.shape)
        print('<<Added noise of multimodel mean class>>')
        
        ### Add new class
        noiseModelClass = randomNoiseTwin[np.newaxis,:,:,:,:]
        
        ### Mask land or ocean if necessary
        if maskNoiseClass != 'none':
            if maskNoiseClass == 'land':
                emptyobs = np.full((noiseModelClass.shape[2],noiseModelClass.shape[3],noiseModelClass.shape[4]),np.nan)
                noiseModelClass,wrong_obs = remove_ocean(noiseModelClass,emptyobs,lat_bounds,lon_bounds) 
                print('\n*Removed land data - OCEAN TWIN*')
            elif maskNoiseClass == 'ocean':
                emptyobs = np.full((noiseModelClass.shape[2],noiseModelClass.shape[3],noiseModelClass.shape[4]),np.nan)
                noiseModelClass,wrong_obs = remove_land(noiseModelClass,emptyobs,lat_bounds,lon_bounds)                 
                print('\n*Removed land data - NOISE TWIN*')  
            else:
                print(ValueError('SOMETHING IS WRONG WITH MASKING NOISE TWIN!'))
                sys.exit()
        
        ### Make new class of noisy twin subsample
        dataclass = np.append(data,noiseModelClass,axis=0)
        
    elif sizeOfTwin == 5:
        """ 
        Adds random noise numbers
        """
        integer = 35
        randomNoiseTwin = np.random.uniform(low=-integer,high=integer,size=(data.shape[1],data.shape[2],data.shape[3],data.shape[4]))
        
        ### Add new class
        noiseModelClass = randomNoiseTwin[np.newaxis,:,:,:,:]
        print('--Size of noise twin --->',randomNoiseTwin.shape)
        print('<<Added noise of random numbers class for +-%s>>' % integer)
        
        ### Mask land or ocean if necessary
        if maskNoiseClass != 'none':
            if maskNoiseClass == 'land':
                emptyobs = np.full((noiseModelClass.shape[2],noiseModelClass.shape[3],noiseModelClass.shape[4]),np.nan)
                noiseModelClass,wrong_obs = remove_ocean(noiseModelClass,emptyobs,lat_bounds,lon_bounds) 
                print('\n*Removed land data - OCEAN TWIN*')
            elif maskNoiseClass == 'ocean':
                emptyobs = np.full((noiseModelClass.shape[2],noiseModelClass.shape[3],noiseModelClass.shape[4]),np.nan)
                noiseModelClass,wrong_obs = remove_land(noiseModelClass,emptyobs,lat_bounds,lon_bounds)                 
                print('\n*Removed land data - NOISE TWIN*')  
            else:
                print(ValueError('SOMETHING IS WRONG WITH MASKING NOISE TWIN!'))
                sys.exit()
        
        ### Make new class of noisy twin subsample
        dataclass = np.append(data,noiseModelClass,axis=0)
        
    elif sizeOfTwin == 6:
        """ 
        Smoothes data
        """
        newmodels = data.copy()
        testens = newmodels[6,:,:,:,:] # 6 for LENS

        newmodeltest = np.empty(testens.shape)
        for sh in range(testens.shape[0]):
            ensnum = np.arange(data.shape[1])
            slices = np.random.choice(ensnum,size=data.shape[0],replace=False)
            slicenewmodel = np.nanmean(testens[slices,:,:,:],axis=0)
            newmodeltest[sh,:,:,:] = slicenewmodel
        
        ### Add new class
        noiseModelClass = newmodeltest[np.newaxis,:,:,:,:]
        print('--Size of noise twin --->',newmodeltest.shape)
        print('<<Added noise of shuffled model 7x for 16 ensembles>>')
        
        ### Mask land or ocean if necessary
        if maskNoiseClass != 'none':
            if maskNoiseClass == 'land':
                emptyobs = np.full((noiseModelClass.shape[2],noiseModelClass.shape[3],noiseModelClass.shape[4]),np.nan)
                noiseModelClass,wrong_obs = remove_ocean(noiseModelClass,emptyobs,lat_bounds,lon_bounds) 
                print('\n*Removed land data - OCEAN TWIN*')
            elif maskNoiseClass == 'ocean':
                emptyobs = np.full((noiseModelClass.shape[2],noiseModelClass.shape[3],noiseModelClass.shape[4]),np.nan)
                noiseModelClass,wrong_obs = remove_land(noiseModelClass,emptyobs,lat_bounds,lon_bounds)                 
                print('\n*Removed land data - NOISE TWIN*')  
            else:
                print(ValueError('SOMETHING IS WRONG WITH MASKING NOISE TWIN!'))
                sys.exit()
        
        ### Make new class of noisy twin subsample
        dataclass = np.append(data,noiseModelClass,axis=0)
        
    else:
        print(ValueError('Double check experiment for random class!'))
        sys.exit()
    
    print('--NEW Size of noise class--->',dataclass.shape)
    print('----------- ENDING EXPERIMENT CLASS #%s -----------' % sizeOfTwin)
    return dataclass

###############################################################################

def smoothedEnsembles(data,lat_bounds,lon_bounds):
    """ 
    Smoothes all ensembles by taking subsamples
    """
    ### Import modules
    import numpy as np
    import sys
    print('\n------- Beginning of smoothing the ensembles per model -------')
       
    ### Save MM
    newmodels = data.copy()
    mmean = newmodels[-1,:,:,:,:] # 7 for MMmean
    otherens = newmodels[:7,:,:,:,:]

    newmodeltest = np.empty(otherens.shape)
    for modi in range(otherens.shape[0]):
        for sh in range(otherens.shape[1]):
            ensnum = np.arange(otherens.shape[1])
            slices = np.random.choice(ensnum,size=otherens.shape[0],replace=False)
            modelsmooth = otherens[modi]
            slicenewmodel = np.nanmean(modelsmooth[slices,:,:,:],axis=0)
            newmodeltest[modi,sh,:,:,:] = slicenewmodel
    
    ### Add new class
    smoothClass = np.append(newmodeltest,mmean[np.newaxis,:,:,:],axis=0)
    print('--Size of smooth twin --->',newmodeltest.shape)
    
    print('--NEW Size of smoothedclass--->',smoothClass.shape)
    print('------- Ending of smoothing the ensembles per model -------')
    return smoothClass

###############################################################################
###############################################################################
###############################################################################