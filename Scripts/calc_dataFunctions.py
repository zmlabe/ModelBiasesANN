"""
Functions are useful untilities for data processing in the NN
 
Notes
-----
    Author : Zachary Labe
    Date   : 16 February 2021
    
Usage
-----
    [1] readFiles(variq,dataset,monthlychoice,numOfEns,lensalso,ravelyearsbinary,ravelbinary)
    [2] getRegion(data,lat1,lon1,lat_bounds,lon_bounds)
"""

def readFiles(variq,dataset,monthlychoice,numOfEns,lensalso,ravelyearsbinary,ravelbinary):
    """
    Function reads in data for selected dataset

    Parameters
    ----------
    variq : string
        variable for analysis
    dataset : string
        name of data set for primary data
    monthlychoice : string
        time period of analysis
    numOfEns : integer
        number of ensembles to include
    lensalso : whether to include lens model
        binary
    ravelyearsbinary : whether to ravel years and ens/models together
        binary
    rivalbinary : whether to ravel the models together or not
        binary
        
    Returns
    -------
    data : numpy array
        data from selected data set
    lat1 : 1d numpy array
        latitudes
    lon1 : 1d numpy array
        longitudes

    Usage
    -----
    data,lat1,lon1 = readFiles(variq,dataset)
    """
    print('\n>>>>>>>>>> Using readFiles function!')
    
    ### Import modules
    import numpy as np
    
    if dataset == 'best':
        import read_BEST as BB
        directorydataBB = '/Users/zlabe/Data/BEST/'
        sliceyearBB = np.arange(1956,2019+1,1)
        sliceshapeBB = 3
        slicenanBB = 'nan'
        addclimoBB = True
        ENSmean = np.nan
        lat1,lon1,data = BB.read_BEST(directorydataBB,monthlychoice,
                                      sliceyearBB,sliceshapeBB,addclimoBB,
                                      slicenanBB)
    elif dataset == 'ERA5':
        import read_ERA5_monthly as ER
        directorydataER = '/Users/zlabe/Data/ERA5/'
        sliceyearER = np.arange(1979,2019+1,1)
        sliceshapeER = 3
        slicenanER = 'nan'
        addclimoER = True
        ENSmean = np.nan
        lat1,lon1,data = ER.read_ERA5_monthly(variq,directorydataER,
                                              monthlychoice,sliceyearER,
                                              sliceshapeER,addclimoER,
                                              slicenanER)
    elif dataset == '20CRv3':
        import read_20CRv3_monthly as TW
        directorydataTW = '/Users/zlabe/Data/20CRv3/'
        sliceyearTW = np.arange(1836,2015+1,1)
        sliceshapeTW = 3
        slicenanTW = 'nan'
        addclimoTW = True
        ENSmean = np.nan
        lat1,lon1,data = TW.read_20CRv3_monthly(variq,directorydataTW,
                                              monthlychoice,sliceyearTW,
                                              sliceshapeTW,addclimoTW,
                                              slicenanTW)
    elif dataset == 'RANDOM':
        import read_randomData_monthly as RA
        directorydataRA = '/Users/zlabe/Data/'
        slicebaseRA = np.arange(1951,1980+1,1)
        sliceshapeRA = 4
        slicenanRA = 'nan'
        addclimoRA = True
        takeEnsMeanRA = False
        lat1,lon1,data,ENSmean = RA.read_randomData_monthly(directorydataRA,variq,
                                               monthlychoice,slicebaseRA,
                                               sliceshapeRA,addclimoRA,
                                               slicenanRA,takeEnsMeanRA)
    elif dataset == 'SMILE':
        import read_SMILE_historical as SM
        directorydataSM = '/Users/zlabe/Data/SMILE/'
        modelGCMsSM = ['CCCma_canesm2','MPI','CSIRO_MK3.6','KNMI_ecearth',
                      'GFDL_CM3','GFDL_ESM2M']
        sliceperiodSM = 'annual'
        sliceshapeSM = 4
        slicenanSM = 'nan'
        numOfEnsSM = 16
        ENSmean = np.nan
        lat1,lon1,data = SM.readAllSmileDataHist(directorydataSM,modelGCMsSM,
                                                         variq,sliceperiodSM,sliceshapeSM,
                                                         slicenanSM,numOfEnsSM,ravelbinary,
                                                         lensalso,ravelyearsbinary)    
    elif any([dataset=='XGHG',dataset=='XAER',
              dataset=='XBMB',dataset=='XLULC']):
        import read_SINGLE_LENS as SI
        directorySI = '/Users/zlabe/Data/LENS/SINGLEFORCING/'
        simulationSI = dataset
        slicebaseSI = np.arange(1951,1980+1,1)
        sliceshapeSI = 4
        slicenanSI = 'nan'
        addclimoSI = True
        takeEnsMeanSI = False
        lat1,lon1,data,ENSmean = SI.read_SINGLE_LENS(directorySI,simulationSI,variq,monthlychoice,
                                                slicebaseSI,sliceshapeSI,addclimoSI,
                                                slicenanSI,takeEnsMeanSI)
    else:
        ValueError('WRONG DATA SET SELECTED!')
        
    print('>>>>>>>>>> Completed: Finished readFiles function!')
    return data,lat1,lon1  

def getRegion(data,lat1,lon1,lat_bounds,lon_bounds):
    """
    Function masks out region for data set

    Parameters
    ----------
    data : 3d+ numpy array
        original data set
    lat1 : 1d array
        latitudes
    lon1 : 1d array
        longitudes
    lat_bounds : 2 floats
        (latmin,latmax)
    lon_bounds : 2 floats
        (lonmin,lonmax)
        
    Returns
    -------
    data : numpy array
        MASKED data from selected data set
    lat1 : 1d numpy array
        latitudes
    lon1 : 1d numpy array
        longitudes

    Usage
    -----
    data,lats,lons = getRegion(data,lat1,lon1,lat_bounds,lon_bounds)
    """
    print('\n>>>>>>>>>> Using get_region function!')
    
    ### Import modules
    import numpy as np
    
    ### Note there is an issue with 90N latitude (fixed!)
    lat1 = np.round(lat1,3)
    
    ### Mask latitudes
    if data.ndim == 2:
        latq = np.where((lat1 >= lat_bounds[0]) & (lat1 <= lat_bounds[1]))[0]
        latn = lat1[latq]
        datalatq = data[latq,:] 
        ### Mask longitudes
        lonq = np.where((lon1 >= lon_bounds[0]) & (lon1 <= lon_bounds[1]))[0]
        lonn = lon1[lonq]
        datalonq = datalatq[:,lonq]
        
    elif data.ndim == 3:
        latq = np.where((lat1 >= lat_bounds[0]) & (lat1 <= lat_bounds[1]))[0]
        latn = lat1[latq]
        datalatq = data[:,latq,:] 
        ### Mask longitudes
        lonq = np.where((lon1 >= lon_bounds[0]) & (lon1 <= lon_bounds[1]))[0]
        lonn = lon1[lonq]
        datalonq = datalatq[:,:,lonq]
        
    elif data.ndim == 4:
        latq = np.where((lat1 >= lat_bounds[0]) & (lat1 <= lat_bounds[1]))[0]
        latn = lat1[latq]
        datalatq = data[:,:,latq,:]        
        ### Mask longitudes
        lonq = np.where((lon1 >= lon_bounds[0]) & (lon1 <= lon_bounds[1]))[0]
        lonn = lon1[lonq]
        datalonq = datalatq[:,:,:,lonq]
        
    elif data.ndim == 5:
        latq = np.where((lat1 >= lat_bounds[0]) & (lat1 <= lat_bounds[1]))[0]
        latn = lat1[latq]
        datalatq = data[:,:,:,latq,:]
        ### Mask longitudes
        lonq = np.where((lon1 >= lon_bounds[0]) & (lon1 <= lon_bounds[1]))[0]
        lonn = lon1[lonq]
        datalonq = datalatq[:,:,:,:,lonq]
    
    ### New variable name
    datanew = datalonq
    
    print('>>>>>>>>>> Completed: getRegion function!')
    return datanew,latn,lonn   

### Test functions - do not use!
# import numpy as np
# import matplotlib.pyplot as plt
# import calc_Utilities as UT
# numOfEns = 16
# lensalso = True
# ravelyearsbinary = False
# ravelbinary = False
# data,lat1,lon1 = readFiles('T2M','SMILE','annual',numOfEns,lensalso,ravelyearsbinary,ravelbinary)