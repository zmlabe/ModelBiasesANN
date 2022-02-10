"""
Function reads in monthly data from NCEP2
 
Notes
-----
    Author : Zachary Labe
    Date   : 10 January 2022
    
Usage
-----
    [1] read_NCEP2(directory,sliceperiod,sliceyear,
                  sliceshape,addclimo,slicenan)
"""

def read_NCEP2(directory,sliceperiod,sliceyear,sliceshape,addclimo,slicenan):
    """
    Function reads monthly data from NCEP2
    
    Parameters
    ----------
    directory : string
        path for data
    sliceperiod : string
        how to average time component of data
    sliceyear : string
        how to slice number of years for data
    sliceshape : string
        shape of output array
    addclimo : binary
        True or false to add climatology
    slicenan : string or float
        Set missing values
        
    Returns
    -------
    lat : 1d numpy array
        latitudes
    lon : 1d numpy array
        longitudes
    var : 3d numpy array or 4d numpy array 
        [time,lat,lon] or [year,month,lat,lon]
        
    Usage
    -----
    lat,lon,var = read_NCEP2(directory,sliceperiod,sliceyear,
                            sliceshape,addclimo,slicenan)
    """
    print('\n>>>>>>>>>> STARTING read_NCEP2 function!')
    
    ### Import modules
    import numpy as np
    from netCDF4 import Dataset
    import warnings
    import calc_Utilities as UT
    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter(action='ignore', category=RuntimeWarning)
    
    ###########################################################################
    ### Parameters
    time = np.arange(1979,2021+1,1)
    monthslice = sliceyear.shape[0]*12
    mon = 12
    
    ###########################################################################
    ### Read in data
    filename = 'T2M_NCEP2_1979-2021.nc'
    data = Dataset(directory + filename,'r')
    lat1 = data.variables['latitude'][:]
    lon1 = data.variables['longitude'][:]
    anom = data.variables['T2M'][:,:,:]
    data.close()
    
    print('Years of output =',sliceyear.min(),'to',sliceyear.max())
    ###########################################################################
    ### Reshape data into [year,month,lat,lon]
    datamon = np.reshape(anom,(anom.shape[0]//mon,mon,
                               lat1.shape[0],lon1.shape[0]))
    
    ###########################################################################
    ### Return absolute temperature (1981-2010 baseline)
    if addclimo == True:
        filename = 'CLIM_NCEP2_1979-2021.n'
        datac = Dataset(directory + filename,'r')
        clim = datac['CLIM'][:,:,:]
        datac.close()
        
        ### Add [anomaly+climatology]
        tempmon = datamon + clim
        print('Completed: calculated absolute temperature!')
    else:
        tempmon = datamon
        print('Completed: calculated anomalies!')
    
    ###########################################################################
    ### Slice over months (currently = [yr,mn,lat,lon])
    ### Shape of output array
    if sliceperiod == 'annual':
        temptime = np.nanmean(tempmon,axis=1)
        if sliceshape == 1:
            tempshape = temptime.ravel()
        elif sliceshape == 3:
            tempshape = temptime
        print('Shape of output = ', tempshape.shape,[[tempshape.ndim]])
        print('Completed: ANNUAL MEAN!')
        print('Completed: ANNUAL MEAN!')
    elif sliceperiod == 'DJF':
        tempshape = UT.calcDecJanFeb(tempmon,lat1,lon1,'surface',1)
        print('Shape of output = ', tempshape.shape,[[tempshape.ndim]])
        print('Completed: DJF MEAN!')
    elif sliceperiod == 'JJA':
        temptime = np.nanmean(tempmon[:,5:8,:,:],axis=1)
        if sliceshape == 1:
            tempshape = temptime.ravel()
        elif sliceshape == 3:
            tempshape = temptime
        print('Shape of output = ', tempshape.shape,[[tempshape.ndim]])
        print('Completed: JJA MEAN!')
    elif sliceperiod == 'none':
        temptime = tempmon
        if sliceshape == 1:
            tempshape = tempshape.ravel()
        elif sliceshape == 3:
            tempshape = np.reshape(temptime,(temptime.shape[0]*temptime.shape[1],
                                             temptime.shape[2],temptime.shape[3]))
        elif sliceshape == 4:
            tempshape = tempmon
        print('Shape of output =', tempshape.shape, [[tempshape.ndim]])
        print('Completed: ALL MONTHS!')
        
    ###########################################################################
    ### Change missing values
    if slicenan == 'nan':
        tempshape[np.where(np.isnan(tempshape))] = np.nan
        print('Completed: missing values are =',slicenan)
    else:
        tempshape[np.where(np.isnan(tempshape))] = slicenan
        
    ###########################################################################
    ### Change units
    tempshape = tempshape - 273.15 # K to C
    print('Completed: Changed units (K to C)!')
       
    print('>>>>>>>>>> ENDING read_NCEP2 function!')
    return lat1,lon1,tempshape

### Test functions - do not use!
# import numpy as np
# import matplotlib.pyplot as plt
# directory = '/Users/zlabe/Data/NCEP2/'
# sliceperiod = 'DJF'
# sliceyear = np.arange(1979,2021+1,1)
# sliceshape = 3
# slicenan = 'nan'
# addclimo = True
# lat,lon,var = read_NCEP2(directory,sliceperiod,sliceyear,sliceshape,addclimo,slicenan)