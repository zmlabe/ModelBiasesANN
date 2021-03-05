"""
Function(s) reads in monthly data that is generated from random noise in the 
shape of climate models and observations
 
Notes
-----
    Author : Zachary Labe
    Date   : 5 March 2021
    
Usage
-----
    [1] read_randomData_monthly(directorydata,variq,sliceperiod,
                               sliceshape,slicenan,numOfEns,ensYes)

"""

def read_randomData_monthly(directorydata,variq,sliceperiod,
                            sliceshape,slicenan,numOfEns,ensYes):
    """
    Function generates RANDOM DATA
    
    Parameters
    ----------
    directory : string
        path for data
    vari : string
        variable for analysis
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
    takeEnsMean : binary
        whether to take ensemble mean
        
    Returns
    -------
    lat : 1d numpy array
        latitudes
    lon : 1d numpy array
        longitudes
    var : numpy array
        processed variable
    ENSmean : numpy array
        ensemble mean
        
    Usage
    -----
    read_randomData_monthly(directorydataRA,variq,
                                monthlychoice,slicebaseRA,
                                sliceshapeRA,addclimoRA,
                                slicenanRA,takeEnsMeanRA)
    """
    print('\n>>>>>>>>>> STARTING read_randomData_monthly function!')
    
    ### Import modules
    import numpy as np
    from netCDF4 import Dataset
    import warnings
    import calc_Utilities as UT
    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter(action='ignore', category=RuntimeWarning)
    
    ###########################################################################
    ### Parameters
    time = np.arange(1950,2019+1,1)
    mon = 12
    allens = np.arange(1,numOfEns+1,1) 
    ens = list(map('{:03d}'.format, allens))
    
    ###########################################################################
    ### Create data
    data = Dataset('/Users/zlabe/Data/LENS/monthly/T2M/T2M_001_1920-2100.nc','r')
    lat1 = data.variables['latitude'][:]
    lon1 = data.variables['longitude'][:]
    data.close()

    print('Completed: read all members!\n')
        
    ###########################################################################
    ### Slice over months (currently = [ens,yr,mn,lat,lon])
    ### Shape of output array
    if ensYes == True:
        if sliceperiod == 'annual':
            ensshape = np.random.randn(len(ens),time.shape[0],lat1.shape[0],lon1.shape[0])
            print('Shape of output = ', ensshape.shape,[[ensshape.ndim]])
            print('Completed: ANNUAL MEAN!')
        elif sliceperiod == 'DJF':
            ensshape = np.random.randn(len(ens),time.shape[0],lat1.shape[0],lon1.shape[0])
            print('Shape of output = ', ensshape.shape,[[ensshape.ndim]])
            print('Completed: DJF MEAN!')
        elif sliceperiod == 'MAM':
            ensshape = np.random.randn(len(ens),time.shape[0],lat1.shape[0],lon1.shape[0])
            print('Shape of output = ', ensshape.shape,[[ensshape.ndim]])
            print('Completed: MAM MEAN!')
        elif sliceperiod == 'JJA':
            ensshape = np.random.randn(len(ens),time.shape[0],lat1.shape[0],lon1.shape[0])
            print('Shape of output = ', ensshape.shape,[[ensshape.ndim]])
            print('Completed: JJA MEAN!')
        elif sliceperiod == 'SON':
            ensshape = np.random.randn(len(ens),time.shape[0],lat1.shape[0],lon1.shape[0])
            print('Shape of output = ', ensshape.shape,[[ensshape.ndim]])
            print('Completed: SON MEAN!')
        elif sliceperiod == 'JFM':
            ensshape = np.random.randn(len(ens),time.shape[0],lat1.shape[0],lon1.shape[0])
            print('Shape of output = ', ensshape.shape,[[ensshape.ndim]])
            print('Completed: JFM MEAN!')
        elif sliceperiod == 'AMJ':
            ensshape = np.random.randn(len(ens),time.shape[0],lat1.shape[0],lon1.shape[0])
            print('Shape of output = ', ensshape.shape,[[ensshape.ndim]])
            print('Completed: AMJ MEAN!')
        elif sliceperiod == 'JAS':
            ensshape = np.random.randn(len(ens),time.shape[0],lat1.shape[0],lon1.shape[0])
            print('Shape of output = ', ensshape.shape,[[ensshape.ndim]])
            print('Completed: JAS MEAN!')
        elif sliceperiod == 'OND':
            ensshape = np.random.randn(len(ens),time.shape[0],lat1.shape[0],lon1.shape[0])
            print('Shape of output = ', ensshape.shape,[[ensshape.ndim]])
            print('Completed: OND MEAN!')
        elif sliceperiod == 'none':
            ensshape = np.random.randn(len(ens),time.shape[0],mon,lat1.shape[0],lon1.shape[0])
            print('Shape of output =', ensshape.shape, [[ensshape.ndim]])
            print('Completed: ALL MONTHS!')
    elif ensYes == False:
        if sliceperiod == 'none':
            ensshape = np.random.randn(time.shape[0],mon,lat1.shape[0],lon1.shape[0])
            print('Shape of output =', ensshape.shape, [[ensshape.ndim]])
            print('Completed: ALL MONTHS-OBS!')
        else:
            ensshape = np.random.randn(time.shape[0],lat1.shape[0],lon1.shape[0])
            print('Shape of output =', ensshape.shape, [[ensshape.ndim]])
            print('Completed: ANNUAL MEAN-OBS!')
        
    ###########################################################################
    ### Change missing values
    if slicenan == 'nan':
        ensshape[np.where(np.isnan(ensshape))] = np.nan
        print('Completed: missing values are =',slicenan)
    else:
        ensshape[np.where(np.isnan(ensshape))] = slicenan

    ###########################################################################
    ENSmean = np.nan
    print('Ensemble mean NOT available!')
        
    print('>>>>>>>>>> ENDING read_randomData_monthly function!')    
    return lat1,lon1,ensshape,ENSmean
        

# ### Test functions - do not use!
# import numpy as np
# import matplotlib.pyplot as plt
# import calc_Utilities as UT
# directorydataRA = '/Users/zlabe/Data/'
# variq = 'T2M'
# monthlychoice = 'annual'
# sliceshapeRA = 4
# slicenanRA = 'nan'
# numOfEnsRA = 16
# ensYes = False
# lat1,lon1,data,ENSmean = read_randomData_monthly(directorydataRA,variq,monthlychoice,
#                                                  sliceshapeRA,slicenanRA,numOfEnsRA,ensYes)