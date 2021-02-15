"""
Function(s) reads in monthly data that is generated from random noise in the 
shape of CESM-LENS
 
Notes
-----
    Author : Zachary Labe
    Date   : 19 October 2020
    
Usage
-----
    [1] read_randomData_monthly(directorydataRA,variq,
                                monthlychoice,slicebaseRA,
                                sliceshapeRA,addclimoRA,
                                slicenanRA,takeEnsMeanRA)
"""

def read_randomData_monthly(directory,vari,sliceperiod,slicebase,sliceshape,addclimo,slicenan,takeEnsMean):
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
    time = np.arange(1920,2080+1,1)
    mon = 12
    # ens1 = np.arange(1,35+1,1)
    # ens2 = np.arange(101,105+1,1)
    # allens = np.append(ens1,ens2)
    allens = np.arange(1,20+1,1) # EDIT TO COMPARE WITH X(LENS)
    ens = list(map('{:03d}'.format, allens))
    
    ###########################################################################
    ### Create data
    data = Dataset('/Users/zlabe/Data/LENS/monthly/T2M/T2M_001_1920-2100.nc','r')
    lat1 = data.variables['latitude'][:]
    lon1 = data.variables['longitude'][:]
    data.close()
    
    ensvar = np.random.randn(len(ens),time.shape[0],mon,lat1.shape[0],lon1.shape[0])

    print('Completed: read all members!\n')
    
    ###########################################################################
    ### Calculate anomalies or not
    if addclimo == True:
        ensvalue = ensvar
        print('Completed: calculated absolute variable!')
    elif addclimo == False:
        yearsq = np.where((time >= slicebase.min()) & (time <= slicebase.max()))[0]
        yearssel = time[yearsq]
        
        mean = np.nanmean(ensvar[:,yearsq,:,:,:])
        ensvalue = ensvar - mean
        print('Completed: calculated anomalies from',
              slicebase.min(),'to',slicebase.max())
        
    ###########################################################################
    ### Slice over months (currently = [ens,yr,mn,lat,lon])
    ### Shape of output array
    if sliceperiod == 'annual':
        ensvalue = np.nanmean(ensvalue,axis=2)
        if sliceshape == 1:
            ensshape = ensvalue.ravel()
        elif sliceshape == 4:
            ensshape = ensvalue
        print('Shape of output = ', ensshape.shape,[[ensshape.ndim]])
        print('Completed: ANNUAL MEAN!')
    elif sliceperiod == 'DJF':
        ensshape = np.empty((ensvalue.shape[0],ensvalue.shape[1]-1,
                             lat1.shape[0],lon1.shape[0]))
        for i in range(ensvalue.shape[0]):                    
            ensshape[i,:,:,:] = UT.calcDecJanFeb(ensvalue[i,:,:,:,:],
                                                 lat1,lon1,'surface',1)
        print('Shape of output = ', ensshape.shape,[[ensshape.ndim]])
        print('Completed: DJF MEAN!')
    elif sliceperiod == 'MAM':
        enstime = np.nanmean(ensvalue[:,:,2:5,:,:],axis=2)
        if sliceshape == 1:
            ensshape = enstime.ravel()
        elif sliceshape == 4:
            ensshape = enstime
        print('Shape of output = ', ensshape.shape,[[ensshape.ndim]])
        print('Completed: MAM MEAN!')
    elif sliceperiod == 'JJA':
        enstime = np.nanmean(ensvalue[:,:,5:8,:,:],axis=2)
        if sliceshape == 1:
            ensshape = enstime.ravel()
        elif sliceshape == 4:
            ensshape = enstime
        print('Shape of output = ', ensshape.shape,[[ensshape.ndim]])
        print('Completed: JJA MEAN!')
    elif sliceperiod == 'SON':
        enstime = np.nanmean(ensvalue[:,:,8:11,:,:],axis=2)
        if sliceshape == 1:
            ensshape = enstime.ravel()
        elif sliceshape == 4:
            ensshape = enstime
        print('Shape of output = ', ensshape.shape,[[ensshape.ndim]])
        print('Completed: SON MEAN!')
    elif sliceperiod == 'JFM':
        enstime = np.nanmean(ensvalue[:,:,0:3,:,:],axis=2)
        if sliceshape == 1:
            ensshape = enstime.ravel()
        elif sliceshape == 4:
            ensshape = enstime
        print('Shape of output = ', ensshape.shape,[[ensshape.ndim]])
        print('Completed: JFM MEAN!')
    elif sliceperiod == 'AMJ':
        enstime = np.nanmean(ensvalue[:,:,3:6,:,:],axis=2)
        if sliceshape == 1:
            ensshape = enstime.ravel()
        elif sliceshape == 4:
            ensshape = enstime
        print('Shape of output = ', ensshape.shape,[[ensshape.ndim]])
        print('Completed: AMJ MEAN!')
    elif sliceperiod == 'JAS':
        enstime = np.nanmean(ensvalue[:,:,6:9,:,:],axis=2)
        if sliceshape == 1:
            ensshape = enstime.ravel()
        elif sliceshape == 4:
            ensshape = enstime
        print('Shape of output = ', ensshape.shape,[[ensshape.ndim]])
        print('Completed: JAS MEAN!')
    elif sliceperiod == 'OND':
        enstime = np.nanmean(ensvalue[:,:,9:,:,:],axis=2)
        if sliceshape == 1:
            ensshape = enstime.ravel()
        elif sliceshape == 4:
            ensshape = enstime
        print('Shape of output = ', ensshape.shape,[[ensshape.ndim]])
        print('Completed: OND MEAN!')
    elif sliceperiod == 'none':
        if sliceshape == 1:
            ensshape = ensvalue.ravel()
        elif sliceshape == 3:
            ensshape= np.reshape(ensvalue,(ensvalue.shape[0]*ensvalue.shape[1],
                                             ensvalue.shape[2],ensvalue.shape[3]))
        elif sliceshape == 5:
            ensshape = ensvalue
        print('Shape of output =', ensshape.shape, [[ensshape.ndim]])
        print('Completed: ALL MONTHS!')
        
    ###########################################################################
    ### Change missing values
    if slicenan == 'nan':
        ensshape[np.where(np.isnan(ensshape))] = np.nan
        print('Completed: missing values are =',slicenan)
    else:
        ensshape[np.where(np.isnan(ensshape))] = slicenan

    ###########################################################################
    ### Take ensemble mean
    if takeEnsMean == True:
        ENSmean = np.nanmean(ensshape,axis=0)
        print('Ensemble mean AVAILABLE!')
    elif takeEnsMean == False:
        ENSmean = np.nan
        print('Ensemble mean NOT available!')
    else:
        ValueError('WRONG OPTION!')
        
    print('>>>>>>>>>> ENDING read_randomData_monthly function!')
        
    return lat1,lon1,ensshape,ENSmean
        

### Test functions - do not use!
# import numpy as np
# import matplotlib.pyplot as plt
# import calc_Utilities as UT
# directorydataRA = '/Users/zlabe/Data/'
# variq = 'T2M'
# monthlychoice = 'annual'
# slicebaseRA = np.arange(1951,1980+1,1)
# sliceshapeRA = 4
# slicenanRA = 'nan'
# addclimoRA = True
# takeEnsMeanRA = False
# lat1,lon1,data,ENSmean = read_randomData_monthly(directorydataRA,variq,
#                                        monthlychoice,slicebaseRA,
#                                        sliceshapeRA,addclimoRA,
#                                        slicenanRA,takeEnsMeanRA)