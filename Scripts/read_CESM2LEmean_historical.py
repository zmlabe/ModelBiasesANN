"""
Function(s) reads in monthly data from the ensemble mean of CESM2-LE for 
the historical period

Notes
-----
    Author : Zachary Labe
    Date   : 21 June 2021

Usage
-----
    [1] readCESM2LEmean(directory,vari,sliceperiod,sliceshape,slicenan)
"""

def readCESM2LEmean(directory,vari,sliceperiod,sliceshape,slicenan):
    """
    Function reads monthly data from LENS2 mean

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

    Returns
    -------
    lat : 1d numpy array
        latitudes
    lon : 1d numpy array
        longitudes
    var : numpy array
        processed variable

    Usage
    -----
    readCESM2LEmean(directory,vari,sliceperiod,sliceshape,slicenan)
    """
    print('\n>>>>>>>>>> STARTING readCESM2LEmean function!')

    ### Import modules
    import numpy as np
    from netCDF4 import Dataset
    import calc_Utilities as UT
    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter(action='ignore', category=RuntimeWarning)

    ###########################################################################
    ### Parameters
    time = np.arange(1850,2100+1,1)
    mon = 12

    ###########################################################################
    ### Read in data
    filename = directory + '%s/%s_1850-2100_ensmean.nc' % (vari,vari)
    data = Dataset(filename,'r')
    lat1 = data.variables['latitude'][:]
    lon1 = data.variables['longitude'][:]
    var = data.variables['%s' % vari][:,:,:]
    data.close()

    print('Completed: read ensemble mean - CESM2-LE')

    membersvar = np.asarray(var)
    ensvalue = np.reshape(membersvar,(time.shape[0],mon,
                                    lat1.shape[0],lon1.shape[0]))

    ###########################################################################
    ### Slice over months (currently = [yr,mn,lat,lon])
    ### Shape of output array
    if sliceperiod == 'annual':
        ensvalue = np.nanmean(ensvalue,axis=1)
        if sliceshape == 1:
            ensshape = ensvalue.ravel()
        elif sliceshape == 4:
            ensshape = ensvalue
        print('Shape of output = ', ensshape.shape,[[ensshape.ndim]])
        print('Completed: ANNUAL MEAN!')
    elif sliceperiod == 'DJF':
        ensshape = UT.calcDecJanFeb(ensvalue,lat1,lon1,'surface',1)
        print('Shape of output = ', ensshape.shape,[[ensshape.ndim]])
        print('Completed: DJF MEAN!')
    elif sliceperiod == 'MAM':
        enstime = np.nanmean(ensvalue[:,2:5,:,:],axis=2)
        if sliceshape == 1:
            ensshape = enstime.ravel()
        elif sliceshape == 4:
            ensshape = enstime
        print('Shape of output = ', ensshape.shape,[[ensshape.ndim]])
        print('Completed: MAM MEAN!')
    elif sliceperiod == 'JJA':
        enstime = np.nanmean(ensvalue[:,5:8,:,:],axis=2)
        if sliceshape == 1:
            ensshape = enstime.ravel()
        elif sliceshape == 4:
            ensshape = enstime
        print('Shape of output = ', ensshape.shape,[[ensshape.ndim]])
        print('Completed: JJA MEAN!')
    elif sliceperiod == 'SON':
        enstime = np.nanmean(ensvalue[:,8:11,:,:],axis=2)
        if sliceshape == 1:
            ensshape = enstime.ravel()
        elif sliceshape == 4:
            ensshape = enstime
        print('Shape of output = ', ensshape.shape,[[ensshape.ndim]])
        print('Completed: SON MEAN!')
    elif sliceperiod == 'JFM':
        enstime = np.nanmean(ensvalue[:,0:3,:,:],axis=2)
        if sliceshape == 1:
            ensshape = enstime.ravel()
        elif sliceshape == 4:
            ensshape = enstime
        print('Shape of output = ', ensshape.shape,[[ensshape.ndim]])
        print('Completed: JFM MEAN!')
    elif sliceperiod == 'AMJ':
        enstime = np.nanmean(ensvalue[:,3:6,:,:],axis=2)
        if sliceshape == 1:
            ensshape = enstime.ravel()
        elif sliceshape == 4:
            ensshape = enstime
        print('Shape of output = ', ensshape.shape,[[ensshape.ndim]])
        print('Completed: AMJ MEAN!')
    elif sliceperiod == 'JAS':
        enstime = np.nanmean(ensvalue[:,6:9,:,:],axis=2)
        if sliceshape == 1:
            ensshape = enstime.ravel()
        elif sliceshape == 4:
            ensshape = enstime
        print('Shape of output = ', ensshape.shape,[[ensshape.ndim]])
        print('Completed: JAS MEAN!')
    elif sliceperiod == 'OND':
        enstime = np.nanmean(ensvalue[:,9:,:,:],axis=2)
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
                                             ensvalue.shape[3],ensvalue.shape[4]))
        elif sliceshape == 4:
            ensshape = ensvalue
        print('Shape of output =', ensshape.shape, [[ensshape.ndim]])
        print('Completed: ALL RAVELED MONTHS!')

    ###########################################################################
    ### Change missing values
    if slicenan == 'nan':
        ensshape[np.where(np.isnan(ensshape))] = np.nan
        print('Completed: missing values are =',slicenan)
    else:
        ensshape[np.where(np.isnan(ensshape))] = slicenan

    ###########################################################################
    ### Change units
    if vari == 'SLP':
        ensshape = ensshape/100 # Pa to hPa
        print('Completed: Changed units (Pa to hPa)!')
    elif vari == 'T2M':
        ensshape = ensshape - 273.15 # K to C
        print('Completed: Changed units (K to C)!')
    elif any([vari == 'PRECL',vari == 'PRECC',vari == 'PRECT']):
        ensshape = ensshape * 86400 # kg/m2/s to mm/day
        ### "Average Monthly Rate of Precipitation"
        print('*** CURRENT UNITS ---> [[ mm/day ]]! ***')
    
    ###########################################################################
    ### Change years
    yearhistq = np.where((time >= 1950) & (time <= 2019))[0]
    print(time[yearhistq])
    histmodel = ensshape[yearhistq,:,:]
    # histmodel = ensshape

    print('Shape of output FINAL = ', histmodel.shape,[[histmodel.ndim]])
    print('>>>>>>>>>> ENDING readCESM2LEmean function!')    
    return lat1,lon1,histmodel 

# ### Test functions - do not use!
# import numpy as np
# import matplotlib.pyplot as plt
# import calc_Utilities as UT
# directory = '/Users/zlabe/Data/CESM2-LE/monthly/'
# vari = 'T2M'
# sliceperiod = 'annual'
# sliceshape = 4
# slicenan = 'nan'
# ravelbinary = True
# lensalso = True
# lat,lon,var = readCESM2LEmean(directory,vari,sliceperiod,sliceshape,slicenan)
# lon2,lat2 = np.meshgrid(lon,lat)
# ave = UT.calc_weightedAve(var,lat2)

# from netCDF4 import Dataset
# data = Dataset('/Users/zlabe/Data/CESM2-LE/monthly/T2M/raw/TREFHT_1850-2100_ensmean.nc')
# var2 = data.variables['trefht'][:]
# lato2 = data.variables['lat'][:]
# lono2 = data.variables['lon'][:]
# data.close()
# lonss,latss = np.meshgrid(lono2,lato2)
# mean = np.nanmean(var2.reshape(251,12,181,360),axis=1)
# ave2 = UT.calc_weightedAve(mean,latss)-273.15

# yearsh = np.arange(1950,2019+1,1)
# years = np.arange(1850,2100+1,1)
# yearsold = np.arange(1920,2100+1,1)

# from netCDF4 import Dataset
# data = Dataset('/Users/zlabe/Data/LENS/monthly/T2M/T2M_ens-mean_1920-2100.nc')
# var3 = data.variables['T2M'][:]
# lato3 = data.variables['latitude'][:]
# lono3 = data.variables['longitude'][:]
# data.close()
# lonss3,latss3 = np.meshgrid(lono3,lato3)
# mean3 = np.nanmean(var3.reshape(181,12,96,144),axis=1)
# ave3 = UT.calc_weightedAve(mean3,latss3)-273.15
