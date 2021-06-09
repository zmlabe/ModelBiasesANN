"""
Function(s) reads in monthly data from LENS for different variables using # of
ensemble members for a set future period

Notes
-----
    Author : Zachary Labe
    Date   : 14 April 2021

Usage
-----
    [1] read_LENSfuture(directory,vari,sliceperiod,sliceshape,
                             slicenan,numOfEns)
"""

def read_LENSfuture(directory,vari,sliceperiod,sliceshape,slicenan,numOfEns):
    """
    Function reads monthly data from LENS

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
    numOfEns : number of ensembles
        integer

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
    read_LENSfuture(directory,vari,sliceperiod,sliceshape,
                             slicenan,numOfEns)
    """
    print('\n>>>>>>>>>> STARTING read_LENSfuture function!')

    ### Import modules
    import numpy as np
    from netCDF4 import Dataset
    import calc_Utilities as UT
    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter(action='ignore', category=RuntimeWarning)

    ###########################################################################
    ### Parameters
    time = np.arange(1920,2100+1,1)
    mon = 12
    ens1 = np.arange(1,numOfEns+1,1)
    allens = ens1
    ens = list(map('{:03d}'.format, allens))

    ###########################################################################
    ### Read in data
    membersvar = []
    for i,ensmember in enumerate(ens):
        if vari == 'SLP':
            filename = directory + '%s/%s_%s_1920_2100.nc' % (vari,vari,
                                                              ensmember)
        elif vari == 'P':
            filename = directory + '%s/%s_%s_1920-2100.nc' % (vari,vari,
                                                              (i+1))
        else:
            filename = directory + '%s/%s_%s_1920-2100.nc' % (vari,vari,
                                                              ensmember)
        data = Dataset(filename,'r')
        lat1 = data.variables['latitude'][:]
        lon1 = data.variables['longitude'][:]
        var = data.variables['%s' % vari][:,:,:]
        data.close()

        print('Completed: read ensemble --%s--' % ensmember)
        membersvar.append(var)
        del var
    membersvar = np.asarray(membersvar)
    ensvalue = np.reshape(membersvar,(len(ens),time.shape[0],mon,
                                    lat1.shape[0],lon1.shape[0]))
    del membersvar
    print('Completed: read all members!\n')

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
        elif sliceshape == 4:
            ensshape= np.reshape(ensvalue,(ensvalue.shape[0],ensvalue.shape[1]*ensvalue.shape[2],
                                             ensvalue.shape[3],ensvalue.shape[4]))
        elif sliceshape == 5:
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
    elif vari == 'P':
        ensshape = ensshape * 86400 # kg/m2/s to mm/day
        ### "Average Monthly Rate of Precipitation"
        print('*** CURRENT UNITS ---> [[ mm/day ]]! ***')
    
    ###########################################################################
    ### Change years
    yearfutq = np.where((time >= 2020) & (time <= 2099))[0]
    print(time[yearfutq])
    futmodel = ensshape[:,yearfutq,:,:]

    print('Shape of output FINAL = ', futmodel.shape,[[futmodel.ndim]])
    print('>>>>>>>>>> ENDING read_LENSfuture function!')    
    return lat1,lon1,futmodel 

# ### Test functions - do not use!
# import numpy as np
# import matplotlib.pyplot as plt
# import calc_Utilities as UT
# directory = '/Users/zlabe/Data/LENS/monthly/'
# vari = 'T2M'
# sliceperiod = 'annual'
# sliceshape = 4
# slicenan = 'nan'
# numOfEns = 16
# ravelbinary = True
# lensalso = True
# lat,lon,var = read_LENSfuture(directory,vari,
#                                             sliceperiod,sliceshape,
#                                             slicenan,numOfEns)
# lon2,lat2 = np.meshgrid(lon,lat)
# ave = UT.calc_weightedAve(var,lat2)
