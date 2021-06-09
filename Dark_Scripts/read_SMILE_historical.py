"""
Function(s) reads in monthly data from the MMLEA for selected
variables over the historical period
 
Notes
-----
    Author : Zachary Labe
    Date   : 11 February 2021
    
Usage
-----
    [1] read_SMILEhistorical(directory,simulation,vari,sliceperiod,sliceshape,
                             slicenan,numOfEns)
    [2] readAllSmileDataHist(directory,simulation,vari,sliceperiod,sliceshape,
                             slicenan,numOfEns,ravelbinary,lensalso,randomalso,
                             ravelyearsbinary,shuffletype)
"""

def read_SMILEhistorical(directory,simulation,vari,sliceperiod,sliceshape,slicenan,numOfEns):
    """
    Function reads monthly data from the MMLEA
    
    Parameters
    ----------
    directory : string
        path for data
    simulation : string
        name of the model
    vari : string
        variable for analysis
    sliceperiod : string
        how to average time component of data
    sliceyear : string
        how to slice number of years for data
    sliceshape : string
        shape of output array
    slicenan : string or float
        Set missing values
    numOfEns : number of ensembles
        integer
    shuffletype : string
        how to generate random numbers
        
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
    read_SMILEhistorical(directory,simulation,vari,sliceperiod,sliceshape,
                         slicenan,numOfEns)
    """
    print('\n>>>>>>>>>> STARTING read_SMILEhistorical function!')
    
    ### Import modules
    import numpy as np
    from netCDF4 import Dataset
    import calc_Utilities as UT
    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter(action='ignore', category=RuntimeWarning)
    
    ###########################################################################
    ### Parameters
    if simulation=='CCCma_canesm2':
        time = np.arange(1950,2100+1,1)
        timeslice = '%s-%s' % (time.min(),time.max())
        mon = 12
        ens = np.arange(1,50+1,1)
    elif simulation == 'CSIRO_MK3.6':
        time = np.arange(1850,2100+1,1)
        timeslice = '%s-%s' % (time.min(),time.max())
        mon = 12
        ens = np.arange(1,30+1,1)
    elif simulation == 'GFDL_CM3':
        time = np.arange(1920,2100+1,1)
        timeslice = '%s-%s' % (time.min(),time.max())
        mon = 12
        ens = np.arange(1,20+1,1)
    elif simulation == 'GFDL_ESM2M':
        time = np.arange(1950,2100+1,1)
        timeslice = '%s-%s' % (time.min(),time.max())
        mon = 12
        ens = np.arange(1,30+1,1)
    elif simulation == 'KNMI_ecearth':
        time = np.arange(1860,2100+1,1)
        timeslice = '%s-%s' % (time.min(),time.max())
        mon = 12
        ens = np.arange(1,16+1,1)
    elif simulation == 'MPI':
        time = np.arange(1850,2099+1,1)
        timeslice = '%s-%s' % (time.min(),time.max())
        mon = 12
        ens = np.arange(1,100+1,1)
    else:
        ValueError('WRONG SMILE SELECTED!')
    
    ###########################################################################
    ### Read in data
    numOfEnslist = np.arange(1,numOfEns+1,1)
    ens = numOfEnslist
        
    membersvar = []
    for i,ensmember in enumerate(numOfEnslist):
        filename = directory + '%s/monthly/%s_%s_%s.nc' % (simulation,vari,ensmember,timeslice)                                                          
        data = Dataset(filename,'r')
        lat1 = data.variables['latitude'][:]
        lon1 = data.variables['longitude'][:]
        var = data.variables['%s' % vari][:,:,:]
        data.close()
        
        print('Completed: read ensemble --%s for %s for %s--' % (simulation,ensmember,vari))
        membersvar.append(var)
        del var
    membersvar = np.asarray(membersvar)
    ensvalue = np.reshape(membersvar,(len(ens),time.shape[0],mon,
                                    lat1.shape[0],lon1.shape[0]))
    del membersvar
    print('Completed: read all members!\n')
    
    ###########################################################################
    ### Check for missing data
    ensvalue[np.where(ensvalue  <= -999)] = np.nan
        
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
    elif slicenan == False:
        ensshape = ensshape
    else:
        ensshape[np.where(np.isnan(ensshape))] = slicenan
        
    ###########################################################################
    ### Change units
    if vari == 'SLP':
        if simulation != 'GFDL_CM3' and simulation != 'GFDL_ESM2M':
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
    yearhistq = np.where((time >= 1950) & (time <= 2019))[0]
    print(time[yearhistq])
    histmodel = ensshape[:,yearhistq,:,:]

    print('Shape of output FINAL = ', histmodel.shape,[[histmodel.ndim]])
    print('>>>>>>>>>> ENDING read_SMILEhistorical function!')
    return lat1,lon1,histmodel

def readAllSmileDataHist(directory,simulation,vari,sliceperiod,sliceshape,slicenan,numOfEns,ravelbinary,lensalso,randomalso,ravelyearsbinary,shuffletype):
    """
    Function reads in all models from the SMILE archive
    
    Parameters
    ----------
    directory : string
        path for data
    simulation : list
        models to loop through and save
    vari : string
        variable for analysis
    sliceperiod : string
        how to average time component of data
    sliceyear : string
        how to slice number of years for data
    sliceshape : string
        shape of output array
    slicenan : string or float
        Set missing values
    numOfEns : number of ensembles
        integer
    rivalbinary : whether to ravel the models together or not
        binary
    lensalso : whether to include lens model
        binary
    randomalso : whether to include a rnadom numbers model
        binary
    ravelyearsbinary : whether to ravel years and ens/models together
        binary
        
    Returns
    -------
    lat : 1d numpy array
        latitudes
    lon : 1d numpy array
        longitudes
    var : numpy array
        processed models
        
    Usage
    -----
    readAllSmileDataHist(directory,simulation,vari,sliceperiod,sliceshape,
                        slicenan,numOfEns,ravelbinary,lensalso,randomalso,
                        ravelyearsbinary)
    """
    print('\n>>>>>>>>>> STARTING readAllSmileDataHist function!')
    
    ### Import modules
    import numpy as np
    import warnings
    import read_LENS_historical as LLL
    import read_randomData_monthly as RAN
    
    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter(action='ignore', category=RuntimeWarning)    
    
    ###########################################################################
    ### Read in smile archive
    yearshistorical = np.arange(1950,2019+1,1)
    allmodelstestq = np.empty((len(simulation),numOfEns,len(yearshistorical),96,144))
    for i in range(len(simulation)):
        lat,lon,allmodelstestq[i,:,:,:,:] = read_SMILEhistorical(directory,
                                                                simulation[i],vari,
                                                                sliceperiod,sliceshape,
                                                                slicenan,numOfEns)
  
    ###########################################################################
    ### Add LENS to the data      
    if lensalso == True:
        directorylens = '/Users/zlabe/Data/LENS/monthly/'
        lat,lon,lens = LLL.read_LENShistorical(directorylens,vari,sliceperiod,
                                           sliceshape,slicenan,numOfEns)
        
        allmodelstest = np.append(allmodelstestq,lens[np.newaxis,:,:,:,:],axis=0)
        print('Completed: added LENS')
    else:
        allmodelstest = allmodelstestq
 
    ###########################################################################
    ### Add random numbers to the data     
    if randomalso == True:
        directorydataRA = '/Users/zlabe/Data/'
        latr,lonr,datarand,ENSmeanr = RAN.read_randomData_monthly(directorydataRA,
                                                              vari,sliceperiod,
                                                              sliceshape,
                                                              slicenan,
                                                              numOfEns,
                                                              True,shuffletype,'historical')
        
        allmodelstestadd = np.append(allmodelstest,datarand[np.newaxis,:,:,:,:],axis=0)
        print('Completed: added RANDOM-%s' % shuffletype)
    else:
        allmodelstestadd = allmodelstest
     
    ###########################################################################
    ### Reshape the models and ensembles together
    if ravelbinary == True:
        comb = np.reshape(allmodelstestadd,(allmodelstestadd.shape[0]*allmodelstestadd.shape[1],
                                         allmodelstestadd.shape[2],allmodelstestadd.shape[3],
                                         allmodelstestadd.shape[4]))
        print('Completed: combined models and ensembles')
    else:
        comb = allmodelstestadd
        
    ###########################################################################
    ### Reshape the models and ensembles together
    if ravelyearsbinary == True:
        combyr = np.reshape(comb,(comb.shape[0]*comb.shape[1],comb.shape[2],comb.shape[3]))
        print('Completed: combined models/ensembles and years')
    else:
        combyr = comb

    print('Shape of output = ', combyr.shape,[[combyr.ndim]])
    print('>>>>>>>>>> ENDING readAllSmileDataHist function!')       
    return lat,lon,combyr


### Test functions - do not use!
# import numpy as np
# import matplotlib.pyplot as plt
# import calc_Utilities as UT
# modelGCMs = ['CCCma_canesm2','MPI','CSIRO_MK3.6','KNMI_ecearth',
#               'GFDL_CM3','GFDL_ESM2M']
# directory = '/Users/zlabe/Data/SMILE/'
# simulation = modelGCMs[4]
# vari = 'T2M'
# sliceperiod = 'annual'
# sliceshape = 4
# slicenan = 'nan'
# numOfEns = 16
# lensalso = True
# randomalso = True
# ravelyearsbinary = False
# ravelbinary = False
# shuffletype = 'RANDGAUSS'
# lat,lon,var = read_SMILEhistorical(directory,simulation,vari,
#                                             sliceperiod,sliceshape,
#                                             slicenan,numOfEns)
# lat,lon,comb = readAllSmileDataHist(directory,modelGCMs,
#                                     vari,sliceperiod,sliceshape,
#                                     slicenan,numOfEns,ravelbinary,
#                                     lensalso,randomalso,ravelyearsbinary,shuffletype)
# lon2,lat2 = np.meshgrid(lon,lat)
# ave = UT.calc_weightedAve(comb,lat2)
    
# mean = np.empty((comb.shape[0],comb.shape[1],comb.shape[2]))
# for i in range(ave.shape[0]):
#     mean[i,:,:] = ave[i] - np.nanmean(ave[i],axis=0)