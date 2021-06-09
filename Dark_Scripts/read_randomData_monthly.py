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
                               sliceshape,slicenan,numOfEns,ensYes,
                               shuffletype,timeper)

"""

def read_randomData_monthly(directorydata,variq,sliceperiod,
                            sliceshape,slicenan,numOfEns,ensYes,
                            shuffletype,timeper):
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
    shuffletype : string
        type of shuffled numbers
    timeper : string
        future or historical
        
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
    import read_LENS_historical as LL
    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter(action='ignore', category=RuntimeWarning)
    
    ###########################################################################
    ### Parameters
    if timeper == 'historical':
        time = np.arange(1950,2019+1,1)
    elif timeper == 'future':
        time = np.arange(2020,2099+1,1)
    else:
        print(ValueError('WRONG TIME PERIOD SELECTED FOR RANDOM DATA!'))
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
        if shuffletype == 'RANDGAUSS':
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
            print('<<< %s >>>' % shuffletype)
                
        elif shuffletype == 'TIMEENS': # shuffle ensembles with years
            if sliceperiod != 'none':
                directorylens = '/Users/zlabe/Data/LENS/monthly/'
                lat,lon,datall = LL.read_LENShistorical(directorylens,variq,sliceperiod,
                                                        sliceshape,slicenan,numOfEns)
                ensshape = np.empty((datall.shape))
                for iil in range(datall.shape[2]):
                    for jjl in range(datall.shape[3]):
                        temp = datall[:,:,iil,jjl].ravel()
                        np.random.shuffle(temp)
                        tempq = np.reshape(temp,(datall.shape[0],datall.shape[1]))
                        ensshape[:,:,iil,jjl] = tempq
                print('<<< %s >>>' % shuffletype)
        elif shuffletype == 'ALLENSRAND': # shuffle the entire 4d array
            if sliceperiod != 'none':
                directorylens = '/Users/zlabe/Data/LENS/monthly/'
                lat,lon,datall = LL.read_LENShistorical(directorylens,variq,sliceperiod,
                                                        sliceshape,slicenan,numOfEns)
                temp = datall.ravel()
                np.random.shuffle(temp)
                ensshape = np.reshape(temp,(datall.shape[0],datall.shape[1],
                                            datall.shape[2],datall.shape[3]))
                print('<<< %s >>>' % shuffletype)
        elif shuffletype == 'ALLENSRANDrmmean': # rm ensemble mean, then shuffle
            if sliceperiod != 'none':
                directorylens = '/Users/zlabe/Data/LENS/monthly/'
                lat,lon,datall = LL.read_LENShistorical(directorylens,variq,sliceperiod,
                                                        sliceshape,slicenan,numOfEns)
                datallm = datall - np.nanmean(datall[:,:,:,:],axis=0)
                temp = datallm.ravel()
                np.random.shuffle(temp)
                ensshape = np.reshape(temp,(datall.shape[0],datall.shape[1],
                                            datall.shape[2],datall.shape[3]))
                print('<<< %s >>>' % shuffletype)
    elif ensYes == False:
        if sliceperiod == 'none':
            ensshape = np.random.randn(time.shape[0],mon,lat1.shape[0],lon1.shape[0])
            print('Shape of output =', ensshape.shape, [[ensshape.ndim]])
            print('Completed: ALL MONTHS-OBS!')
        else:
            ensshape = np.random.randn(time.shape[0],lat1.shape[0],lon1.shape[0])
            print('Shape of output =', ensshape.shape, [[ensshape.ndim]])
            print('Completed: ANNUAL MEAN-OBS!')
            print('<<< %s >>>' % shuffletype)
       
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
        
    print('>>>>>>>>>> ENDING read_randomData_monthly function!\n')    
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
# ensYes = True
# shuffletype = 'ALLENSRANDrmmean'
# lat1,lon1,data,ENSmean = read_randomData_monthly(directorydataRA,variq,monthlychoice,
#                                                   sliceshapeRA,slicenanRA,numOfEnsRA,ensYes,
#                                                   shuffletype,timeper)
# lon2,lat2 = np.meshgrid(lon1,lat1)
# ave = UT.calc_weightedAve(data,lat2).transpose()