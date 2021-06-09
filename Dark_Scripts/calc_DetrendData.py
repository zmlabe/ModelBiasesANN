"""
Functions remove the linear trend at each grid point for the period of 
1950-2019.
 
Notes
-----
    Author : Zachary Labe
    Date   : 15 April 2021
    
Usage
-----
    detrendData(datavar,timeperiod)
    detrendDataR(datavar,timeperiod)
"""

def detrendData(datavar,level,timeperiod):
    """
    Function removes linear trend

    Parameters
    ----------
    datavar : 5d numpy array or 6d numpy array 
        [model,ensemble,year,lat,lon] or [model,ensemble,year,level,lat,lon]
    level : string
        Height of variable (surface or profile)
    timeperiod : string
        daily or monthly
    
    Returns
    -------
    datavardt : 5d numpy array or 6d numpy array 
        [model,ensemble,year,lat,lon] or [model,ensemble,year,level,lat,lon]

    Usage
    -----
    datavardt = detrendData(datavar,level,timeperiod)
    """
    print('\n>>> Using detrendData function! \n')
    ###########################################################################
    ###########################################################################
    ###########################################################################
    ### Import modules
    import numpy as np
    import scipy.stats as sts
    
    ### Detrend data array
    if level == 'surface':
        x = np.arange(datavar.shape[2])
        
        slopes = np.empty((datavar.shape[0],datavar.shape[1],datavar.shape[3],
                          datavar.shape[4]))
        intercepts = np.empty((datavar.shape[0],datavar.shape[1],datavar.shape[3],
                      datavar.shape[4]))
        for ens in range(datavar.shape[0]):
            print('-- Detrended data for model -- #%s!' % (ens+1))
            for mo in range(datavar.shape[1]):
                for i in range(datavar.shape[3]):
                    for j in range(datavar.shape[4]):
                        mask = np.isfinite(datavar[ens,mo,:,i,j])
                        y = datavar[ens,mo,:,i,j]
                        
                        if np.sum(mask) == y.shape[0]:
                            xx = x
                            yy = y
                        else:
                            xx = x[mask]
                            yy = y[mask]
                        
                        if np.isfinite(np.nanmean(yy)):
                            slopes[ens,mo,i,j],intercepts[ens,mo,i,j], \
                            r_value,p_value,std_err = sts.linregress(xx,yy)
                        else:
                            slopes[ens,mo,i,j] = np.nan
                            intercepts[ens,mo,i,j] = np.nan
        print('Completed: Detrended data for each grid point!')
                            
        datavardt = np.empty(datavar.shape)
        for ens in range(datavar.shape[0]):
            for yr in range(datavar.shape[1]):
                for mo in range(datavar.shape[2]):
                    datavardt[ens,yr,mo,:,:] = datavar[ens,yr,mo,:,:] - \
                                        (slopes[ens,yr,:,:]*x[mo] + \
                                         intercepts[ens,yr,:,:])
                                
    elif level == 'profile':
        x = np.arange(datavar.shape[1])
        
        slopes = np.empty((datavar.shape[0],datavar.shape[2],datavar.shape[3],
                          datavar.shape[4],datavar.shape[5]))
        intercepts = np.empty((datavar.shape[0],datavar.shape[2],datavar.shape[3],
                      datavar.shape[4],datavar.shape[5]))
        for ens in range(datavar.shape[0]):
            print('-- Detrended data for ensemble member -- #%s!' % (ens+1))
            for mo in range(datavar.shape[2]):
                for le in range(datavar.shape[3]):
                    for i in range(datavar.shape[4]):
                        for j in range(datavar.shape[5]):
                            mask = np.isfinite(datavar[ens,:,mo,le,i,j])
                            y = datavar[ens,:,mo,le,i,j]
                            
                            if np.sum(mask) == y.shape[0]:
                                xx = x
                                yy = y
                            else:
                                xx = x[mask]
                                yy = y[mask]
                            
                            if np.isfinite(np.nanmean(yy)):
                                slopes[ens,mo,le,i,j],intercepts[ens,mo,le,i,j], \
                                r_value,p_value,std_err= sts.linregress(xx,yy)
                            else:
                                slopes[ens,mo,le,i,j] = np.nan
                                intercepts[ens,mo,le,i,j] = np.nan
        print('Completed: Detrended data for each grid point!')
                            
        datavardt = np.empty(datavar.shape)
        for yr in range(datavar.shape[1]):
            datavardt[:,yr,:,:,:,:] = datavar[:,yr,:,:,:,:] - \
                                    (slopes*x[yr] + intercepts)        
    else:
        print(ValueError('Selected wrong height - (surface or profile!)!')) 

    ### Save memory
    del datavar
    
    print('\n>>> Completed: Finished detrendData function!')
    return datavardt

###############################################################################

def detrendDataR(datavar,level,timeperiod):
    """
    Function removes linear trend from reanalysis data

    Parameters
    ----------
    datavar : 3d numpy array or 4d numpy array or 5d numpy array 
        [year,month,lat,lon] or [year,month,level,lat,lon]
    level : string
        Height of variable (surface or profile)
    timeperiod : string
        daily or monthly
    
    Returns
    -------
    datavardt : 4d numpy array or 5d numpy array 
        [year,month,lat,lon] or [year,month,level,lat,lon]
        
    Usage
    -----
    datavardt = detrendDataR(datavar,level,timeperiod)
    """
    print('\n>>> Using detrendData function! \n')
    ###########################################################################
    ###########################################################################
    ###########################################################################
    ### Import modules
    import numpy as np
    import sys
    import scipy.stats as sts
    
    ### Detrend data array
    if level == 'surface':
        if datavar.ndim == 3:
            x = np.arange(datavar.shape[0])
            
            slopes = np.empty((datavar.shape[1],datavar.shape[2]))
            intercepts = np.empty((datavar.shape[1],datavar.shape[2]))
            for i in range(datavar.shape[1]):
                for j in range(datavar.shape[2]):
                    mask = np.isfinite(datavar[:,i,j])
                    y = datavar[:,i,j]
                    
                    if np.sum(mask) == y.shape[0]:
                        xx = x
                        yy = y
                    else:
                        xx = x[mask]
                        yy = y[mask]
                    
                    if np.isfinite(np.nanmean(yy)):
                        slopes[i,j],intercepts[i,j], \
                        r_value,p_value,std_err = sts.linregress(xx,yy)
                    else:
                        slopes[i,j] = np.nan
                        intercepts[i,j] = np.nan
            print('Completed: Detrended data for each grid point!')
                                
            datavardt = np.empty(datavar.shape)
            for yr in range(datavar.shape[0]):
                datavardt[yr,:,:] = datavar[yr,:,:] - (slopes*x[yr] + intercepts)
                
        elif datavar.ndim == 4:
            x = np.arange(datavar.shape[0])
            
            slopes = np.empty((datavar.shape[1],datavar.shape[2],datavar.shape[3]))
            intercepts = np.empty((datavar.shape[1],datavar.shape[2],
                                   datavar.shape[3]))
            for mo in range(datavar.shape[1]):
                print('Completed: detrended -- Month %s --!' % (mo+1))
                for i in range(datavar.shape[2]):
                    for j in range(datavar.shape[3]):
                        mask = np.isfinite(datavar[:,mo,i,j])
                        y = datavar[:,mo,i,j]
                        
                        if np.sum(mask) == y.shape[0]:
                            xx = x
                            yy = y
                        else:
                            xx = x[mask]
                            yy = y[mask]
                        
                        if np.isfinite(np.nanmean(yy)):
                            slopes[mo,i,j],intercepts[mo,i,j], \
                            r_value,p_value,std_err = sts.linregress(xx,yy)
                        else:
                            slopes[mo,i,j] = np.nan
                            intercepts[mo,i,j] = np.nan
            print('Completed: Detrended data for each grid point!')
                                
            datavardt = np.empty(datavar.shape)
            for yr in range(datavar.shape[0]):
                datavardt[yr,:,:,:] = datavar[yr,:,:,:] - (slopes*x[yr] + intercepts)
                
        else:
            print('ARRAY SHAPED WRONG!')
            sys.exit()
                                
    elif level == 'profile':
        x = np.arange(datavar.shape[0])
        
        slopes = np.empty((datavar.shape[1],datavar.shape[2],
                          datavar.shape[3],datavar.shape[4]))
        intercepts = np.empty((datavar.shape[1],datavar.shape[2],
                      datavar.shape[3],datavar.shape[4]))
        for mo in range(datavar.shape[1]):
            print('Completed: detrended -- Month %s --!' % (mo+1))
            for le in range(datavar.shape[2]):
                print('Completed: detrended Level %s!' % (le+1))
                for i in range(datavar.shape[3]):
                    for j in range(datavar.shape[4]):
                        mask = np.isfinite(datavar[:,mo,le,i,j])
                        y = datavar[:,mo,le,i,j]
                        
                        if np.sum(mask) == y.shape[0]:
                            xx = x
                            yy = y
                        else:
                            xx = x[mask]
                            yy = y[mask]
                        
                        if np.isfinite(np.nanmean(yy)):
                            slopes[mo,le,i,j],intercepts[mo,le,i,j], \
                            r_value,p_value,std_err= sts.linregress(xx,yy)
                        else:
                            slopes[mo,le,i,j] = np.nan
                            intercepts[mo,le,i,j] = np.nan
        print('Completed: Detrended data for each grid point!')
                            
        datavardt = np.empty(datavar.shape)
        for yr in range(datavar.shape[1]):
            datavardt[yr,:,:,:,:] = datavar[yr,:,:,:,:] - \
                                    (slopes*x[yr] + intercepts)        
    else:
        print(ValueError('Selected wrong height - (surface or profile!)!')) 

    ### Save memory
    del datavar
    
    print('\n>>> Completed: Finished detrendDataR function!')
    return datavardt