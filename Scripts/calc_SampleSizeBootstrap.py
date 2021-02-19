"""
Functions are for statistics on random data

Notes
-----
    Author     : Zachary M. Labe
    Date       : 19 February 2021
    Version    : 1

Usage
-----
    [1] randomAccuracy(Ndata,Nrandom,noOfClasses,statistic)
"""

def randomAccuracy(Ndata,Nrandom,noOfClasses,statistic):
    """
    Function calculates statistics on the accuracy of random 
    chance for classification problems in ANNs

    Parameters
    ----------
    Ndata : integer
        Length of each sample size
    Nrandom : integer
        Number of random subsamples to calculate
    noOfClasses : integer
        Number of classes for ANN
    statistic : string
        Statistic to calculate on random distribution
        
    Returns
    -------
    randStat : 1d array
        Statistic based on accuracy of random distribution
    
    Usage
    -----
    randStat = randomAccuracy(Ndata,Nrandom,noOfClasses,statistic)
    """
    print('\n>>> Using randomAccuracy() function!')
    
    ### Import packages
    import numpy as np
    
    ### Make parameters integers
    Ndata = int(Ndata)
    Nrandom = int(Nrandom)
    noOfClasses = int(noOfClasses)
    
    ### Calculate random groups of data that are Ndata size
    array = np.empty((Nrandom,Ndata))
    for i in range(Nrandom):
        array[i,:] = np.random.randint(noOfClasses,size=Ndata)
    print(np.unique(array),'----> Double Check - Number of Classes')
        
    ### Calculate mean to generate distribution
    meanRand = np.nanmean(array,axis=1)
    
    if statistic == '90perc':
        randStat = np.percentile(meanRand,90)
    elif statistic == '95perc':
        randStat = np.percentile(meanRand,95)
    elif statistic == '99perc':
        randStat = np.percentile(meanRand,99)
    elif statistic == '50perc':
        randStat = np.percentile(meanRand,50)
     
    ### Following assumptions of balanced classes and random guess (1/k)
    if noOfClasses > 2:
        randStat = randStat/noOfClasses 
        print('Assumption of balanced classes')
    
    print('*Completed: Finished randomAccuracy() function!')
    return randStat

#### Testing functions - do no use!
# import numpy as np
# a = randomAccuracy(24,100000,2,'90perc')
# b = randomAccuracy(240,100000,2,'90perc')
# c = randomAccuracy(7*16*71*0.75,100000,7,'90perc')
# d = randomAccuracy(7*16*71*0.25,100000,7,'90perc')