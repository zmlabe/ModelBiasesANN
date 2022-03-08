"""
Script for plotting heat map of model rankings according to confidence and 
uses a threshold for minimum confidence

Author     : Zachary M. Labe
Date       : 16 December 2021
Version    : 7 - adds validation data for early stopping
"""

### Import packages
import sys
import matplotlib.pyplot as plt
import matplotlib.colors as c
import numpy as np
import palettable.cubehelix as cm
import palettable.scientific.sequential as sss
import palettable.cartocolors.qualitative as cc
import cmocean as cmocean
import cmasher as cmr
import calc_Utilities as UT
import scipy.stats as sts

### Plotting defaults 
plt.rc('text',usetex=True)
plt.rc('font',**{'family':'sans-serif','sans-serif':['Avant Garde']}) 

variablesall = ['T2M','P','SLP']
variablesall = ['T2M']
pickSMILEall = [[]] 
THRESH = 1/7
for va in range(len(variablesall)):
    for m in range(len(pickSMILEall)):
        ###############################################################################
        ###############################################################################
        ###############################################################################
        ### Data preliminaries 
        directorydata = '/Users/zlabe/Documents/Research/ModelComparison/Data/'
        directoryfigure = '/Users/zlabe/Desktop/ModelComparison_v1/v7/'
        letters = ["a","b","c","d","e","f","g","h","i","j","k","l","m","n"]
        ###############################################################################
        ###############################################################################
        modelGCMs = ['CanESM2','MPI','CSIRO-MK3.6','KNMI-ecearth',
                      'GFDL-CM3','GFDL-ESM2M','LENS']
        datasetsingle = ['SMILE']
        dataset_obs = 'ERA5BE'
        seasons = ['annual']
        variq = variablesall[va]
        reg_name = 'Arctic'
        if reg_name == 'SMILEGlobe':
            reg_nameq = 'GLOBAL'
        elif reg_name == 'LowerArctic':
            reg_nameq = 'LOWER ARCTIC'
        elif reg_name == 'Arctic':
            reg_nameq = 'ARCTIC'
        elif reg_name == 'narrowTropics':
            reg_nameq = 'TROPICS'
        elif reg_name == 'NH':
            reg_nameq = 'N. HEMISPHERE'
        elif reg_name == 'SH':
            reg_nameq = 'S. HEMISPHERE'
        elif reg_name == 'SouthernOcean':
            reg_nameq = 'SOUTHERN OCEAN'
        elif reg_name == 'GlobeNoPoles':
            reg_nameq = 'NO POLES'
        elif reg_name == 'Antarctic':
            reg_nameq = 'ANTARCTIC'
        timeper = 'historical'
        ###############################################################################
        ###############################################################################
        pickSMILE = pickSMILEall[m]
        if len(pickSMILE) >= 1:
            lenOfPicks = len(pickSMILE)
        else:
            lenOfPicks = len(modelGCMs)
        ###############################################################################
        ###############################################################################
        land_only = False
        ocean_only = False
        ###############################################################################
        ###############################################################################
        rm_merid_mean = False
        rm_annual_mean = False
        ###############################################################################
        ###############################################################################
        rm_ensemble_mean = False
        rm_observational_mean = False
        ###############################################################################
        ###############################################################################
        calculate_anomalies = False
        if calculate_anomalies == True:
            baseline = np.arange(1951,1980+1,1)
        ###############################################################################
        ###############################################################################
        window = 0
        ensTypeExperi = 'ENS'
        # shuffletype = 'TIMEENS'
        # shuffletype = 'ALLENSRAND'
        # shuffletype = 'ALLENSRANDrmmean'
        shuffletype = 'RANDGAUSS'
        ###############################################################################
        ###############################################################################
        if ensTypeExperi == 'ENS':
            if window == 0:
                rm_standard_dev = False
                yearsall = np.arange(1950,2019+1,1)
                ravel_modelens = False
                ravelmodeltime = False
            else:
                rm_standard_dev = True
                yearsall = np.arange(1950+window,2019+1,1)
                ravelmodeltime = False
                ravel_modelens = True
        elif ensTypeExperi == 'GCM':
            if window == 0:
                rm_standard_dev = False
                yearsall = np.arange(1950,2019+1,1)
                ravel_modelens = False
                ravelmodeltime = False
            else:
                rm_standard_dev = True
                yearsall = np.arange(1950+window,2019+1,1)
                ravelmodeltime = False
                ravel_modelens = True
        ###############################################################################
        ###############################################################################
        numOfEns = 16
        if len(modelGCMs) == 6:
            lensalso = False
        elif len(modelGCMs) == 7:
            lensalso = True
        lentime = len(yearsall)
        ###############################################################################
        ###############################################################################
        ravelyearsbinary = False
        ravelbinary = False
        num_of_class = lenOfPicks
        ###############################################################################
        ###############################################################################
        lrpRule = 'z'
        normLRP = True
        ###############################################################################
        modelGCMsNames = modelGCMs

        ###############################################################################
        ###############################################################################
        ###############################################################################
        ###############################################################################
        ### Picking experiment to save
        typeOfAnalysis = 'issueWithExperiment'
        
        # Experiment #1
        if rm_ensemble_mean == True:
            if window > 1:
                if calculate_anomalies == False:
                    if rm_merid_mean == False:
                        if rm_observational_mean == False:
                            if rm_annual_mean == False:
                                typeOfAnalysis = 'Experiment-1'
        # Experiment #2
        if rm_ensemble_mean == True:
            if window == 0:
                if calculate_anomalies == False:
                    if rm_merid_mean == False:
                        if rm_observational_mean == False:
                            if rm_annual_mean == False:
                                typeOfAnalysis = 'Experiment-2'
        # Experiment #3 (raw data)
        if rm_ensemble_mean == False:
            if window == 0:
                if calculate_anomalies == False:
                    if rm_merid_mean == False:
                        if rm_observational_mean == False:
                            if rm_annual_mean == False:
                                typeOfAnalysis = 'Experiment-3'
        # Experiment #4
        if rm_ensemble_mean == False:
            if window == 0:
                if calculate_anomalies == False:
                    if rm_merid_mean == False:
                        if rm_observational_mean == False:
                            if rm_annual_mean == True:
                                typeOfAnalysis = 'Experiment-4'
        # Experiment #5
        if rm_ensemble_mean == False:
            if window == 0:
                if calculate_anomalies == False:
                    if rm_merid_mean == False:
                        if rm_observational_mean == True:
                            if rm_annual_mean == False:
                                typeOfAnalysis = 'Experiment-5'
        # Experiment #6
        if rm_ensemble_mean == False:
            if window == 0:
                if calculate_anomalies == False:
                    if rm_merid_mean == False:
                        if rm_observational_mean == True:
                            if rm_annual_mean == True:
                                typeOfAnalysis = 'Experiment-6'
        # Experiment #7
        if rm_ensemble_mean == False:
            if window == 0:
                if calculate_anomalies == True:
                    if rm_merid_mean == False:
                        if rm_observational_mean == True:
                            if rm_annual_mean == False:
                                typeOfAnalysis = 'Experiment-7'
        # Experiment #8
        if rm_ensemble_mean == False:
            if window == 0:
                if calculate_anomalies == True:
                    if rm_merid_mean == False:
                        if rm_observational_mean == False:
                            if rm_annual_mean == False:
                                typeOfAnalysis = 'Experiment-8'
        # Experiment #9
        if rm_ensemble_mean == False:
            if window > 1:
                if calculate_anomalies == True:
                    if rm_merid_mean == False:
                        if rm_observational_mean == False:
                            if rm_annual_mean == False:
                                typeOfAnalysis = 'Experiment-9'
                                
        print('\n<<<<<<<<<<<< Analysis == %s (%s) ! >>>>>>>>>>>>>>>\n' % (typeOfAnalysis,timeper))
        if typeOfAnalysis == 'issueWithExperiment':
            sys.exit('Wrong parameters selected to analyze')
            
        ### Select how to save files
        if land_only == True:
            saveData = timeper + '_' + seasons[0] + '_LAND' + '_NoiseTwinSingleMODDIF4_' + typeOfAnalysis + '_' + variq + '_' + reg_name + '_' + dataset_obs + '_' + 'NumOfSMILE-' + str(num_of_class) + '_Method-' + ensTypeExperi
            typemask = 'LAND'
        elif ocean_only == True:
            saveData = timeper + '_' + seasons[0] + '_OCEAN' + '_NoiseTwinSingleMODDIF4_' + typeOfAnalysis + '_' + variq + '_' + reg_name + '_' + dataset_obs + '_' + 'NumOfSMILE-' + str(num_of_class) + '_Method-' + ensTypeExperi
            typemask = 'OCEAN'
        else:
            saveData = timeper + '_' + seasons[0] + '_NoiseTwinSingleMODDIF4_' + typeOfAnalysis + '_' + variq + '_' + reg_name + '_' + dataset_obs + '_' + 'NumOfSMILE-' + str(num_of_class) + '_Method-' + ensTypeExperi
            typemask = 'GLOBAL'
        print('*Filename == < %s >' % saveData) 
        ###############################################################################
        ###############################################################################
        ###############################################################################
        ###############################################################################
        ### Create sample class labels for each model for my own testing
        if seasons != 'none':
            classesl = np.empty((lenOfPicks,numOfEns,len(yearsall)))
            for i in range(lenOfPicks):
                classesl[i,:,:] = np.full((numOfEns,len(yearsall)),i)  
                
            if ensTypeExperi == 'ENS':
                classeslnew = np.swapaxes(classesl,0,1)
            elif ensTypeExperi == 'GCM':
                classeslnew = classesl

        ###############################################################################
        ###############################################################################
        ###############################################################################
        ###############################################################################                      
        ### Read in observational data
        obsout = np.genfromtxt(directorydata + 'obsConfid_' + saveData + '.txt')
        
        #######################################################################
        ### Pick threshold for minimum confidence
        obsout[np.where(obsout < THRESH)] = 0.
        
        ### Calculate temperature rankings
        rank = np.empty(obsout.shape)
        for i in range(obsout.shape[0]):
            rank[i,:] = abs(sts.rankdata(obsout[i,:],method='min')-8)
            
        rank = np.transpose(rank)
        directorydataMS = '/Users/zlabe/Documents/Research/ModelComparison/Data/RevisitResults_v7/'
        if typeOfAnalysis == 'Experiment-3':
            np.save(directorydataMS + 'Ranks_thresh-%s_%s_%s.npy' % (THRESH,reg_name,dataset_obs),rank)
        elif typeOfAnalysis == 'Experiment-4':
            np.save(directorydataMS + 'Ranks_thresh-%s_%s_%s_GLO.npy' % (THRESH,reg_name,dataset_obs),rank)
        
        