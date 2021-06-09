"""
Linear model for evaluating model biases, differences, and other thresholds 
using explainable AI for historical data

Author     : Zachary M. Labe
Date       : 18 May 2021
Version    : 1 - adds extra class (#8), but tries the MMean
"""

### Import packages
import matplotlib.pyplot as plt
import numpy as np
import sys
from netCDF4 import Dataset
from mpl_toolkits.basemap import Basemap, addcyclic, shiftgrid
import palettable.cubehelix as cm
import palettable.cartocolors.qualitative as cc
from sklearn.metrics import accuracy_score
import scipy.stats as sts
import cmasher as cmr
import cmocean

### Plotting defaults 
plt.rc('text',usetex=True)
plt.rc('font',**{'family':'sans-serif','sans-serif':['Avant Garde']}) 

variablesall = ['T2M','P','SLP']
variablesall = ['P']
pickSMILEall = [[]] 
ridge_penalty = [0,0.01,0.1,1,5]
for va in range(len(variablesall)):
    for m in range(len(pickSMILEall)):
        weights = []
        for ww in range(len(ridge_penalty)):
            ###############################################################################
            ###############################################################################
            ###############################################################################
            ### Data preliminaries 
            directorydata = '/Users/zlabe/Documents/Research/ModelComparison/Data/'
            directoryfigure = '/Users/zlabe/Desktop/ModelComparison_v1/v2-LINEAR/'
            letters = ["a","b","c","d","e","f","g","h","i","j","k","l","m","n"]
            ###############################################################################
            ###############################################################################
            modelGCMs = ['CanESM2-ens','MPI-ens','CSIRO-MK3.6-ens','KNMI-ecearth-ens',
                          'GFDL-CM3-ens','GFDL-ESM2M-ens','LENS-ens']
            datasetsingle = ['SMILE']
            dataset_obs = 'ERA5BE'
            seasons = ['annual']
            variq = variablesall[va]
            reg_name = 'SMILEGlobe'
            timeper = 'historical'
            ###############################################################################
            ###############################################################################
            pickSMILE = pickSMILEall[m]
            if len(pickSMILE) >= 1:
                lenOfPicks = len(pickSMILE) + 1 # For random class
            else:
                lenOfPicks = len(modelGCMs) + 1 # For random class
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
            # integer = 5 # random noise value to add/subtract from each grid point
            sizeOfTwin = 1 # number of classes to add to other models
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
            modelGCMsNames = np.append(modelGCMs,['MMmean'])
    
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
                saveData = timeper + '_LAND' + '_LINEAR_MODDIF4_' + typeOfAnalysis + '_' + variq + '_' + reg_name + '_' + dataset_obs + '_' + 'NumOfSMILE-' + str(num_of_class) + '_Method-' + ensTypeExperi
                typemask = 'LAND'
            elif ocean_only == True:
                saveData = timeper + '_OCEAN' + '_LINEAR_MODDIF4_' + typeOfAnalysis + '_' + variq + '_' + reg_name + '_' + dataset_obs + '_' + 'NumOfSMILE-' + str(num_of_class) + '_Method-' + ensTypeExperi
                typemask = 'OCEAN'
            else:
                saveData = timeper + '_LINEAR_MODDIF4_' + typeOfAnalysis + '_' + variq + '_' + reg_name + '_' + dataset_obs + '_' + 'NumOfSMILE-' + str(num_of_class) + '_Method-' + ensTypeExperi
                typemask = 'GLOBAL'
            print('*Filename == < %s >' % saveData) 
            
            ### Adding new file name for linear model
            saveData = saveData + '_L2-%s' % ridge_penalty[ww]
            print('\n>>>NEW FILE NAME = %s\n' % saveData)
            
            ###############################################################################
            ###############################################################################
            ###############################################################################
            ###############################################################################
            ### Read in regression weights
            weightsn = np.genfromtxt(directorydata + 'weights_' + saveData + '.txt')
            weights.append(weightsn)
            
            ### Read in some latitude and longitudes
            lat1 = np.load('/Users/zlabe/Documents/Research/ModelComparison/Data/Climatologies/annual_T2M_SMILEGlobe_historical_PointByPoint_lats.npz')['arr_0']
            lon1 = np.load('/Users/zlabe/Documents/Research/ModelComparison/Data/Climatologies/annual_T2M_SMILEGlobe_historical_PointByPoint_lons.npz')['arr_0']

###############################################################################
###############################################################################
###############################################################################
### Plot subplot of LRP means training
limit = np.arange(-0.025,0.02501,0.0001)
barlim = np.round(np.arange(-0.025,0.02501,0.025),3)
cmap = cmocean.cm.balance
label = r'\textbf{Linear Regression Weights - [ %s ] - %s}' % (variq,typeOfAnalysis)

fig = plt.figure(figsize=(10,2))
for r in range(len(ridge_penalty)):
    var = weights[r]
    
    ax1 = plt.subplot(1,len(ridge_penalty),r+1)
    m = Basemap(projection='moll',lon_0=0,resolution='l',area_thresh=10000)
    m.drawcoastlines(color='dimgrey',linewidth=0.27)
        
    var, lons_cyclic = addcyclic(var, lon1)
    var, lons_cyclic = shiftgrid(180., var, lons_cyclic, start=False)
    lon2d, lat2d = np.meshgrid(lons_cyclic, lat1)
    x, y = m(lon2d, lat2d)
       
    circle = m.drawmapboundary(fill_color='dimgrey',color='dimgray',
                      linewidth=0.7)
    circle.set_clip_on(False)
    
    cs1 = m.contourf(x,y,var,limit,extend='both')
    cs1.set_cmap(cmap) 
    
    if ocean_only == True:
        m.fillcontinents(color='dimgrey',lake_color='dimgrey')
    elif land_only == True:
        m.drawlsmask(land_color=(0,0,0,0),ocean_color='darkgrey',lakes=True,zorder=5)
            
    ax1.annotate(r'\textbf{L$_{2}$=%s}' % ridge_penalty[r],xy=(0,0),xytext=(0.5,1.10),
                  textcoords='axes fraction',color='dimgrey',fontsize=8,
                  rotation=0,ha='center',va='center')
    ax1.annotate(r'\textbf{[%s]}' % letters[r],xy=(0,0),xytext=(0.86,0.97),
                  textcoords='axes fraction',color='k',fontsize=6,
                  rotation=330,ha='center',va='center')
    
###############################################################################
cbar_ax1 = fig.add_axes([0.36,0.15,0.3,0.03])                
cbar1 = fig.colorbar(cs1,cax=cbar_ax1,orientation='horizontal',
                    extend='both',extendfrac=0.07,drawedges=False)
cbar1.set_label(label,fontsize=9,color='dimgrey',labelpad=1.4)  
cbar1.set_ticks(barlim)
cbar1.set_ticklabels(list(map(str,barlim)))
cbar1.ax.tick_params(axis='x', size=.01,labelsize=5)
cbar1.outline.set_edgecolor('dimgrey')

plt.tight_layout()
if lenOfPicks == 3:
    plt.subplots_adjust(top=0.85,wspace=0.02,hspace=0.02,bottom=0.24)
else: 
    plt.subplots_adjust(top=0.85,wspace=0.02,hspace=0.02,bottom=0.14)

plt.savefig(directoryfigure + '%s/LinearWeightsL2_%s.png' % (typeOfAnalysis,saveData),dpi=300)