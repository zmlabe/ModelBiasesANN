"""
Script for plotting softmax confidence after testing on observations for
regional masks for looping iterations

Author     : Zachary M. Labe
Date       : 1 June 2021
Version    : 4 (ANNv4)
"""

### Import packages
import sys
import matplotlib.pyplot as plt
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

variablesall = ['T2M']
pickSMILEall = [[]] 
latarctic = 70
obsoutall = []
regions = ['SMILEglobe','NH','SH','narrowTropics','Arctic','SouthernOcean']
regionnames = ['GLOBE','N. HEMISPHERE','S. HEMISPHERE','TROPICS','ARCTIC(%s)' % latarctic,'SOUTHERN OCEAN']
for va in range(len(variablesall)):
    for m in range(len(pickSMILEall)):
        for rr in range(len(regions)):
            ###############################################################################
            ###############################################################################
            ###############################################################################
            ### Data preliminaries 
            directorydata = '/Users/zlabe/Documents/Research/ModelComparison/Data/Loop/'
            directoryfigure = '/Users/zlabe/Desktop/ModelComparison_v1/v2-Mmean/'
            letters = ["a","b","c","d","e","f","g","h","i","j","k","l","m","n"]
            ###############################################################################
            ###############################################################################
            modelGCMs = ['CanESM2','MPI','CSIRO-MK3.6','KNMI-ecearth',
                          'GFDL-CM3','GFDL-ESM2M','LENS']
            datasetsingle = ['SMILE']
            dataset_obs = 'ERA5BE'
            seasons = ['annual']
            variq = variablesall[va]
            reg_name = regions[rr]
            if reg_name == 'Arctic':
                reg_name = 'Arctic%s' % latarctic
            timeper = 'historical'
            SAMPLEQ = 100
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
            modelGCMsNames = np.append(modelGCMs,['MMean'])
    
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
                                    
            print('\n<<<<<<<<<<<< Analysis == %s (%s) ! >>>>>>>>>>>>>>>' % (typeOfAnalysis,timeper))
            if typeOfAnalysis == 'issueWithExperiment':
                sys.exit('Wrong parameters selected to analyze')
              
            ### Select how to save files
            if land_only == True:
                saveData = str(SAMPLEQ) + '_' + timeper + '_' + seasons[0] + '_LAND' + '_NoiseTwinSingleMODDIF4_' + typeOfAnalysis + '_' + variq + '_' + reg_name + '_' + dataset_obs + '_' + 'NumOfSMILE-' + str(num_of_class) + '_Method-' + ensTypeExperi
            elif ocean_only == True:
                saveData = str(SAMPLEQ) + '_' + timeper + '_' + seasons[0] + '_OCEAN' + '_NoiseTwinSingleMODDIF4_' + typeOfAnalysis + '_' + variq + '_' + reg_name + '_' + dataset_obs + '_' + 'NumOfSMILE-' + str(num_of_class) + '_Method-' + ensTypeExperi
            else:
                saveData = str(SAMPLEQ) + '_' + timeper + '_' + seasons[0] + '_NoiseTwinSingleMODDIF4_' + typeOfAnalysis + '_' + variq + '_' + reg_name + '_' + dataset_obs + '_' + 'NumOfSMILE-' + str(num_of_class) + '_Method-' + ensTypeExperi
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
                    
                ### Add random noise models
                randomNoiseClass = np.full((sizeOfTwin,numOfEns,len(yearsall)),i+1)
                classesl = np.append(classesl,randomNoiseClass,axis=0)
                    
                if ensTypeExperi == 'ENS':
                    classeslnew = np.swapaxes(classesl,0,1)
                elif ensTypeExperi == 'GCM':
                    classeslnew = classesl
    
            ###############################################################################
            ###############################################################################
            ###############################################################################
            ###############################################################################       
            ### Read in data
                    
            ### Read in observational data
            obsout= np.load(directorydata + 'obsout_' + saveData + '.npz')['arr_0'][:]
            obsoutall.append(obsout)
            
###############################################################################
###############################################################################
###############################################################################
### See all regional data
conf = np.asarray(obsoutall).squeeze()

### Pick arg max
mean = np.nanmean(conf,axis=1)
maxc = np.argmax(mean,axis=2)

### Count how many per year
argyr = np.argmax(conf,axis=3)
count = np.empty((maxc.shape))
for ri in range(argyr.shape[0]):
    for yy in range(argyr.shape[2]):
        yrco = argyr[ri,:,yy]
        count[ri,yy] = np.count_nonzero(yrco == maxc[ri,yy])

###############################################################################
###############################################################################
###############################################################################
def adjust_spines(ax, spines):
    for loc, spine in ax.spines.items():
        if loc in spines:
            spine.set_position(('outward', 5))
        else:
            spine.set_color('none')  
    if 'left' in spines:
        ax.yaxis.set_ticks_position('left')
    else:
        ax.yaxis.set_ticks([])

    if 'bottom' in spines:
        ax.xaxis.set_ticks_position('bottom')
    else:
            ax.xaxis.set_ticks([]) 
            
### Begin plot
fig = plt.figure(figsize=(8,5))
color=cc.Antique_6.mpl_colormap(np.linspace(0,1,len(regions)))
for r,c in zip(range(len(regions)),color):
    ax = plt.subplot(2,3,r+1)

    obspred = maxc[r]
    obscoun = count[r]
    adjust_spines(ax, ['left', 'bottom'])
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.spines['left'].set_color('dimgrey')
    ax.spines['bottom'].set_color('dimgrey')
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.tick_params('both',length=4,width=2,which='major',color='dimgrey')
    ax.yaxis.grid(zorder=1,color='dimgrey',alpha=0.35)
    
    x=np.arange(1950,2019+1,1)
    color = cmr.infinity(np.linspace(0.00,1,len(modelGCMsNames)))
    ctest = []
    for i,c in zip(range(len(modelGCMsNames)),color):
        if i == 7:
            c = 'k'
        else:
            c = c
        ctest.append(c)
    for cct in range(len(obspred)):
        if obspred[cct] == 0:
            col = ctest[0]
        elif obspred[cct] == 1:
            col = ctest[1]
        elif obspred[cct] == 2:
            col = ctest[2]
        elif obspred[cct] == 3:
            col = ctest[3]
        elif obspred[cct] == 4:
            col = ctest[4]
        elif obspred[cct] == 5:
            col = ctest[5]
        elif obspred[cct] == 6:
            col = ctest[6]
        elif obspred[cct] == 7:
            col = ctest[7]
        plt.scatter(x[cct],obspred[cct],color=col,s=9,clip_on=False,
                    edgecolor='none',linewidth=0.4,zorder=10,alpha=obscoun[cct]/100)
    
    plt.xticks(np.arange(1950,2030+1,20),map(str,np.arange(1950,2030+1,20)),size=5)
    if any([r==0,r==3]):
        plt.yticks(np.arange(0,lenOfPicks+1,1),modelGCMsNames,size=3)
    else:
        plt.yticks(np.arange(0,lenOfPicks+1,1),modelGCMsNames,size=3)
        ax.set_yticklabels([])
    plt.xlim([1950,2020])   
    plt.ylim([0,lenOfPicks-1])
    if r == 0:
        plt.text(1923,-3.5,r'\textbf{Average Prediction}',color='k',
             fontsize=11,ha='left',rotation=90)       
    plt.text(1950,0.1,r'\textbf{%s}' % regionnames[r],color='dimgrey',
         fontsize=15,ha='left')  
        
# plt.tight_layout()
# plt.subplots_adjust(bottom=0.15)
plt.savefig(directoryfigure + '%s/Regions/Confidence/RegionsMaxLabel_%s.png' % (typeOfAnalysis,latarctic),dpi=300)