"""
Comparing ANN and linear model for evaluating model biases, differences, and 
other thresholds using explainable AI for historical data (and SMOOTHED data)

Author     : Zachary M. Labe
Date       : 19 May 2021
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
from statsmodels.nonparametric.smoothers_lowess import lowess

### Plotting defaults 
plt.rc('text',usetex=True)
plt.rc('font',**{'family':'sans-serif','sans-serif':['Avant Garde']}) 

variablesall = ['T2M','P','SLP']
variablesall = ['T2M']
pickSMILEall = [[]] 
ridge_penalty = [0,0.1]
lab = ['LINEAR','LINEAR-L$_{2}$=0.1','ANN-L$_{2}$=0.1','ANN(smooth)-L$_{2}$=0.1']
for va in range(len(variablesall)):
    for m in range(len(pickSMILEall)):
        ###############################################################################
        ###############################################################################
        ###############################################################################
        ### Data preliminaries 
        directorydata = '/Users/zlabe/Documents/Research/ModelComparison/Data/'
        directoryfigure = '/Users/zlabe/Desktop/ModelComparison_v1/v2-LINEAR/'
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
            saveData = timeper + '_' + seasons[0] + '_LAND' + '_LINEAR_MODDIF4_' + typeOfAnalysis + '_' + variq + '_' + reg_name + '_' + dataset_obs + '_' + 'NumOfSMILE-' + str(num_of_class) + '_Method-' + ensTypeExperi
            typemask = 'LAND'
        elif ocean_only == True:
            saveData = timeper + '_' + seasons[0] + '_OCEAN' + '_LINEAR_MODDIF4_' + typeOfAnalysis + '_' + variq + '_' + reg_name + '_' + dataset_obs + '_' + 'NumOfSMILE-' + str(num_of_class) + '_Method-' + ensTypeExperi
            typemask = 'OCEAN'
        else:
            saveData = timeper + '_' + seasons[0] + '_LINEAR_MODDIF4_' + typeOfAnalysis + '_' + variq + '_' + reg_name + '_' + dataset_obs + '_' + 'NumOfSMILE-' + str(num_of_class) + '_Method-' + ensTypeExperi
            typemask = 'GLOBAL'
        print('*Filename == < %s >' % saveData) 
        
        ###############################################################################
        ###############################################################################
        ###############################################################################
        ###############################################################################
        ### Select how to save files for ANN
        if land_only == True:
            saveDataANN = timeper + '_' + seasons[0] + '_LAND' + '_NoiseTwinSingleMODDIF4_' + typeOfAnalysis + '_' + variq + '_' + reg_name + '_' + dataset_obs + '_' + 'NumOfSMILE-' + str(num_of_class) + '_Method-' + ensTypeExperi
        elif ocean_only == True:
            saveDataANN = timeper + '_' + seasons[0] + '_OCEAN' + '_NoiseTwinSingleMODDIF4_' + typeOfAnalysis + '_' + variq + '_' + reg_name + '_' + dataset_obs + '_' + 'NumOfSMILE-' + str(num_of_class) + '_Method-' + ensTypeExperi
        else:
            saveDataANN = timeper + '_' + seasons[0] + '_NoiseTwinSingleMODDIF4_' + typeOfAnalysis + '_' + variq + '_' + reg_name + '_' + dataset_obs + '_' + 'NumOfSMILE-' + str(num_of_class) + '_Method-' + ensTypeExperi
        print('*Filename == < %s >' % saveDataANN) 
        
        ###############################################################################
        ###############################################################################
        ###############################################################################
        ###############################################################################
        ### Select how to save files for smoothed ANN
        if land_only == True:
            saveDataANNs = timeper + '_' + seasons[0] + '_LAND' + '_NoiseTwinSingleMODDIF4_SMOOTHER_' + typeOfAnalysis + '_' + variq + '_' + reg_name + '_' + dataset_obs + '_' + 'NumOfSMILE-' + str(num_of_class) + '_Method-' + ensTypeExperi
        elif ocean_only == True:
            saveDataANNs = timeper + '_' + seasons[0] + '_OCEAN' + '_NoiseTwinSingleMODDIF4_SMOOTHER_' + typeOfAnalysis + '_' + variq + '_' + reg_name + '_' + dataset_obs + '_' + 'NumOfSMILE-' + str(num_of_class) + '_Method-' + ensTypeExperi
        else:
            saveDataANNs = timeper + '_' + seasons[0] + '_NoiseTwinSingleMODDIF4_SMOOTHER_' + typeOfAnalysis + '_' + variq + '_' + reg_name + '_' + dataset_obs + '_' + 'NumOfSMILE-' + str(num_of_class) + '_Method-' + ensTypeExperi
        print('*Filename == < %s >' % saveDataANNs) 
        
        ###############################################################################
        ###############################################################################
        ###############################################################################
        ###############################################################################
        ### Read in confidence for linear model
        linear_lab = np.load(directorydata + 'PlotLINEARLabels_' + saveData + '.npz')['arr_0']
        linear_conf = np.load(directorydata + 'PlotLINEARConfidence_' + saveData + '.npz')['arr_0']
        
        ann_lab = np.int_(np.genfromtxt(directorydata + 'obsLabels_' + saveDataANN + '.txt'))
        ann_conf = np.genfromtxt(directorydata + 'obsConfid_' + saveDataANN + '.txt')
        
        ann_labs = np.int_(np.genfromtxt(directorydata + 'obsLabels_' + saveDataANNs + '.txt'))
        ann_confs = np.genfromtxt(directorydata + 'obsConfid_' + saveDataANNs + '.txt')
        
        ### Combine data together
        pred = [linear_lab[0],linear_lab[1],ann_lab,ann_labs]
        conf = [linear_conf[0],linear_conf[1],ann_conf,ann_confs]
        
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
fig = plt.figure(figsize=(9,5))
for r in range(len(lab)*2):
    ax = plt.subplot(2,4,r+1)

    if r < 4:
        obsout = conf[r]
        label = np.argmax(obsout,axis=1)
    
        adjust_spines(ax, ['left', 'bottom'])
        ax.spines['top'].set_color('none')
        ax.spines['right'].set_color('none')
        ax.spines['left'].set_color('dimgrey')
        ax.spines['bottom'].set_color('dimgrey')
        ax.spines['left'].set_linewidth(2)
        ax.spines['bottom'].set_linewidth(2)
        ax.tick_params('both',length=4,width=2,which='major',color='dimgrey')
        # ax.yaxis.grid(zorder=1,color='dimgrey',alpha=0.3)
        
        color = cmr.infinity(np.linspace(0.00,1,len(modelGCMsNames)))
        ctest = []
        for i,c in zip(range(len(modelGCMsNames)),color):
            if i == 7:
                c = 'k'
            else:
                c = c
            plt.plot(yearsall,obsout[:,i],color=c,linewidth=0.2,
                        label=r'\textbf{%s}' % modelGCMsNames[i],zorder=1,
                        clip_on=False,alpha=0)
            plt.scatter(yearsall,obsout[:,i],color=c,s=9,zorder=12,
                        clip_on=False,alpha=0.2,edgecolors='none')
            ctest.append(c)
            
            for yr in range(yearsall.shape[0]):
                la = label[yr]
                if i == la:
                    plt.scatter(yearsall[yr],obsout[yr,i],color=c,s=9,zorder=12,
                                clip_on=False,alpha=1,edgecolors='none')
        
        low = lowess(np.nanmax(obsout,axis=1),yearsall,frac=1/3)
        plt.plot(yearsall,low[:,1],linestyle='-',linewidth=0.5,color='r')
                
        
        if r == 1:
            leg = plt.legend(shadow=False,fontsize=6,loc='upper center',
                          bbox_to_anchor=(1.2,-1.35),fancybox=True,ncol=4,frameon=False,
                          handlelength=0,handletextpad=0)
            for line,text in zip(leg.get_lines(), leg.get_texts()):
                text.set_color(line.get_color())
                
        if r == 0:
            plt.yticks(np.arange(0,1.01,0.1),map(str,np.round(np.arange(0,1.01,0.1),2)),size=3)
        else:
            plt.yticks(np.arange(0,1.01,0.1),map(str,np.round(np.arange(0,1.01,0.1),2)),size=3)
            ax.set_yticklabels([])
        
        plt.xticks(np.arange(1950,2030+1,20),map(str,np.arange(1950,2030+1,20)),size=5)
        plt.xlim([1950,2020])   
        plt.ylim([0,1.0])           
        
        if r == 0:  
            if land_only == True:
                plt.ylabel(r'\textbf{Confidence [%s-%s-LAND-%s]' % (seasons[0],variq,reg_name),color='dimgrey',fontsize=6,labelpad=23)
            elif ocean_only == True:
                plt.ylabel(r'\textbf{Confidence [%s-%s-OCEAN-%s]' % (seasons[0],variq,reg_name),color='dimgrey',fontsize=6,labelpad=23)
            else:
                plt.ylabel(r'\textbf{Confidence [%s-%s-%s]' % (seasons[0],variq,reg_name),color='dimgrey',fontsize=6,labelpad=23)
        plt.title(r'\textbf{%s}' % (lab[r]),color='dimgrey',fontsize=10)
    else:
        obspred = pred[r-4]
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
                        edgecolor='none',linewidth=0.4,zorder=10)
        
        plt.xticks(np.arange(1950,2030+1,20),map(str,np.arange(1950,2030+1,20)),size=5)
        if r-4 == 0:
            plt.yticks(np.arange(0,lenOfPicks+1,1),modelGCMsNames,size=3)
        else:
            plt.yticks(np.arange(0,lenOfPicks+1,1),modelGCMsNames,size=3)
            ax.set_yticklabels([])
        plt.xlim([1950,2020])   
        plt.ylim([0,lenOfPicks-1])
        if r-4 == 0:  
            if land_only == True:
                plt.ylabel(r'\textbf{Prediction [%s-%s-LAND-%s]' % (seasons[0],variq,reg_name),color='dimgrey',fontsize=6,labelpad=7.5)
            elif ocean_only == True:
                plt.ylabel(r'\textbf{Prediction [%s-%s-OCEAN-%s]' % (seasons[0],variq,reg_name),color='dimgrey',fontsize=6,labelpad=7.5)
            else:
                plt.ylabel(r'\textbf{Prediction [%s-%s-%s]' % (seasons[0],variq,reg_name),color='dimgrey',fontsize=6,labelpad=7.5)
        
# plt.tight_layout()
# plt.subplots_adjust(bottom=0.15)
plt.savefig(directoryfigure + '%s/Confidence/LinearANNsmoothcomparison_%s.png' % (typeOfAnalysis,saveDataANN),dpi=300)