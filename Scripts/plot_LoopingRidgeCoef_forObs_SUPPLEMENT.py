"""
Plots loops through different ridge penalties, but uses same initialization 
seeds and combinations of training/testing/validation data

Reference  : Barnes et al. [2020, JAMES]
Author     : Zachary M. Labe
Date       : 8 March 2022
Version    : 7 - adds validation data for early stopping
"""

### Import packages
import matplotlib.pyplot as plt
import numpy as np
import sys
import scipy.stats as sts
import cmocean

### Plotting defaults 
plt.rc('text',usetex=True)
plt.rc('font',**{'family':'sans-serif','sans-serif':['Avant Garde']}) 

variablesall = ['T2M']
pickSMILEall = [[]] 
ridge_penalty = [0,0.01,0.5,1,5]
con = []
lab = []
###############################################################################
###############################################################################
###############################################################################
### Data preliminaries 
directorydata = '/Users/zlabe/Documents/Research/ModelComparison/Data/LoopL2/'
directoryfigure = '/Users/zlabe/Desktop/ModelComparison_v1/v7/'
letters = ["a","b","c","d","e","f","g","h","i","j","k","l","m","n"]
###############################################################################
###############################################################################
modelGCMs = ['CanESM2','MPI','CSIRO-MK3.6','EC-EARTH','GFDL-CM3','GFDL-ESM2M','LENS']
datasetsingle = ['SMILE']
dataset_obs = 'ERA5BE'
seasons = ['annual']
variq = variablesall[0]
reg_name = 'Arctic'
timeper = 'historical'
###############################################################################
###############################################################################
pickSMILE = pickSMILEall[0]
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
shuffletype = 'RANDGAUSS'
sizeOfTwin = 0 # number of classes to add to other models
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
elif ocean_only == True:
    saveData = timeper + '_' + seasons[0] + '_OCEAN' + '_NoiseTwinSingleMODDIF4_' + typeOfAnalysis + '_' + variq + '_' + reg_name + '_' + dataset_obs + '_' + 'NumOfSMILE-' + str(num_of_class) + '_Method-' + ensTypeExperi
else:
    saveData = timeper + '_' + seasons[0] + '_NoiseTwinSingleMODDIF4_' + typeOfAnalysis + '_' + variq + '_' + reg_name + '_' + dataset_obs + '_' + 'NumOfSMILE-' + str(num_of_class) + '_Method-' + ensTypeExperi
print('*Filename == < %s >' % saveData) 

###############################################################################
###############################################################################
###############################################################################
###############################################################################
### Read in confidence
con = np.asarray(np.load(directorydata + 'Loopl2_obsout_' + saveData + '.npz')['arr_0'])

### Find Label
lab = np.argmax(con,axis=2)
          
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
for r in range(len(lab)*2):
    ax = plt.subplot(2,5,r+1)

    if r < 5:
        obsout = con[r]
        label = np.argmax(obsout,axis=1)
    
        adjust_spines(ax, ['left', 'bottom'])
        ax.spines['top'].set_color('none')
        ax.spines['right'].set_color('none')
        ax.spines['left'].set_color('dimgrey')
        ax.spines['bottom'].set_color('dimgrey')
        ax.spines['left'].set_linewidth(2)
        ax.spines['bottom'].set_linewidth(2)
        ax.tick_params('both',length=4,width=2,which='major',color='dimgrey')
        
        color = cmocean.cm.phase(np.linspace(0.00,1,len(modelGCMsNames)))
        ctest = []
        for i,c in zip(range(len(modelGCMsNames)),color):
            if i == 6:
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
                
        
        if r == 2:
            leg = plt.legend(shadow=False,fontsize=10,loc='upper center',
                          bbox_to_anchor=(0.5,-1.43),fancybox=True,ncol=4,frameon=False,
                          handlelength=0,handletextpad=0)
            for line,text in zip(leg.get_lines(), leg.get_texts()):
                text.set_color(line.get_color())
                
        if r == 0:
            plt.yticks(np.arange(0,1.01,0.1),map(str,np.round(np.arange(0,1.01,0.1),2)),size=6)
        else:
            plt.yticks(np.arange(0,1.01,0.1),map(str,np.round(np.arange(0,1.01,0.1),2)),size=6)
            ax.set_yticklabels([])
        
        plt.xticks(np.arange(1950,2030+1,20),map(str,np.arange(1950,2030+1,20)),size=6)
        plt.xlim([1950,2020])   
        plt.ylim([0,1.0])    
        
        plt.text(1950,1.04,r'\textbf{[%s]}' % letters[r],color='k',fontsize=6,
                 ha='center',va='center')
        
        if r == 0:  
            plt.ylabel(r'\textbf{Confidence}',color='dimgrey',fontsize=8,labelpad=39)
        plt.title(r'\textbf{L$_{2}$=%s}' % (ridge_penalty[r]),color='dimgrey',fontsize=10)
    else:
        obspred = lab[r-5]
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
            plt.scatter(x[cct],obspred[cct],color=col,s=9,clip_on=False,
                        edgecolor='none',linewidth=0.4,zorder=10)
        
        plt.xticks(np.arange(1950,2030+1,20),map(str,np.arange(1950,2030+1,20)),size=6)
        if r-5 == 0:
            plt.yticks(np.arange(0,lenOfPicks+1,1),modelGCMsNames,size=6)
        else:
            plt.yticks(np.arange(0,lenOfPicks+1,1),modelGCMsNames,size=6)
            ax.set_yticklabels([])
        plt.xlim([1950,2020])   
        plt.ylim([0,lenOfPicks-1])
        if r-5 == 0:  
            plt.ylabel(r'\textbf{Prediction}',color='dimgrey',fontsize=8,labelpad=7.5)
            
        plt.text(1950,6.25,r'\textbf{[%s]}' % letters[r],color='k',fontsize=6,
                 ha='center',va='center')
        
# plt.tight_layout()
plt.subplots_adjust(hspace=0.3,bottom=0.15)
plt.savefig(directoryfigure + 'LoopingRidgeCoef_forObs_SUPPLEMENT.png',dpi=900)