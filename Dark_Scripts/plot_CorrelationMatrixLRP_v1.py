"""
Calculate spatial correlations between LRP maps

Reference  : Barnes et al. [2020, JAMES]
Author     : Zachary M. Labe
Date       : 11 March 2021
Version    : 1 
"""

### Import packages
import matplotlib.pyplot as plt
import numpy as np
import sys
from netCDF4 import Dataset
from mpl_toolkits.basemap import Basemap, addcyclic, shiftgrid
import palettable.cubehelix as cm
import palettable.cartocolors.qualitative as cc
import matplotlib.colors as c
from sklearn.metrics import accuracy_score
import calc_Utilities as UT
import cmocean

### Plotting defaults 
plt.rc('text',usetex=True)
plt.rc('font',**{'family':'sans-serif','sans-serif':['Avant Garde']}) 
directoryfigure = '/Users/zlabe/Desktop/ModelComparison_v1/v2/'

### Set parameters
pickSMILEall = [
                [],
                ['CSIRO-MK3.6','lens'],
                ['CSIRO-MK3.6','GFDL-CM3','LENS'],
                ['CanESM2','CSIRO-MK3.6','GFDL-CM3','GFDL-ESM2M','LENS']
                ] 

### Read function
def readLRPMapData(variablesall,pickSMILEall):
    lrpallmaps = []
    for m in range(len(pickSMILEall)):
        ###############################################################################
        ###############################################################################
        ###############################################################################
        ### Data preliminaries 
        directorydata = '/Users/zlabe/Documents/Research/ModelComparison/Data/'
        directoryfigure = '/Users/zlabe/Desktop/ModelComparison_v1/v2/'
        letters = ["a","b","c","d","e","f","g","h","i","j","k","l","m","n"]
        ###############################################################################
        ###############################################################################
        modelGCMs = ['CanESM2','MPI','CSIRO-MK3.6','KNMI-ecearth',
                      'GFDL-CM3','GFDL-ESM2M','LENS']
        datasetsingle = ['SMILE']
        dataset_obs = 'ERA5BE'
        seasons = ['annual']
        variq = variablesall[0]
        reg_name = 'SMILEGlobe'
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
        rm_observational_mean = True
        ###############################################################################
        ###############################################################################
        calculate_anomalies = True
        if calculate_anomalies == True:
            baseline = np.arange(1951,1980+1,1)
        ###############################################################################
        ###############################################################################
        window = 0
        ensTypeExperi = 'ENS'
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
        ###############################################################################
        if lenOfPicks == 7:
            modelGCMsNames = modelGCMs
        else:
            modelGCMsNames = pickSMILE 
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
                                
        print('\n<<<<<<<<<<<< Analysis == %s ! >>>>>>>>>>>>>>>\n' % typeOfAnalysis)
        if typeOfAnalysis == 'issueWithExperiment':
            sys.exit('Wrong parameters selected to analyze')
            
        ### Select how to save files
        saveData = typeOfAnalysis + '_' + variq + '_' + reg_name + '_' + dataset_obs + '_' + 'NumOfSMILE-' + str(num_of_class) + '_Method-' + ensTypeExperi
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
        ### Read in data
        def readData(directory,typemodel,saveData):
            """
            Read in LRP maps
            """
            
            name = 'LRPMap' + typemodel + '_' + saveData + '.nc'
            filename = directory + name
            data = Dataset(filename)
            lat = data.variables['lat'][:]
            lon = data.variables['lon'][:]
            lrp = data.variables['LRP'][:]
            data.close()
            
            return lrp,lat,lon
        
        ### Read in training and testing predictions and labels
        classesltrain = np.int_(np.genfromtxt(directorydata + 'trainingTrueLabels_' + saveData + '.txt'))      
        predtrain = np.int_(np.genfromtxt(directorydata + 'trainingPredictedLabels_' + saveData + '.txt'))
        
        ### Read in observational data
        obspred = np.int_(np.genfromtxt(directorydata + 'obsLabels_' + saveData + '.txt'))
        uniqueobs,countobs = np.unique(obspred,return_counts=True)
        percPickObs = np.nanmax(countobs)/len(obspred)*100.
        modelPickObs = modelGCMsNames[uniqueobs[np.argmax(countobs)]]
        
        randpred = np.int_(np.genfromtxt(directorydata + 'RandLabels_' + saveData + '.txt'))
        uniquerand,countrand = np.unique(randpred,return_counts=True)
        percPickrand = np.nanmax(countrand)/len(randpred)*100.
        modelPickrand= modelGCMsNames[uniquerand[np.argmax(countrand)]]
        
        ### Read in LRP maps
        lrptraindata,lat1,lon1 = readData(directorydata,'Training',saveData)
        
        ###############################################################################
        ###############################################################################
        ############################################################################### 
        ### Find which model
        model_train = []
        for i in range(lenOfPicks):
            modelloc = np.where((predtrain == int(i)))[0]
            model_train.append(modelloc)
        
        ### Composite lrp maps of the models
        lrptrain = []
        for i in range(lenOfPicks):
            lrpmodel = lrptraindata[model_train[i]]
            lrpmodelmean = np.nanmean(lrpmodel,axis=0)
            lrptrain.append(lrpmodelmean)
        lrptrain = np.asarray(lrptrain,dtype=object)
        
        lrpallmaps.append(lrptrain)
    return lrpallmaps,lat1,lon1,saveData

### Read data
lrpt,lat,lon,saveData = readLRPMapData(['T2M'],pickSMILEall)
lrpp,lat,lon,saveData = readLRPMapData(['P'],pickSMILEall)

### Model combinations
pickSMILEall = ['CCCma_canesm2','MPI','CSIRO_MK3.6','KNMI_ecearth','GFDL_CM3','GFDL_ESM2M','lens',
                ['CSIRO-MK3.6','LENS'],
                ['CSIRO-MK3.6','GFDL-CM3','LENS'],
                ['CanESM2','CSIRO-MK3.6','GFDL-CM3','GFDL-ESM2M','LENS']] 

def calcCorrMatrix(lrpt,lat,lon):
    ### CSIRO-MK3.6
    corr_cs7c = UT.calc_spatialCorr(lrpt[0][2],lrpt[3][1],lat,lon,'yes')
    corr_cs7b = UT.calc_spatialCorr(lrpt[0][2],lrpt[2][0],lat,lon,'yes')
    corr_cs7a = UT.calc_spatialCorr(lrpt[0][2],lrpt[1][0],lat,lon,'yes')
    
    corr_cs5b = UT.calc_spatialCorr(lrpt[3][1],lrpt[2][0],lat,lon,'yes')
    corr_cs5a = UT.calc_spatialCorr(lrpt[3][1],lrpt[1][0],lat,lon,'yes')
    
    corr_cs3a = UT.calc_spatialCorr(lrpt[2][0],lrpt[1][0],lat,lon,'yes')
    
    corrs_csall = [corr_cs7c,corr_cs7b,corr_cs7a,
              corr_cs5a,corr_cs5b,
              corr_cs3a]
    
    ### LENS
    corr_lens7c = UT.calc_spatialCorr(lrpt[0][-1],lrpt[3][-1],lat,lon,'yes')
    corr_lens7b = UT.calc_spatialCorr(lrpt[0][-1],lrpt[2][-1],lat,lon,'yes')
    corr_lens7a = UT.calc_spatialCorr(lrpt[0][-1],lrpt[1][-1],lat,lon,'yes')
    
    corr_lens5b = UT.calc_spatialCorr(lrpt[3][-1],lrpt[2][-1],lat,lon,'yes')
    corr_lens5a = UT.calc_spatialCorr(lrpt[3][-1],lrpt[1][-1],lat,lon,'yes')
    
    corr_lens3a = UT.calc_spatialCorr(lrpt[2][-1],lrpt[1][-1],lat,lon,'yes')
    
    corrs_lensall = [corr_lens7c,corr_lens7b,corr_lens7a,
              corr_lens5a,corr_lens5b,
              corr_lens3a]
    
    return corrs_csall,corrs_lensall

cs_corrt,lens_corrt = calcCorrMatrix(lrpt,lat,lon)
cs_corrp,lens_corrp = calcCorrMatrix(lrpp,lat,lon)

allcst = np.array([np.append(cs_corrt[0:3],[1]),np.append(cs_corrt[3:5],np.append(1,[np.nan]*1)),np.append(cs_corrt[5:6],np.append(1,[np.nan]*2)),np.append(1,[np.nan]*3)])
allcsp = np.array([np.append(cs_corrp[0:3],[1]),np.append(cs_corrp[3:5],np.append(1,[np.nan]*1)),np.append(cs_corrp[5:6],np.append(1,[np.nan]*2)),np.append(1,[np.nan]*3)])

alllenst = np.array([np.append(lens_corrt[0:3],[1]),np.append(lens_corrt[3:5],np.append(1,[np.nan]*1)),np.append(lens_corrt[5:6],np.append(1,[np.nan]*2)),np.append(1,[np.nan]*3)])
alllensp = np.array([np.append(lens_corrp[0:3],[1]),np.append(lens_corrp[3:5],np.append(1,[np.nan]*1)),np.append(lens_corrp[5:6],np.append(1,[np.nan]*2)),np.append(1,[np.nan]*3)])

###############################################################################
###############################################################################
###############################################################################
### Plot correlation matrix
fig = plt.figure()
ax = plt.subplot(111)
var = allcst

ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')
ax.spines['bottom'].set_color('none')
ax.spines['left'].set_color('none')
ax.get_xaxis().set_tick_params(direction='out', width=2,length=3,
            color='darkgrey')

plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom='on')
plt.tick_params(
    axis='y',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    left=False,      # ticks along the bottom edge are off
    right=False,         # ticks along the top edge are off
    labelleft='on')

csm=plt.get_cmap(cmocean.cm.balance)
norm = c.BoundaryNorm(np.arange(-1,1.1,0.1),csm.N)
cs = plt.pcolormesh(var,shading='faceted',edgecolor='dimgrey',linewidth=1.2,
                    vmin=-1,vmax=1,norm=norm,cmap=csm)

for i in range(var.shape[0]):
    for j in range(var.shape[1]):
        if np.isnan(var[i,j]):
            plt.text(j+0.5,i+0.5,r'\textbf{%s}' % np.round(var[i,j],2),fontsize=10,
                color='w',va='center',ha='center')
        else:
            plt.text(j+0.5,i+0.5,r'\textbf{%s}' % np.round(var[i,j],2),fontsize=10,
                color='k',va='center',ha='center')
                 
cbar = plt.colorbar(cs,orientation='horizontal',aspect=50,pad=0.12)
cbar.set_label(r'\textbf{Correlation Coefficient}',
               color='dimgrey',labelpad=7,fontsize=13)
barlim = np.round(np.arange(-1,1.1,0.25),2)
cbar.set_ticks(barlim)
cbar.set_ticklabels(list(map(str,barlim)))
cbar.ax.tick_params(axis='x', size=.01,labelsize=5)
cbar.outline.set_edgecolor('dimgrey')

xlabels = [r'2',r'3',r'5',r'7']
ylabels = [r'7',r'5',r'3',r'2']
plt.yticks(np.arange(0.5,4.5,1),ylabels,ha='center',color='darkgrey',size=9)
plt.xticks(np.arange(0.5,4.5,1),xlabels,ha='center',color='darkgrey',size=9)
plt.xlim([0,4])
plt.ylim([0,4])

plt.savefig(directoryfigure + 'CSt_%s.png' % saveData,dpi=300)

###############################################################################
###############################################################################
###############################################################################
### Plot correlation matrix
fig = plt.figure()
ax = plt.subplot(111)
var = allcsp

ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')
ax.spines['bottom'].set_color('none')
ax.spines['left'].set_color('none')
ax.get_xaxis().set_tick_params(direction='out', width=2,length=3,
            color='darkgrey')

plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom='on')
plt.tick_params(
    axis='y',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    left=False,      # ticks along the bottom edge are off
    right=False,         # ticks along the top edge are off
    labelleft='on')

csm=plt.get_cmap(cmocean.cm.balance)
norm = c.BoundaryNorm(np.arange(-1,1.1,0.1),csm.N)
cs = plt.pcolormesh(var,shading='faceted',edgecolor='dimgrey',linewidth=1.2,
                    vmin=-1,vmax=1,norm=norm,cmap=csm)

for i in range(var.shape[0]):
    for j in range(var.shape[1]):
        if np.isnan(var[i,j]):
            plt.text(j+0.5,i+0.5,r'\textbf{%s}' % np.round(var[i,j],2),fontsize=10,
                color='w',va='center',ha='center')
        else:
            plt.text(j+0.5,i+0.5,r'\textbf{%s}' % np.round(var[i,j],2),fontsize=10,
                color='k',va='center',ha='center')
                 
cbar = plt.colorbar(cs,orientation='horizontal',aspect=50,pad=0.12)
cbar.set_label(r'\textbf{Correlation Coefficient}',
               color='dimgrey',labelpad=7,fontsize=13)
barlim = np.round(np.arange(-1,1.1,0.25),2)
cbar.set_ticks(barlim)
cbar.set_ticklabels(list(map(str,barlim)))
cbar.ax.tick_params(axis='x', size=.01,labelsize=5)
cbar.outline.set_edgecolor('dimgrey')

xlabels = [r'2',r'3',r'5',r'7']
ylabels = [r'7',r'5',r'3',r'2']
plt.yticks(np.arange(0.5,4.5,1),ylabels,ha='center',color='darkgrey',size=9)
plt.xticks(np.arange(0.5,4.5,1),xlabels,ha='center',color='darkgrey',size=9)
plt.xlim([0,4])
plt.ylim([0,4])

plt.savefig(directoryfigure + 'CSp_%s.png' % saveData,dpi=300)

###############################################################################
###############################################################################
###############################################################################
### Plot correlation matrix
fig = plt.figure()
ax = plt.subplot(111)
var = alllenst

ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')
ax.spines['bottom'].set_color('none')
ax.spines['left'].set_color('none')
ax.get_xaxis().set_tick_params(direction='out', width=2,length=3,
            color='darkgrey')

plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom='on')
plt.tick_params(
    axis='y',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    left=False,      # ticks along the bottom edge are off
    right=False,         # ticks along the top edge are off
    labelleft='on')

csm=plt.get_cmap(cmocean.cm.balance)
norm = c.BoundaryNorm(np.arange(-1,1.1,0.1),csm.N)
cs = plt.pcolormesh(var,shading='faceted',edgecolor='dimgrey',linewidth=1.2,
                    vmin=-1,vmax=1,norm=norm,cmap=csm)

for i in range(var.shape[0]):
    for j in range(var.shape[1]):
        if np.isnan(var[i,j]):
            plt.text(j+0.5,i+0.5,r'\textbf{%s}' % np.round(var[i,j],2),fontsize=10,
                color='w',va='center',ha='center')
        else:
            plt.text(j+0.5,i+0.5,r'\textbf{%s}' % np.round(var[i,j],2),fontsize=10,
                color='k',va='center',ha='center')
                 
cbar = plt.colorbar(cs,orientation='horizontal',aspect=50,pad=0.12)
cbar.set_label(r'\textbf{Correlation Coefficient}',
               color='dimgrey',labelpad=7,fontsize=13)
barlim = np.round(np.arange(-1,1.1,0.25),2)
cbar.set_ticks(barlim)
cbar.set_ticklabels(list(map(str,barlim)))
cbar.ax.tick_params(axis='x', size=.01,labelsize=5)
cbar.outline.set_edgecolor('dimgrey')

xlabels = [r'2',r'3',r'5',r'7']
ylabels = [r'7',r'5',r'3',r'2']
plt.yticks(np.arange(0.5,4.5,1),ylabels,ha='center',color='darkgrey',size=9)
plt.xticks(np.arange(0.5,4.5,1),xlabels,ha='center',color='darkgrey',size=9)
plt.xlim([0,4])
plt.ylim([0,4])

plt.savefig(directoryfigure + 'lenst_%s.png' % saveData,dpi=300)

###############################################################################
###############################################################################
###############################################################################
### Plot correlation matrix
fig = plt.figure()
ax = plt.subplot(111)
var = alllensp

ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')
ax.spines['bottom'].set_color('none')
ax.spines['left'].set_color('none')
ax.get_xaxis().set_tick_params(direction='out', width=2,length=3,
            color='darkgrey')

plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom='on')
plt.tick_params(
    axis='y',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    left=False,      # ticks along the bottom edge are off
    right=False,         # ticks along the top edge are off
    labelleft='on')

csm=plt.get_cmap(cmocean.cm.balance)
norm = c.BoundaryNorm(np.arange(-1,1.1,0.1),csm.N)
cs = plt.pcolormesh(var,shading='faceted',edgecolor='dimgrey',linewidth=1.2,
                    vmin=-1,vmax=1,norm=norm,cmap=csm)

for i in range(var.shape[0]):
    for j in range(var.shape[1]):
        if np.isnan(var[i,j]):
            plt.text(j+0.5,i+0.5,r'\textbf{%s}' % np.round(var[i,j],2),fontsize=10,
                color='w',va='center',ha='center')
        else:
            plt.text(j+0.5,i+0.5,r'\textbf{%s}' % np.round(var[i,j],2),fontsize=10,
                color='k',va='center',ha='center')
                 
cbar = plt.colorbar(cs,orientation='horizontal',aspect=50,pad=0.12)
cbar.set_label(r'\textbf{Correlation Coefficient}',
               color='dimgrey',labelpad=7,fontsize=13)
barlim = np.round(np.arange(-1,1.1,0.25),2)
cbar.set_ticks(barlim)
cbar.set_ticklabels(list(map(str,barlim)))
cbar.ax.tick_params(axis='x', size=.01,labelsize=5)
cbar.outline.set_edgecolor('dimgrey')

xlabels = [r'2',r'3',r'5',r'7']
ylabels = [r'7',r'5',r'3',r'2']
plt.yticks(np.arange(0.5,4.5,1),ylabels,ha='center',color='darkgrey',size=9)
plt.xticks(np.arange(0.5,4.5,1),xlabels,ha='center',color='darkgrey',size=9)
plt.xlim([0,4])
plt.ylim([0,4])

plt.savefig(directoryfigure + 'lenspt_%s.png' % saveData,dpi=300)