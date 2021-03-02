"""
ANN for evaluating model biases, differences, and other thresholds using 
explainable AI

Reference  : Barnes et al. [2020, JAMES]
Author     : Zachary M. Labe
Date       : 1 March 2021
Version    : 1 
"""

### Import packages
import matplotlib.pyplot as plt
import numpy as np
import sys
from netCDF4 import Dataset
import scipy.stats as stats
from mpl_toolkits.basemap import Basemap, addcyclic, shiftgrid
import palettable.cubehelix as cm
import palettable.scientific.sequential as ssss
import palettable.scientific.diverging as dddd
import cmocean as cmocean
import calc_Utilities as UT
import calc_dataFunctions as df
import calc_Stats as dSS
from sklearn.metrics import accuracy_score

### Plotting defaults 
plt.rc('text',usetex=True)
plt.rc('font',**{'family':'sans-serif','sans-serif':['Avant Garde']}) 

###############################################################################
###############################################################################
###############################################################################
### Data preliminaries 
directorydata = '/Users/zlabe/Documents/Research/ModelComparison/Data/'
directoryfigure = '/Users/zlabe/Desktop/ModelComparison/ClimateChange/'
letters = ["a","b","c","d","e","f","g","h","i","j","k","l","m"]
###############################################################################
###############################################################################
modelGCMs = ['CCCma_canesm2','MPI','CSIRO_MK3.6','KNMI_ecearth',
              'GFDL_CM3','GFDL_ESM2M','lens']
datasetsingle = ['SMILE']
dataset_obs = 'ERA5BE'
seasons = ['annual']
variq = 'P'
reg_name = 'SMILEGlobe'
###############################################################################
###############################################################################
pickSMILE = [] # create empty list for ALL 7 GCMs
# pickSMILE = ['MPI','lens']
# pickSMILE = ['CSIRO_MK3.6','GFDL_CM3','lens']
# pickSMILE = ['CCCma_canesm2','CSIRO_MK3.6','GFDL_CM3','GFDL_ESM2M','lens'] 
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
classesltest = np.int_(np.genfromtxt(directorydata + 'testingTrueLabels_' + saveData + '.txt'))

predtrain = np.int_(np.genfromtxt(directorydata + 'trainingPredictedLabels_' + saveData + '.txt'))
predtest = np.int_(np.genfromtxt(directorydata + 'testingPredictedLabels_' + saveData + '.txt'))

### Read in observational data
obspred = np.int_(np.genfromtxt(directorydata + 'obsLabels_' + saveData + '.txt'))
uniqueobs,countobs = np.unique(obspred,return_counts=True)

### Read in LRP maps
lrptraindata,lat1,lon1 = readData(directorydata,'Training',saveData)
lrptestdata,lat1,lon1 = readData(directorydata,'Testing',saveData)
lrpobsdata,lat1,lon1 = readData(directorydata,'Obs',saveData)

### Find which model
print('\nPrinting *length* of predicted labels for training and testing!')
model_train = []
for i in range(lenOfPicks):
    modelloc = np.where((predtrain == int(i)))[0]
    print(len(modelloc))
    model_train.append(modelloc)
    
model_test = []
for i in range(lenOfPicks):
    modelloc = np.where((predtest == int(i)))[0]
    print(len(modelloc))
    model_test.append(modelloc)

### Composite lrp maps of the models
lrptrain = []
for i in range(lenOfPicks):
    lrpmodel = lrptraindata[model_train[i]]
    lrpmodelmean = np.nanmean(lrpmodel,axis=0)
    lrptrain.append(lrpmodelmean)
lrptrain = np.asarray(lrptrain,dtype=object)

lrptest = []
for i in range(lenOfPicks):
    lrpmodel = lrptestdata[model_test[i]]
    lrpmodelmean = np.nanmean(lrpmodel,axis=0)
    lrptest.append(lrpmodelmean)
lrptest = np.asarray(lrptest,dtype=object)

#######################################################################
#######################################################################
#######################################################################
### Plot subplot of LRP means training
plt.rc('text',usetex=True)
plt.rc('font',**{'family':'sans-serif','sans-serif':['Avant Garde']}) 

limit = np.arange(0,0.4001,0.005)
barlim = np.round(np.arange(0,0.5,0.1),2)
cmap = cm.cubehelix2_16.mpl_colormap
label = r'\textbf{RELEVANCE}'

# fig = plt.figure(figsize=(10,2))
# for r in range(len(modelGCMs)):
#     var = lrptrain[r]
    
#     ax1 = plt.subplot(1,7,r+1)
#     m = Basemap(projection='moll',lon_0=0,resolution='l',area_thresh=10000)
#     m.drawcoastlines(color='darkgrey',linewidth=0.27)
        
#     var, lons_cyclic = addcyclic(var, lon1)
#     var, lons_cyclic = shiftgrid(180., var, lons_cyclic, start=False)
#     lon2d, lat2d = np.meshgrid(lons_cyclic, lat1)
#     x, y = m(lon2d, lat2d)
       
#     circle = m.drawmapboundary(fill_color='dimgrey',color='dimgray',
#                       linewidth=0.7)
#     circle.set_clip_on(False)
    
#     cs1 = m.contourf(x,y,var,limit,extend='max')
#     cs1.set_cmap(cmap) 
            
#     ax1.annotate(r'\textbf{%s}' % modelGCMs[r],xy=(0,0),xytext=(0.5,1.10),
#                   textcoords='axes fraction',color='dimgrey',fontsize=8,
#                   rotation=0,ha='center',va='center')
#     ax1.annotate(r'\textbf{[%s]}' % letters[r],xy=(0,0),xytext=(0.86,0.97),
#                   textcoords='axes fraction',color='k',fontsize=6,
#                   rotation=330,ha='center',va='center')
    
# ###############################################################################
# cbar_ax1 = fig.add_axes([0.36,0.15,0.3,0.03])                
# cbar1 = fig.colorbar(cs1,cax=cbar_ax1,orientation='horizontal',
#                     extend='max',extendfrac=0.07,drawedges=False)
# cbar1.set_label(label,fontsize=9,color='dimgrey',labelpad=1.4)  
# cbar1.set_ticks(barlim)
# cbar1.set_ticklabels(list(map(str,barlim)))
# cbar1.ax.tick_params(axis='x', size=.01,labelsize=5)
# cbar1.outline.set_edgecolor('dimgrey')

# plt.tight_layout()
# plt.subplots_adjust(top=0.85,wspace=0.02,hspace=0.02,bottom=0.14)

# plt.savefig(directoryfigure + '.png',dpi=300)