"""
Plot comparison of different explainability methods for the climate model 
results of the ANN

Author     : Zachary M. Labe
Date       : 7 March 2022
Version    : 7 - adds validation data for early stopping
"""

### Import packages
import matplotlib.pyplot as plt
import numpy as np
import sys
from netCDF4 import Dataset
from mpl_toolkits.basemap import Basemap, addcyclic, shiftgrid
import palettable.cubehelix as cm
import palettable.cartocolors.qualitative as cc
import scipy.stats as sts

### Plotting defaults 
plt.rc('text',usetex=True)
plt.rc('font',**{'family':'sans-serif','sans-serif':['Avant Garde']}) 

variablesall = ['T2M','P','SLP']
variablesall = ['T2M']
pickSMILEall = [[]] 
fileLRP = ['LRPMap','LRPMap_E','LRPMap_IG']
fileLRPnames = [r'\textbf{LRP$_{z}$ Rule}',r'\textbf{LRP$_{\epsilon}$ Rule}',r'\textbf{Integrated Gradients}']
allDataLabels = ['CanESM2','MPI','CSIRO-MK3.6','EC-EARTH','GFDL-CM3','GFDL-ESM2M','LENS']
AA = 'none'
lrptestalltypes = []
for va in range(len(variablesall)):
    for m in range(len(pickSMILEall)):
        for lr in range(len(fileLRP)):
            ###############################################################################
            ###############################################################################
            ###############################################################################
            ### Data preliminaries 
            directorydata = '/Users/zlabe/Documents/Research/ModelComparison/Data/'
            directoryfigure = '/Users/zlabe/Desktop/ModelComparison_v1/v7/'
            letters = ["a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v"]
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
            elif ocean_only == True:
                saveData = timeper + '_' + seasons[0] + '_OCEAN' + '_NoiseTwinSingleMODDIF4_' + typeOfAnalysis + '_' + variq + '_' + reg_name + '_' + dataset_obs + '_' + 'NumOfSMILE-' + str(num_of_class) + '_Method-' + ensTypeExperi
            else:
                saveData = timeper + '_' + seasons[0] + '_NoiseTwinSingleMODDIF4_' + typeOfAnalysis + '_' + variq + '_' + reg_name + '_' + dataset_obs + '_' + 'NumOfSMILE-' + str(num_of_class) + '_Method-' + ensTypeExperi
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
            def readData(fileLRP,directory,typemodel,saveData):
                """
                Read in LRP maps
                """
                
                name = fileLRP + typemodel + '_' + saveData + '.nc'
                filename = directory + name
                data = Dataset(filename)
                lat = data.variables['lat'][:]
                lon = data.variables['lon'][:]
                lrp = data.variables['LRP'][:]
                data.close()
                
                return lrp,lat,lon
            
            ### Read in training and testing predictions and labels
            classesltest = np.int_(np.genfromtxt(directorydata + 'testingTrueLabels_' + saveData + '.txt'))
            predtest = np.int_(np.genfromtxt(directorydata + 'testingPredictedLabels_' + saveData + '.txt'))
            
            ### Count testing data
            uniquetest,counttest = np.unique(predtest,return_counts=True)
            
            ### Read in observational data
            obspred = np.int_(np.genfromtxt(directorydata + 'obsLabels_' + saveData + '.txt'))
            uniqueobs,countobs = np.unique(obspred,return_counts=True)
            percPickObs = np.nanmax(countobs)/len(obspred)*100.
            modelPickObs = modelGCMsNames[uniqueobs[np.argmax(countobs)]]
            
            ### Read in LRP maps
            lrptestdata,lat1,lon1 = readData(fileLRP[lr],directorydata,'Testing',saveData)
            
            ### Meshgrid
            lon2,lat2 = np.meshgrid(lon1,lat1)
            
            ###############################################################################
            ###############################################################################
            ############################################################################### 
            ### Find which model
            print('\nPrinting *length* of predicted labels for testing!')
                
            model_test = []
            for i in range(lenOfPicks):
                modelloc = np.where((predtest == int(i)))[0]
                print(len(modelloc))
                model_test.append(modelloc)
            
            if AA == True:
                lrptest = []
                for i in range(lenOfPicks):
                    lrpmodel = lrptestdata[model_test[i]]
                    lrpmodelmeanslice = lrpmodel.reshape(lrpmodel.shape[0]//yearsall.shape[0],yearsall.shape[0],lrpmodel.shape[1],lrpmodel.shape[2])
                    lrpmodelmean1 = np.nanmean(lrpmodelmeanslice[:,-15:,:,:],axis=0)
                    lrpmodelmean = np.nanmean(lrpmodelmean1,axis=0)
                    lrptest.append(lrpmodelmean)
                lrptest = np.asarray(lrptest,dtype=object)
            elif AA == False:
                lrptest = []
                for i in range(lenOfPicks):
                    lrpmodel = lrptestdata[model_test[i]]
                    lrpmodelmeanslice = lrpmodel.reshape(lrpmodel.shape[0]//yearsall.shape[0],yearsall.shape[0],lrpmodel.shape[1],lrpmodel.shape[2])
                    lrpmodelmean1 = np.nanmean(lrpmodelmeanslice[:,:-15,:,:],axis=0)
                    lrpmodelmean = np.nanmean(lrpmodelmean1,axis=0)
                    lrptest.append(lrpmodelmean)
                lrptest = np.asarray(lrptest,dtype=object)
            elif AA == 'none':
                lrptest = []
                for i in range(lenOfPicks):
                    lrpmodel = lrptestdata[model_test[i]]
                    lrpmodelmean = np.nanmean(lrpmodel,axis=0)
                    lrptest.append(lrpmodelmean)
                lrptest = np.asarray(lrptest,dtype=object)          
                
            lrptestalltypes.append(lrptest)
            
### Look at comparison of all LRP methods
lrptestalltypes = np.asarray(lrptestalltypes)
plotlrp = lrptestalltypes.reshape(lrptestalltypes.shape[0]*lrptestalltypes.shape[1],
                                              lrptestalltypes.shape[2],lrptestalltypes.shape[3])

###############################################################################
###############################################################################
###############################################################################
### Plot subplot of LRP means training
limit = np.arange(0,0.80001,0.005)
barlim = np.round(np.arange(0,0.801,0.8),2)
cmap = cm.cubehelix2_16.mpl_colormap
label = r'\textbf{RELEVANCE}'

fig = plt.figure(figsize=(10,4))
for r in range(plotlrp.shape[0]):
    
    var = plotlrp[r]/np.nanmax(plotlrp[r])
    
    ax1 = plt.subplot(3,7,r+1)
    m = Basemap(projection='npstere',boundinglat=61.5,lon_0=0,
                resolution='l',round =True,area_thresh=10000)
    m.drawcoastlines(color='darkgrey',linewidth=0.3)
        
    var, lons_cyclic = addcyclic(var, lon1)
    var, lons_cyclic = shiftgrid(180., var, lons_cyclic, start=False)
    lon2d, lat2d = np.meshgrid(lons_cyclic, lat1)
    x, y = m(lon2d, lat2d)
       
    circle = m.drawmapboundary(fill_color='dimgrey',color='dimgray',
                      linewidth=0.7)
    circle.set_clip_on(False)
    
    cs1 = m.contourf(x,y,var,limit,extend='max')
    cs1.set_cmap(cmap) 
         
    if r < 7:
        ax1.annotate(r'\textbf{%s}' % allDataLabels[r],xy=(0,0),xytext=(0.5,1.13),
                      textcoords='axes fraction',color='dimgrey',fontsize=8,
                      rotation=0,ha='center',va='center')
    ax1.annotate(r'\textbf{[%s]}' % letters[r],xy=(0,0),xytext=(0.86,0.97),
                  textcoords='axes fraction',color='k',fontsize=6,
                  rotation=330,ha='center',va='center')
    

    ### Make colorbar
    cbar_ax = fig.add_axes([0.91,0.35,0.011,0.3])                
    cbar = fig.colorbar(cs1,cax=cbar_ax,orientation='vertical',
                        extend='max',extendfrac=0.07,drawedges=False)    
    cbar.set_label(label,fontsize=7,color='k',labelpad=6.5)      
    cbar.set_ticks(barlim)
    cbar.set_ticklabels(list(map(str,barlim)))
    cbar.ax.tick_params(labelsize=4,pad=7) 
    ticklabs = cbar.ax.get_yticklabels()
    cbar.ax.set_yticklabels(ticklabs,ha='center')
    cbar.ax.tick_params(axis='y', size=.001)
    cbar.outline.set_edgecolor('dimgrey')
    cbar.outline.set_linewidth(0.5)
        
ax1.annotate(fileLRPnames[2],xy=(0,0),xytext=(-6.9,0.5),
              textcoords='axes fraction',color='k',fontsize=8,
              rotation=90,ha='center',va='center')    
ax1.annotate(fileLRPnames[1],xy=(0,0),xytext=(-6.9,1.53),
              textcoords='axes fraction',color='k',fontsize=8,
              rotation=90,ha='center',va='center')
ax1.annotate(fileLRPnames[0],xy=(0,0),xytext=(-6.9,2.54),
              textcoords='axes fraction',color='k',fontsize=8,
              rotation=90,ha='center',va='center')

plt.tight_layout()
plt.subplots_adjust(hspace=0.01,wspace=0.02)
plt.savefig(directoryfigure + 'ExplainabilityMethods_SUPPLEMENT.png',dpi=900)
        