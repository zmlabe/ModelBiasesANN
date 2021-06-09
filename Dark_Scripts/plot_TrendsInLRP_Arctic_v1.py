"""
ANN for evaluating model biases, differences, and other thresholds using 
explainable AI for historical data for regional data

Author     : Zachary M. Labe
Date       : 7 June 2021
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
import cmocean
from sklearn.metrics import accuracy_score
import scipy.stats as sts

### Plotting defaults 
plt.rc('text',usetex=True)
plt.rc('font',**{'family':'sans-serif','sans-serif':['Avant Garde']}) 

variablesall = ['T2M','P','SLP']
variablesall = ['T2M']
pickSMILEall = [[]] 
for va in range(len(variablesall)):
    for m in range(len(pickSMILEall)):
        ###############################################################################
        ###############################################################################
        ###############################################################################
        ### Data preliminaries 
        directorydata = '/Users/zlabe/Documents/Research/ModelComparison/Data/'
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
        reg_name = 'LowerArctic'
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
        
        ### Count testing data
        uniquetest,counttest = np.unique(predtest,return_counts=True)
        
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
        lrptestdata,lat1,lon1 = readData(directorydata,'Testing',saveData)
        lrpobsdata,lat1,lon1 = readData(directorydata,'Obs',saveData)
        
        ### Meshgrid
        lon2,lat2 = np.meshgrid(lon1,lat1)
        
        ###############################################################################
        ###############################################################################
        ############################################################################### 
        ### Find which model
        print('\nPrinting *length* of predicted labels for training and testing!')
        model_test = []
        for i in range(lenOfPicks):
            modelloc = np.where((predtest == int(i)))[0]
            print(len(modelloc))
            model_test.append(modelloc)

        lrptest = []
        for i in range(lenOfPicks):
            lrpmodel = lrptestdata[model_test[i]]
            lrpmodelmean = lrpmodel.reshape(4,lrpmodel.shape[0]//4,lat1.shape[0],lon1.shape[0])
            lrptest.append(np.nanmean(lrpmodelmean[:,:,:,:],axis=0)) # mean over testing ensembles
        lrptest = np.asarray(lrptest,dtype=object)            
        
        ###############################################################################
        ###############################################################################
        ############################################################################### 
        ### Calculate trends
        slope = np.empty((lrptest.shape[0],lrptest.shape[2],lrptest.shape[3]))
        intercept = np.empty((lrptest.shape[0],lrptest.shape[2],lrptest.shape[3]))
        rr = np.empty((lrptest.shape[0],lrptest.shape[2],lrptest.shape[3]))
        p = np.empty((lrptest.shape[0],lrptest.shape[2],lrptest.shape[3]))
        se = np.empty((lrptest.shape[0],lrptest.shape[2],lrptest.shape[3]))
        for m in range(lrptest.shape[0]):
            for i in range(lrptest.shape[2]):
                for j in range(lrptest.shape[3]):
                    xx = np.arange((lrptest.shape[1]))
                    yy = np.asarray(lrptest[m,:,i,j],dtype=np.float)
                    slope[m,i,j],intercept[m,i,j],rr[m,i,j],p[m,i,j],se[m,i,j] = sts.linregress(xx,yy)
        
        ### Calculate decadal trend in relevance
        slopedec = slope * 10.
        
        ###############################################################################
        ###############################################################################
        ###############################################################################
        ### Plot subplot of LRP means training
        limit = np.arange(-0.08,0.081,0.002)
        barlim = np.round(np.arange(-0.08,0.081,0.05),2)
        cmap = cmocean.cm.balance
        label = r'\textbf{Relevance Trend - [ %s ] - %s}' % (variq,typeOfAnalysis)
        
        fig = plt.figure(figsize=(10,2))
        for r in range(lenOfPicks):
            var = slopedec[r]
            
            ax1 = plt.subplot(1,lenOfPicks,r+1)
            if reg_name == 'LowerArctic':
                m = Basemap(projection='npstere',boundinglat=61.5,lon_0=0,
                            resolution='l',round =True,area_thresh=10000)
            else:
                m = Basemap(projection='npstere',boundinglat=71,lon_0=0,
                            resolution='l',round =True,area_thresh=10000)
            m.drawcoastlines(color='darkgrey',linewidth=0.27)
                
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
                    
            ax1.annotate(r'\textbf{%s}' % modelGCMsNames[r],xy=(0,0),xytext=(0.5,1.10),
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
        
        plt.savefig(directoryfigure + '%s/LRPTrends_%s.png' % (typeOfAnalysis,saveData),dpi=300)