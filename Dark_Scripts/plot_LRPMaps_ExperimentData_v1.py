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
from mpl_toolkits.basemap import Basemap, addcyclic, shiftgrid
import palettable.cubehelix as cm
import palettable.cartocolors.qualitative as cc
from sklearn.metrics import accuracy_score

### Plotting defaults 
plt.rc('text',usetex=True)
plt.rc('font',**{'family':'sans-serif','sans-serif':['Avant Garde']}) 

variablesall = ['T2M','P','SLP']
variablesall = ['P']
pickSMILEall = [
                [],
                ['CSIRO-MK3.6','lens'],
                ['CSIRO-MK3.6','GFDL-CM3','LENS'],
                ['CanESM2','CSIRO-MK3.6','GFDL-CM3','GFDL-ESM2M','LENS']
                ] 
pickSMILEall = [[]]
for va in range(len(variablesall)):
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
        variq = variablesall[va]
        reg_name = 'SMILEGlobe'
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
        rm_annual_mean = True
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
                                
        print('\n<<<<<<<<<<<< Analysis == %s (%s) ! >>>>>>>>>>>>>>>\n' % (typeOfAnalysis,timeper))
        if typeOfAnalysis == 'issueWithExperiment':
            sys.exit('Wrong parameters selected to analyze')
            
        ### Select how to save files
        saveData = timeper + '_' + typeOfAnalysis + '_' + variq + '_' + reg_name + '_' + dataset_obs + '_' + 'NumOfSMILE-' + str(num_of_class) + '_Method-' + ensTypeExperi
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
        
        ###############################################################################
        ###############################################################################
        ############################################################################### 
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
        
        ### Composite observations
        obsloc = np.where((obspred == uniqueobs[np.argmax(countobs)]))[0]
        obslocNOT = np.where((obspred != uniqueobs[np.argmax(countobs)]))[0]
        lrp_maxModelObs = np.nanmean(lrpobsdata[obsloc],axis=0)
        lrp_maxModelObsNOT = np.nanmean(lrpobsdata[obslocNOT],axis=0)
        meanlrpobs = np.nanmean(lrpobsdata,axis=0)
        
        ###############################################################################
        ###############################################################################
        ###############################################################################
        ### Calculate Accuracy 
        def accuracyTotalTime(data_pred,data_true):
            """
            Compute accuracy for the entire time series
            """
            
            data_truer = data_true
            data_predr = data_pred
            accdata_pred = accuracy_score(data_truer,data_predr)*100 # 0-100%
                
            return accdata_pred
        
        acctrain = accuracyTotalTime(predtrain,classesltrain)
        acctest = accuracyTotalTime(predtest,classesltest)
        print('\n\nAccuracy Training == %s%%' % acctrain)
        print('Accuracy Testing == %s%%' % acctest)
        
        ###############################################################################
        ###############################################################################
        ###############################################################################
        ### Plot subplot of LRP means training
        if typeOfAnalysis == 'Experiment-4':
            limit = np.arange(0,0.50001,0.005)
            barlim = np.round(np.arange(0,0.501,0.1),2)
        elif typeOfAnalysis == 'Experiment-3':
            limit = np.arange(0,1.00001,0.005)
            barlim = np.round(np.arange(0,1.01,0.1),2)
        elif typeOfAnalysis == 'Experiment-7':
            limit = np.arange(0,0.40001,0.005)
            barlim = np.round(np.arange(0,0.401,0.1),2)
        elif typeOfAnalysis == 'Experiment-8':
            limit = np.arange(0,0.40001,0.005)
            barlim = np.round(np.arange(0,0.401,0.1),2)
        elif typeOfAnalysis == 'Experiment-9':
            limit = np.arange(0,0.40001,0.005)
            barlim = np.round(np.arange(0,0.401,0.1),2)
        cmap = cm.cubehelix2_16.mpl_colormap
        label = r'\textbf{Relevance - [ %s ] - %s}' % (variq,typeOfAnalysis)
        # label = r'\textbf{Precipitation-LRP [Relevance]}'
        
        fig = plt.figure(figsize=(10,2))
        for r in range(lenOfPicks):
            var = lrptest[r]
            
            ax1 = plt.subplot(1,lenOfPicks,r+1)
            m = Basemap(projection='moll',lon_0=0,resolution='l',area_thresh=10000)
            m.drawcoastlines(color='darkgrey',linewidth=0.27)
                
            var, lons_cyclic = addcyclic(var, lon1)
            var, lons_cyclic = shiftgrid(180., var, lons_cyclic, start=False)
            lon2d, lat2d = np.meshgrid(lons_cyclic, lat1)
            x, y = m(lon2d, lat2d)
               
            circle = m.drawmapboundary(fill_color='dimgrey',color='dimgray',
                              linewidth=0.7)
            circle.set_clip_on(False)
            
            cs1 = m.contourf(x,y,var,limit,extend='max')
            cs1.set_cmap(cmap) 
                    
            ax1.annotate(r'\textbf{%s}' % modelGCMsNames[r],xy=(0,0),xytext=(0.5,1.10),
                          textcoords='axes fraction',color='dimgrey',fontsize=8,
                          rotation=0,ha='center',va='center')
            ax1.annotate(r'\textbf{[%s]}' % letters[r],xy=(0,0),xytext=(0.86,0.97),
                          textcoords='axes fraction',color='k',fontsize=6,
                          rotation=330,ha='center',va='center')
            
        ###############################################################################
        cbar_ax1 = fig.add_axes([0.36,0.15,0.3,0.03])                
        cbar1 = fig.colorbar(cs1,cax=cbar_ax1,orientation='horizontal',
                            extend='max',extendfrac=0.07,drawedges=False)
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
         
        if typeOfAnalysis == 'Experiment-4':
            plt.text(-0.62,-0.2,r'\textbf{TRAINING ACCURACY = %s \%%}' % np.round(acctrain,1),color='k',
                  fontsize=7)
            plt.text(-0.62,-1.3,r'\textbf{TESTING ACCURACY = %s \%%}' % np.round(acctest,1),color='k',
                      fontsize=7)
        else:
            plt.text(-0.3,0.00,r'\textbf{TRAINING ACCURACY = %s \%%}' % np.round(acctrain,1),color='k',
                  fontsize=7)
            plt.text(-0.3,-1,r'\textbf{TESTING ACCURACY = %s \%%}' % np.round(acctest,1),color='k',
                      fontsize=7)
        
        plt.savefig(directoryfigure + '%s/LRPComposites_%s.png' % (typeOfAnalysis,saveData),dpi=300)
        
        ###############################################################################
        ###############################################################################
        ###############################################################################
        ### Plot subplot of observations
        limit = np.arange(0,0.31,0.005)
        barlim = np.round(np.arange(0,0.31,0.1),2)
        cmap = cm.cubehelix2_16.mpl_colormap
        label = r'\textbf{Relevance - [ %s - OBS] - %s}' % (variq,typeOfAnalysis)
        
        fig = plt.figure(figsize=(6,8))
        ###############################################################################
        ax1 = plt.subplot(311)
        m = Basemap(projection='moll',lon_0=0,resolution='l',area_thresh=10000)
        m.drawcoastlines(color='darkgrey',linewidth=0.27)
            
        var, lons_cyclic = addcyclic(meanlrpobs, lon1)
        var, lons_cyclic = shiftgrid(180., var, lons_cyclic, start=False)
        lon2d, lat2d = np.meshgrid(lons_cyclic, lat1)
        x, y = m(lon2d, lat2d)
           
        circle = m.drawmapboundary(fill_color='dimgrey',color='dimgray',
                          linewidth=0.7)
        circle.set_clip_on(False)
        
        cs1 = m.contourf(x,y,var,limit,extend='max')
        cs1.set_cmap(cmap) 
        
        ax1.annotate(r'\textbf{1950--2019}' ,xy=(0,0),xytext=(0.01,0.90),
                      textcoords='axes fraction',color='k',fontsize=11,
                      rotation=0,ha='center',va='center')        
        ax1.annotate(r'\textbf{[%s]}' % letters[0],xy=(0,0),xytext=(0.98,0.90),
                      textcoords='axes fraction',color='k',fontsize=11,
                      rotation=0,ha='center',va='center')
        
        ###############################################################################
        
        ax2 = plt.subplot(312)
        m = Basemap(projection='moll',lon_0=0,resolution='l',area_thresh=10000)
        m.drawcoastlines(color='darkgrey',linewidth=0.27)
            
        var, lons_cyclic = addcyclic(lrp_maxModelObs, lon1)
        var, lons_cyclic = shiftgrid(180., var, lons_cyclic, start=False)
        lon2d, lat2d = np.meshgrid(lons_cyclic, lat1)
        x, y = m(lon2d, lat2d)
           
        circle = m.drawmapboundary(fill_color='dimgrey',color='dimgray',
                          linewidth=0.7)
        circle.set_clip_on(False)
        
        cs2 = m.contourf(x,y,var,limit,extend='max')
        cs2.set_cmap(cmap) 
        
        ax2.annotate(r'\textbf{%s}' % modelPickObs,xy=(0,0),xytext=(0.01,0.90),
                      textcoords='axes fraction',color='k',fontsize=11,
                      rotation=0,ha='center',va='center')  
        ax2.annotate(r'\textbf{%s\%%}' % np.round(percPickObs,1),xy=(0,0),xytext=(0.01,0.83),
                      textcoords='axes fraction',color='k',fontsize=11,
                      rotation=0,ha='center',va='center')           
        ax2.annotate(r'\textbf{[%s]}' % letters[1],xy=(0,0),xytext=(0.98,0.90),
                      textcoords='axes fraction',color='k',fontsize=11,
                      rotation=0,ha='center',va='center')
        
        ###############################################################################
        
        ax3 = plt.subplot(313)
        m = Basemap(projection='moll',lon_0=0,resolution='l',area_thresh=10000)
        m.drawcoastlines(color='darkgrey',linewidth=0.27)
            
        var, lons_cyclic = addcyclic(lrp_maxModelObsNOT, lon1)
        var, lons_cyclic = shiftgrid(180., var, lons_cyclic, start=False)
        lon2d, lat2d = np.meshgrid(lons_cyclic, lat1)
        x, y = m(lon2d, lat2d)
           
        circle = m.drawmapboundary(fill_color='dimgrey',color='dimgray',
                          linewidth=0.7)
        circle.set_clip_on(False)
        
        cs3 = m.contourf(x,y,var,limit,extend='max')
        cs3.set_cmap(cmap) 
        
        ax3.annotate(r'\textbf{OTHER}',xy=(0,0),xytext=(0.01,0.90),
                      textcoords='axes fraction',color='k',fontsize=11,
                      rotation=0,ha='center',va='center')  
        ax3.annotate(r'\textbf{%s\%%}' % (np.round(100.-percPickObs,1)),xy=(0,0),xytext=(0.01,0.83),
                      textcoords='axes fraction',color='k',fontsize=11,
                      rotation=0,ha='center',va='center')           
        ax3.annotate(r'\textbf{[%s]}' % letters[2],xy=(0,0),xytext=(0.98,0.90),
                      textcoords='axes fraction',color='k',fontsize=11,
                      rotation=0,ha='center',va='center')
                  
        ###############################################################################
        cbar_ax1 = fig.add_axes([0.035,0.03,0.2,0.01])                
        cbar1 = fig.colorbar(cs1,cax=cbar_ax1,orientation='horizontal',
                            extend='max',extendfrac=0.07,drawedges=False)
        cbar1.set_label(label,fontsize=5,color='dimgrey',labelpad=1.4)  
        cbar1.set_ticks(barlim)
        cbar1.set_ticklabels(list(map(str,barlim)))
        cbar1.ax.tick_params(axis='x', size=.01,labelsize=5)
        cbar1.outline.set_edgecolor('dimgrey')
        
        plt.tight_layout()
        plt.savefig(directoryfigure + '%s/ObservationsData_%s.png' % (typeOfAnalysis,saveData),dpi=300)
        
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
                    
        fig = plt.figure(figsize=(10,5))
        
        ax = plt.subplot(121)
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
        plt.scatter(x,obspred,c=obspred,s=40,clip_on=False,cmap=cc.Antique_7.mpl_colormap,
                    edgecolor='k',linewidth=0.4,zorder=10)
        
        plt.xticks(np.arange(1950,2021,5),map(str,np.arange(1950,2021,5)),size=6)
        plt.yticks(np.arange(0,lenOfPicks+1,1),modelGCMsNames,size=6)
        plt.xlim([1950,2020])   
        plt.ylim([0,lenOfPicks-1])
        plt.xlabel(r'\textbf{Predictions - [ %s - OBSERVATIONS ] - %s}' % (variq,typeOfAnalysis))
        
        ###############################################################################
        
        ax = plt.subplot(122)
        adjust_spines(ax, ['left', 'bottom'])
        ax.spines['top'].set_color('none')
        ax.spines['right'].set_color('none')
        ax.spines['left'].set_color('dimgrey')
        ax.spines['bottom'].set_color('dimgrey')
        ax.spines['left'].set_linewidth(2)
        ax.spines['bottom'].set_linewidth(0)
        ax.tick_params('y',length=4,width=2,which='major',color='dimgrey')
        ax.tick_params('x',length=0,width=0,which='major',color='dimgrey')
        
        rects = plt.bar(uniquetest,counttest, align='center')
        plt.axhline(y=len(predtest)/lenOfPicks,linestyle='--',linewidth=2,color='k',
                    clip_on=False,zorder=100,dashes=(1,0.3))
        
        ### Set color
        colorlist = [cc.Antique_7.mpl_colormap(1/7),
                     cc.Antique_7.mpl_colormap(0/7),
                     cc.Antique_7.mpl_colormap(3/7),
                     cc.Antique_7.mpl_colormap(5/7),
                     cc.Antique_7.mpl_colormap(4/7),
                     cc.Antique_7.mpl_colormap(6/7),
                     cc.Antique_7.mpl_colormap(7/7)]
        for i in range(lenOfPicks):
            rects[i].set_color(colorlist[i])
            rects[i].set_edgecolor(colorlist[i])
        
        plt.xticks(np.arange(0,lenOfPicks+1,1),modelGCMsNames,size=6)
        plt.yticks(np.arange(0,520,50),map(str,np.arange(0,520,50)),size=6)
        plt.xlim([-0.5,lenOfPicks-1+0.5])   
        plt.ylim([0,500])
        plt.xlabel(r'\textbf{Frequency - [ %s - TESTING ] - %s}' % (variq,typeOfAnalysis))
        
        plt.tight_layout()
        plt.savefig(directoryfigure + '%s/PredictedModels_%s.png' % (typeOfAnalysis,saveData),dpi=300)
        
        ###############################################################################
        ###############################################################################
        ###############################################################################
        
        fig = plt.figure(figsize=(10,5))
        
        ax = plt.subplot(121)
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
        plt.scatter(x,randpred,c=randpred,s=40,clip_on=False,cmap=cc.Antique_7.mpl_colormap,
                    edgecolor='k',linewidth=0.4,zorder=10)
        
        plt.xticks(np.arange(1950,2021,5),map(str,np.arange(1950,2021,5)),size=6)
        plt.yticks(np.arange(0,lenOfPicks+1,1),modelGCMsNames,size=6)
        plt.xlim([1950,2020])   
        plt.ylim([0,lenOfPicks-1])
        plt.xlabel(r'\textbf{Predictions - [ %s - RANDOM DATA ] - %s}' % (variq,typeOfAnalysis))
        
        ###############################################################################
        
        ax = plt.subplot(122)
        adjust_spines(ax, ['left', 'bottom'])
        ax.spines['top'].set_color('none')
        ax.spines['right'].set_color('none')
        ax.spines['left'].set_color('dimgrey')
        ax.spines['bottom'].set_color('dimgrey')
        ax.spines['left'].set_linewidth(2)
        ax.spines['bottom'].set_linewidth(0)
        ax.tick_params('y',length=4,width=2,which='major',color='dimgrey')
        ax.tick_params('x',length=0,width=0,which='major',color='dimgrey')
        
        rects = plt.bar(uniquetest,counttest, align='center')
        plt.axhline(y=len(predtest)/lenOfPicks,linestyle='--',linewidth=2,color='k',
                    clip_on=False,zorder=100,dashes=(1,0.3))
        
        ### Set color
        colorlist = [cc.Antique_7.mpl_colormap(1/7),
                     cc.Antique_7.mpl_colormap(0/7),
                     cc.Antique_7.mpl_colormap(3/7),
                     cc.Antique_7.mpl_colormap(5/7),
                     cc.Antique_7.mpl_colormap(4/7),
                     cc.Antique_7.mpl_colormap(6/7),
                     cc.Antique_7.mpl_colormap(7/7)]
        for i in range(lenOfPicks):
            rects[i].set_color(colorlist[i])
            rects[i].set_edgecolor(colorlist[i])
        
        plt.xticks(np.arange(0,lenOfPicks+1,1),modelGCMsNames,size=6)
        plt.yticks(np.arange(0,520,50),map(str,np.arange(0,520,50)),size=6)
        plt.xlim([-0.5,lenOfPicks-1+0.5])   
        plt.ylim([0,500])
        plt.xlabel(r'\textbf{Frequency - [ %s - TESTING ] - %s}' % (variq,typeOfAnalysis))
        
        plt.tight_layout()
        plt.savefig(directoryfigure + '%s/RandomModels_%s.png' % (typeOfAnalysis,saveData),dpi=300)