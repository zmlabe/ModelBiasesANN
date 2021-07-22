"""
Script for plotting differences in model and observational climatologies for 
select variables over the 1950 to 2019 period

Author     : Zachary M. Labe
Date       : 14 July 2021
Version    : 1 
"""

### Import packages
import sys
import math
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
from mpl_toolkits.basemap import Basemap, addcyclic, shiftgrid
import palettable.cubehelix as cm
import palettable.scientific.sequential as sss
import cmocean as cmocean
import calc_Utilities as UT
import calc_dataFunctions as df
import calc_Stats as dSS
import scipy.stats as sts
import matplotlib
import cmasher as cmr

### Plotting defaults 
plt.rc('text',usetex=True)
plt.rc('font',**{'family':'sans-serif','sans-serif':['Avant Garde']})

###############################################################################
###############################################################################
###############################################################################
### Data preliminaries 
modelGCMs = ['CanESM2','MPI','CSIRO-MK3.6','KNMI-ecearth','GFDL-CM3','GFDL-ESM2M','LENS','MM-Mean']
dataset_obs = 'ERA5BE'
allDataLabels = [dataset_obs,'CanESM2','MPI','CSIRO-MK3.6','EC-EARTH','GFDL-CM3','GFDL-ESM2M','LENS','MM-Mean']
letters = ["a","b","c","d","e","f","g","h","i","j","k","l","m","n"]
datasetsingle = ['SMILE']
monthlychoiceq = ['annual']
variables = ['T2M']
reg_name = 'LowerArctic'
level = 'surface'
###############################################################################
###############################################################################
randomalso = False
timeper = 'historical'
shuffletype = 'GAUSS'
###############################################################################
###############################################################################
land_only = False
ocean_only = False
###############################################################################
###############################################################################
baseline = np.arange(1951,1980+1,1)
###############################################################################
###############################################################################
window = 0
yearsall = np.arange(1950+window,2019+1,1)
###############################################################################
###############################################################################
numOfEns = 16
lentime = len(yearsall)
###############################################################################
###############################################################################
dataset = datasetsingle[0]
lat_bounds,lon_bounds = UT.regions(reg_name)
###############################################################################
###############################################################################
ravelyearsbinary = False
ravelbinary = False
lensalso = True
###############################################################################
###############################################################################
###############################################################################
###############################################################################
### Read in model and observational/reanalysis data
def read_primary_dataset(variq,dataset,monthlychoice,numOfEns,lensalso,randomalso,ravelyearsbinary,ravelbinary,shuffletype,timeper,lat_bounds=lat_bounds,lon_bounds=lon_bounds):
    data,lats,lons = df.readFiles(variq,dataset,monthlychoice,numOfEns,lensalso,randomalso,ravelyearsbinary,ravelbinary,shuffletype,timeper)
    datar,lats,lons = df.getRegion(data,lats,lons,lat_bounds,lon_bounds)
    print('\nOur dataset: ',dataset,' is shaped',data.shape)
    return datar,lats,lons
  
def read_obs_dataset(variq,dataset_obs,numOfEns,lensalso,randomalso,ravelyearsbinary,ravelbinary,shuffletype,lat_bounds=lat_bounds,lon_bounds=lon_bounds):
    data_obs,lats_obs,lons_obs = df.readFiles(variq,dataset_obs,monthlychoice,numOfEns,lensalso,randomalso,ravelyearsbinary,ravelbinary,shuffletype,timeper)
    data_obs,lats_obs,lons_obs = df.getRegion(data_obs,lats_obs,lons_obs,lat_bounds,lon_bounds)
    
    print('our OBS dataset: ',dataset_obs,' is shaped',data_obs.shape)
    return data_obs,lats_obs,lons_obs

### Call functions
for vv in range(1):
    for mo in range(1):
        variq = variables[vv]
        monthlychoice = monthlychoiceq[mo]
        directoryfigure = '/Users/zlabe/Desktop/ModelComparison_v1/Climatologies/Arctic/Anomalies/'
        saveData =  monthlychoice + '_' + variq + '_' + reg_name + '_' + dataset_obs
        print('*Filename == < %s >' % saveData) 
    
        ### Read data
        modelraw,lats,lons = read_primary_dataset(variq,dataset,monthlychoice,numOfEns,
                                                lensalso,randomalso,ravelyearsbinary,
                                                ravelbinary,shuffletype,timeper,
                                                lat_bounds,lon_bounds)
        obsraw,lats_obs,lons_obs = read_obs_dataset(variq,dataset_obs,numOfEns,lensalso,randomalso,ravelyearsbinary,ravelbinary,shuffletype,lat_bounds=lat_bounds,lon_bounds=lon_bounds)
        models,obs = dSS.calculate_anomalies(modelraw,obsraw,lats,lons,baseline,yearsall)

        ### Add mmean
        mmmean = np.nanmean(models,axis=0)[np.newaxis,:,:,:,:]
        models = np.append(models,mmmean,axis=0)
        
        ### Slice time for Arctic Amplification
        AAyr = 15
        # models = models[:,:,-AAyr:,:,:]
        # obs = obs[-AAyr:,:,:]
        models = models[:,:,:-AAyr,:,:]
        obs = obs[:-AAyr,:,:]
        
###############################################################################        
        ### Calculate statistics
        clim = np.nanmean(models[:,:,:,:,:],axis=2)
        climobs = np.nanmean(obs[:,:,:],axis=0)
        
        std = np.nanstd(models[:,:,:,:,:],axis=2)
        stdobs = np.nanstd(obs[:,:,:],axis=0)
        
        bias = models - obs
        biasmean = np.nanmean(bias[:,:,:,:,:],axis=2)

###############################################################################          
        ### Calculate ensemble spread statistics
        maxens = np.nanmax(models[:,:,:,:,:],axis=1)
        minens = np.nanmin(models[:,:,:,:,:],axis=1)
        spread = maxens - minens
        spreadmean = np.nanmean(spread[:,:,:,:],axis=1) # across all years

###############################################################################          
        ### Ensemble mean
        climens = np.nanmean(clim[:,:,:,:],axis=1)
        stdens = np.nanmean(std[:,:,:,:],axis=1)
        biasens = np.nanmean(biasmean[:,:,:,:],axis=1)

###############################################################################          
        ### Assemble all data for plotting
        stdall = np.append(stdobs[np.newaxis,:,:],stdens,axis=0)
        climall = np.append(climobs[np.newaxis,:,:],climens,axis=0)
        biasall = biasens
        spreadall = spreadmean
        
        ### Read in data from LRP for statistical significance
        mask = True
        directorydataANN = '/Users/zlabe/Documents/Research/ModelComparison/Data/MSFigures_v1/'
        lrpAA = np.load(directorydataANN + 'LRPcomposites_LowerArctic_8classes.npy',allow_pickle=True)
        
        lrpAAn = np.empty((lrpAA.shape))
        for i in range(lrpAA.shape[0]):
            lrpAAn[i] = lrpAA[i]/np.nanmax(lrpAA[i])
            
        lrpthresh = 0.1
        lrpAAn[np.where(lrpAAn < lrpthresh)] = 0  
        lrpAAn[np.where(lrpAAn >= lrpthresh)] = 1
        
        ### Mask data
        biasallm = biasall * lrpAAn
        biasallm[np.where(biasallm == 0)] = np.nan
   
###############################################################################     
        if variq == 'T2M':
            limit = np.arange(-1,1.01,0.01)
            barlim = np.round(np.arange(-1,2,1),2)
            cmap = cmocean.cm.balance
            label = r'\textbf{[T2M-BIAS : $^{\circ}$C]}'
        
        fig = plt.figure(figsize=(10,2))
        for r in range(len(modelGCMs)):
            var = biasall[r]
            varno = biasallm[r]
            
            ax1 = plt.subplot(1,len(modelGCMs),r+1)
            if reg_name == 'LowerArctic':
                m = Basemap(projection='npstere',boundinglat=61.5,lon_0=0,
                            resolution='l',round =True,area_thresh=10000)
            else:
                m = Basemap(projection='npstere',boundinglat=71,lon_0=0,
                            resolution='l',round =True,area_thresh=10000)
            m.drawcoastlines(color='k',linewidth=0.27)
                
            var, lons_cyclic = addcyclic(var, lons)
            var, lons_cyclic = shiftgrid(180., var, lons_cyclic, start=False)
            varno, lons_cyclic = addcyclic(varno, lons)
            varno, lons_cyclic = shiftgrid(180., varno, lons_cyclic, start=False)
            
            lon2d, lat2d = np.meshgrid(lons_cyclic, lats)
            x, y = m(lon2d, lat2d)
               
            circle = m.drawmapboundary(fill_color='dimgrey',color='dimgray',
                              linewidth=0.7)
            circle.set_clip_on(False)
            
            cs1 = m.contourf(x,y,varno,limit,extend='both')
            cs2 = m.contourf(x,y,var,limit,extend='both',
                              alpha=0.6,antialiased=True)
            cs1.set_cmap(cmap) 
            cs2.set_cmap(cmap) 
            
            if ocean_only == True:
                m.fillcontinents(color='dimgrey',lake_color='dimgrey')
            elif land_only == True:
                m.drawlsmask(land_color=(0,0,0,0),ocean_color='darkgrey',lakes=True,zorder=5)
                    
            ax1.annotate(r'\textbf{%s}' % modelGCMs[r],xy=(0,0),xytext=(0.5,1.10),
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
        plt.subplots_adjust(top=0.85,wspace=0.02,hspace=0.02,bottom=0.14)
        
        plt.savefig(directoryfigure + 'Arctic_Biasperiod_Anomalies.png',dpi=300)
        
###############################################################################     
        if variq == 'T2M':
            limit = np.arange(0,7.01,0.25)
            barlim = np.round(np.arange(0,8,1),2)
            cmap = cmr.fall_r
            label = r'\textbf{[T2M-SPREAD : $^{\circ}$C]}'
        
        fig = plt.figure(figsize=(10,2))
        for r in range(len(modelGCMs)):
            var = spreadall[r]
            
            ax1 = plt.subplot(1,len(modelGCMs),r+1)
            if reg_name == 'LowerArctic':
                m = Basemap(projection='npstere',boundinglat=61.5,lon_0=0,
                            resolution='l',round =True,area_thresh=10000)
            else:
                m = Basemap(projection='npstere',boundinglat=71,lon_0=0,
                            resolution='l',round =True,area_thresh=10000)
            m.drawcoastlines(color='darkgrey',linewidth=0.27)
                
            var, lons_cyclic = addcyclic(var, lons)
            var, lons_cyclic = shiftgrid(180., var, lons_cyclic, start=False)
            lon2d, lat2d = np.meshgrid(lons_cyclic, lats)
            x, y = m(lon2d, lat2d)
               
            circle = m.drawmapboundary(fill_color='dimgrey',color='dimgray',
                              linewidth=0.7)
            circle.set_clip_on(False)
            
            cs1 = m.contourf(x,y,var,limit,extend='max')
            cs1.set_cmap(cmap) 
            
            if ocean_only == True:
                m.fillcontinents(color='dimgrey',lake_color='dimgrey')
            elif land_only == True:
                m.drawlsmask(land_color=(0,0,0,0),ocean_color='darkgrey',lakes=True,zorder=5)
                    
            ax1.annotate(r'\textbf{%s}' % modelGCMs[r],xy=(0,0),xytext=(0.5,1.10),
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
        plt.subplots_adjust(top=0.85,wspace=0.02,hspace=0.02,bottom=0.14)
        
        plt.savefig(directoryfigure + 'Arctic_Spreadperiod_Anomalies.png',dpi=300)
        
###############################################################################     
        if variq == 'T2M':
            limit = np.arange(0,3.01,0.25)
            barlim = np.round(np.arange(0,4,1),2)
            cmap = cmr.dusk
            label = r'\textbf{[T2M-STDev : $^{\circ}$C]}'
        
        fig = plt.figure(figsize=(10,2))
        for r in range(len(allDataLabels)):
            var = stdall[r]
            
            ax1 = plt.subplot(1,len(allDataLabels),r+1)
            if reg_name == 'LowerArctic':
                m = Basemap(projection='npstere',boundinglat=61.5,lon_0=0,
                            resolution='l',round =True,area_thresh=10000)
            else:
                m = Basemap(projection='npstere',boundinglat=71,lon_0=0,
                            resolution='l',round =True,area_thresh=10000)
            m.drawcoastlines(color='darkgrey',linewidth=0.27)
                
            var, lons_cyclic = addcyclic(var, lons)
            var, lons_cyclic = shiftgrid(180., var, lons_cyclic, start=False)
            lon2d, lat2d = np.meshgrid(lons_cyclic, lats)
            x, y = m(lon2d, lat2d)
               
            circle = m.drawmapboundary(fill_color='dimgrey',color='dimgray',
                              linewidth=0.7)
            circle.set_clip_on(False)
            
            cs1 = m.contourf(x,y,var,limit,extend='max')
            cs1.set_cmap(cmap) 
            
            if ocean_only == True:
                m.fillcontinents(color='dimgrey',lake_color='dimgrey')
            elif land_only == True:
                m.drawlsmask(land_color=(0,0,0,0),ocean_color='darkgrey',lakes=True,zorder=5)
                    
            ax1.annotate(r'\textbf{%s}' % allDataLabels[r],xy=(0,0),xytext=(0.5,1.10),
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
        plt.subplots_adjust(top=0.85,wspace=0.02,hspace=0.02,bottom=0.14)
        
        plt.savefig(directoryfigure + 'Arctic_STDperiod_Anomalies.png',dpi=300)