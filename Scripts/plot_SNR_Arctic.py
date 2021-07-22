"""
Script for exploring intermodel differences in the Arctic for trends

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
        directoryfigure = '/Users/zlabe/Desktop/ModelComparison_v1/Climatologies/Arctic/Trends/'
        saveData =  monthlychoice + '_' + variq + '_' + reg_name + '_' + dataset_obs
        print('*Filename == < %s >' % saveData) 
    
        ### Read data
        models,lats,lons = read_primary_dataset(variq,dataset,monthlychoice,numOfEns,
                                                lensalso,randomalso,ravelyearsbinary,
                                                ravelbinary,shuffletype,timeper,
                                                lat_bounds,lon_bounds)
        obs,lats_obs,lons_obs = read_obs_dataset(variq,dataset_obs,numOfEns,lensalso,randomalso,ravelyearsbinary,ravelbinary,shuffletype,lat_bounds=lat_bounds,lon_bounds=lon_bounds)

        ### Add mmean
        mmmean = np.nanmean(models,axis=0)[np.newaxis,:,:,:,:]
        models = np.append(models,mmmean,axis=0)
        
        ### Slice time for Arctic Amplification
        AAyr = 15
        AA = 'pre'
        if AA == 'now':
            models = models[:,:,-AAyr:,:,:]
            obs = obs[-AAyr:,:,:]
            yearsall = yearsall[-AAyr:]
        elif AA == 'pre':
            models = models[:,:,:-AAyr,:,:]
            obs = obs[:-AAyr,:,:]
            yearsall = yearsall[:-AAyr]
        
###############################################################################        
        ### Calculate Trends
        dectrendmodels = np.empty((models.shape[0],models.shape[1],models.shape[3],models.shape[4]))
        for i in range(models.shape[0]):
            trendmodel = UT.linearTrend(models[i,:,:,:,:],yearsall,level,yearsall.min(),yearsall.max())
            dectrendmodels[i,:,:,:] = trendmodel * 10.
            print('Completed: trends for model: %s!' % modelGCMs[i])
            
        trendsobs = UT.linearTrendR(obs,yearsall,level,yearsall.min(),yearsall.max())
        dectrendobs = trendsobs * 10.
        
        ensmeantrend = np.nanmean(dectrendmodels,axis=1)
        
        ### Calculate SNR
        stdtrends = np.nanstd(dectrendmodels,axis=1)
        SNR = abs(ensmeantrend) / stdtrends
        
###############################################################################                  
        ### Read in data from LRP for statistical significance
        mask = True
        directorydataANN = '/Users/zlabe/Documents/Research/ModelComparison/Data/MSFigures_v1/'
        if AA == 'now':
            lrpAA = np.load(directorydataANN + 'LRPcomposites_LowerArcticAA_8classes.npy',allow_pickle=True)
        else:
            lrpAA = np.load(directorydataANN + 'LRPcomposites_LowerArctic_8classes.npy',allow_pickle=True)
        
        lrpAAn = np.empty((lrpAA.shape))
        for i in range(lrpAA.shape[0]):
            lrpAAn[i] = lrpAA[i]/np.nanmax(lrpAA[i])
            
        lrpthresh = 0.1
        lrpAAn[np.where(lrpAAn < lrpthresh)] = 0  
        lrpAAn[np.where(lrpAAn >= lrpthresh)] = 1
        
        ### Mask data
        SNRm = SNR * lrpAAn
        SNRm[np.where(SNRm == 0)] = np.nan
   
###############################################################################     
        if variq == 'T2M':
            limit = np.arange(0,4.1,0.5)
            barlim = np.round(np.arange(0,4.1,1),2)
            cmap = plt.cm.CMRmap
            label = r'\textbf{[T2M-SNR]}'
        
        fig = plt.figure(figsize=(10,2))
        for r in range(len(modelGCMs)):
            var = SNR[r]
            varno = SNRm[r]
            
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
            
            cs1 = m.contourf(x,y,varno,limit,extend='max',antialiased=True)
            cs2 = m.contourf(x,y,var,limit,extend='max',
                              alpha=0.4,antialiased=True)
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
                            extend='max',extendfrac=0.07,drawedges=True)
        cbar1.set_label(label,fontsize=9,color='dimgrey',labelpad=1.4)  
        cbar1.set_ticks(barlim)
        cbar1.set_ticklabels(list(map(str,barlim)))
        cbar1.ax.tick_params(axis='x', size=.001,labelsize=7)
        cbar1.dividers.set_color('dimgrey')
        cbar1.dividers.set_linewidth(1)
        cbar1.outline.set_edgecolor('dimgrey')
        cbar1.outline.set_linewidth(1)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.85,wspace=0.02,hspace=0.02,bottom=0.14)
        
        if AA == 'now':
            plt.savefig(directoryfigure + 'Arctic_SNR_AA.png',dpi=300)
        else:
            plt.savefig(directoryfigure + 'Arctic_SNR.png',dpi=300)