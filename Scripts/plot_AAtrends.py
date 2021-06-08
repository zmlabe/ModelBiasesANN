"""
Script for plotting trends in Arctic amplification in models and observations

Author     : Zachary M. Labe
Date       : 27 May 2021
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

### Plotting defaults 
plt.rc('text',usetex=True)
plt.rc('font',**{'family':'sans-serif','sans-serif':['Avant Garde']}) 

###############################################################################
###############################################################################
###############################################################################
### Data preliminaries 
modelGCMs = ['CanESM2','MPI','CSIRO-MK3.6','KNMI-ecearth','GFDL-CM3','GFDL-ESM2M','LENS','MMmean']
dataset_obs = 'ERA5BE'
allDataLabels = [dataset_obs,'CanESM2','MPI','CSIRO-MK3.6','KNMI-ecearth','GFDL-CM3','GFDL-ESM2M','LENS','MMmean']
letters = ["a","b","c","d","e","f","g","h","i","j","k","l","m"]
datasetsingle = ['SMILE']
monthlychoiceq = ['JFM','AMJ','JAS','OND','annual']
monthlychoiceq = ['annual']
variables = ['T2M','P','SLP']
# variables = ['T2M']
slicetimeperiod = 'preAA'
reg_name = 'SMILEGlobe'
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
  
def read_obs_dataset(variq,monthlychoice,dataset_obs,numOfEns,lensalso,randomalso,ravelyearsbinary,ravelbinary,shuffletype,lat_bounds=lat_bounds,lon_bounds=lon_bounds):
    data_obs,lats_obs,lons_obs = df.readFiles(variq,dataset_obs,monthlychoice,numOfEns,lensalso,randomalso,ravelyearsbinary,ravelbinary,shuffletype,timeper)
    data_obs,lats_obs,lons_obs = df.getRegion(data_obs,lats_obs,lons_obs,lat_bounds,lon_bounds)
    
    print('our OBS dataset: ',dataset_obs,' is shaped',data_obs.shape)
    return data_obs,lats_obs,lons_obs
###############################################################################
###############################################################################
###############################################################################
### Call functions
def readInData(variables,monthlychoice,reg_name,dataset_obs,slicetimeperiod):
    variq = variables
    directoryfigure = '/Users/zlabe/Desktop/ModelComparison_v1/Climatologies/%s/' % variq
    saveData =  monthlychoice + '_' + variq + '_' + reg_name + '_' + dataset_obs
    print('*Filename == < %s >' % saveData) 

    ### Read data
    modelsc,lats,lons = read_primary_dataset(variq,dataset,monthlychoice,numOfEns,
                                            lensalso,randomalso,ravelyearsbinary,
                                            ravelbinary,shuffletype,timeper,
                                            lat_bounds,lon_bounds)
    obs,lats_obs,lons_obs = read_obs_dataset(variq,monthlychoice,dataset_obs,numOfEns,lensalso,randomalso,ravelyearsbinary,ravelbinary,shuffletype,lat_bounds=lat_bounds,lon_bounds=lon_bounds)

###############################################################################        
    ### Add multimodel mean
    mm = np.nanmean(modelsc,axis=0)
    models = np.append(modelsc,mm[np.newaxis,:,:,:,:],axis=0)
    
###############################################################################        
    ### Slicetimeperiod
    if slicetimeperiod == 'AA':
        yearsq = np.where((yearsall >= 2000) & (yearsall <= 2019))[0]
        yearsnew = yearsall[yearsq]
        models = models[:,:,yearsq,:,:]
        obs = obs[yearsq,:,:]
    elif slicetimeperiod == 'preAA':
        yearsq = np.where((yearsall >= 1950) & (yearsall <= 1989))[0]
        yearsnew = yearsall[yearsq]
        models = models[:,:,yearsq,:,:]
        obs = obs[yearsq,:,:]
    else:
        print(ValueError('WRONG TIME PERIOD FOR TREND!'))
        sys.exit()
    
###############################################################################          
    ### Calculate decadal trends
    dectrendmodels = np.empty((models.shape[0],models.shape[1],models.shape[3],models.shape[4]))
    for i in range(models.shape[0]):
        trendmodel = UT.linearTrend(models[i,:,:,:,:],yearsnew,level,yearsnew.min(),yearsnew.max())
        dectrendmodels[i,:,:,:] = trendmodel * 10.
        print('Completed: trends for model: %s!' % modelGCMs[i])
        
    trendsobs = UT.linearTrendR(obs,yearsnew,level,yearsnew.min(),yearsnew.max())
    dectrendobs = trendsobs * 10.

    meantrend = np.nanmean(dectrendmodels[:,:,:,:],axis=1)
###############################################################################          
    ### Assemble all data for plotting
    trendall = np.append(dectrendobs[np.newaxis,:,:],meantrend,axis=0)
    
    return trendall,lats,lons,saveData,directoryfigure

###############################################################################
###############################################################################
###############################################################################     
###############################################################################
###############################################################################
###############################################################################     
# for vv in range(len(variables)):
#     for mo in range(len(monthlychoiceq)):
for vv in range(1):
    for mo in range(1): 
        variq = variables[vv]
        monthlychoice = monthlychoiceq[mo]
        trendall,lats,lons,saveData,directoryfigure = readInData(variables[vv],monthlychoiceq[mo],reg_name,dataset_obs,slicetimeperiod)
        
###############################################################################
###############################################################################
###############################################################################     
        #######################################################################
        #######################################################################
        #######################################################################
        ### Plot subplot of model trends        
        if variq == 'T2M':
            if slicetimeperiod == 'AA':
                limit = np.arange(-2,2.01,0.05)
                barlim = np.round(np.arange(-2,2.01,0.5),2)
                cmap = cmocean.cm.balance
                label = r'\textbf{%s -- [$^{\circ}$C PER DECADE] -- 2000-2019}' % variq
            if slicetimeperiod == 'preAA':
                limit = np.arange(-0.5,0.51,0.01)
                barlim = np.round(np.arange(-0.5,0.51,0.25),2)
                cmap = cmocean.cm.balance
                label = r'\textbf{%s -- [$^{\circ}$C PER DECADE] -- 1950-1989}' % variq
        elif variq == 'P':
            limit = np.arange(-0.3,0.301,0.01)
            barlim = np.round(np.arange(-0.30,0.301,0.1),2)
            cmap = cmocean.cm.tarn                                                                                                                                   
            label = r'\textbf{%s -- [mm/day PER DECADE] -- 1950-2019}' % variq
        elif variq == 'SLP':
            limit = np.arange(-0.4,0.401,0.01)
            barlim = np.round(np.arange(-0.4,0.401,0.2),2)
            cmap = cmocean.cm.diff
            label = r'\textbf{%s -- [hPa PER DECADE] -- 1950-2019}' % variq
        
        fig = plt.figure(figsize=(8,4))
        for r in range(len(allDataLabels)):
            var = trendall[r]
            
            ax1 = plt.subplot(2,5,r+1)
            m = Basemap(projection='npstere',boundinglat=60,lon_0=0,
                        resolution='l',round =True,area_thresh=10000)
            m.drawcoastlines(color='dimgrey',linewidth=0.27)
                
            var, lons_cyclic = addcyclic(var, lons)
            var, lons_cyclic = shiftgrid(180., var, lons_cyclic, start=False)
            lon2d, lat2d = np.meshgrid(lons_cyclic, lats)
            x, y = m(lon2d, lat2d)
               
            if r == 0:
                circle = m.drawmapboundary(fill_color='white',color='k',
                                  linewidth=4)
                circle.set_clip_on(False)
            else:
                circle = m.drawmapboundary(fill_color='white',color='dimgray',
                                  linewidth=0.7)
                circle.set_clip_on(False)
            
            cs1 = m.contourf(x,y,var,limit,extend='both')
            cs1.set_cmap(cmap) 
                    
            ax1.annotate(r'\textbf{%s}' % allDataLabels[r],xy=(0,0),xytext=(0.05,0.92),
                          textcoords='axes fraction',color='dimgrey',fontsize=6,
                          rotation=50,ha='center',va='center')
            ax1.annotate(r'\textbf{[%s]}' % letters[r],xy=(0,0),xytext=(0.86,0.97),
                          textcoords='axes fraction',color='k',fontsize=6,
                          rotation=0,ha='center',va='center')
            
        ###############################################################################
        cbar_ax1 = fig.add_axes([0.36,0.09,0.3,0.03])                
        cbar1 = fig.colorbar(cs1,cax=cbar_ax1,orientation='horizontal',
                            extend='both',extendfrac=0.07,drawedges=False)
        cbar1.set_label(label,fontsize=9,color='dimgrey',labelpad=1.4)  
        cbar1.set_ticks(barlim)
        cbar1.set_ticklabels(list(map(str,barlim)))
        cbar1.ax.tick_params(axis='x', size=.01,labelsize=5)
        cbar1.outline.set_edgecolor('dimgrey')
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.85,wspace=0.02,hspace=0.03,bottom=0.15)
        
        plt.savefig(directoryfigure + 'Trend-%s-Arctic-%s.png' % (slicetimeperiod,saveData),dpi=300)