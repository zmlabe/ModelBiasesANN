"""
Script for plotting differences in model and observational climatologies for 
select variables over the 1950 to 2019 period

Author     : Zachary M. Labe
Date       : 8 March 2021
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
plt.rc('savefig',facecolor='black')
plt.rc('axes',edgecolor='darkgrey')
plt.rc('xtick',color='darkgrey')
plt.rc('ytick',color='darkgrey')
plt.rc('axes',labelcolor='darkgrey')
plt.rc('axes',facecolor='black')

###############################################################################
###############################################################################
###############################################################################
### Data preliminaries 
modelGCMs = ['CanESM2','MPI','CSIRO-MK3.6','KNMI-ecearth','GFDL-CM3','GFDL-ESM2M','LENS','MM-Mean']
dataset_obs = 'ERA5BE'
allDataLabels = ['CanESM2','MPI','CSIRO-MK3.6','EC-EARTH','GFDL-CM3','GFDL-ESM2M','LENS','MM-Mean']
letters = ["a","b","c","d","e","f","g","h","i","j","k","l","m","n"]
datasetsingle = ['SMILE']
monthlychoiceq = ['JFM','AMJ','JAS','OND','annual']
monthlychoiceq = ['annual']
variables = ['T2M','P','SLP']
variables = ['T2M']
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
  
def read_obs_dataset(variq,dataset_obs,numOfEns,lensalso,randomalso,ravelyearsbinary,ravelbinary,shuffletype,lat_bounds=lat_bounds,lon_bounds=lon_bounds):
    data_obs,lats_obs,lons_obs = df.readFiles(variq,dataset_obs,monthlychoice,numOfEns,lensalso,randomalso,ravelyearsbinary,ravelbinary,shuffletype,timeper)
    data_obs,lats_obs,lons_obs = df.getRegion(data_obs,lats_obs,lons_obs,lat_bounds,lon_bounds)
    
    print('our OBS dataset: ',dataset_obs,' is shaped',data_obs.shape)
    return data_obs,lats_obs,lons_obs

def calcCorrs(era,mod):  
    ### Calculate correlation coefficients at each grid point
    corrm = np.empty((mod.shape[0],mod.shape[2],mod.shape[3]))
    pvalue = np.empty((mod.shape[0],mod.shape[2],mod.shape[3]))
    for ru in range(mod.shape[0]):
        for i in range(mod.shape[2]):
            for j in range(mod.shape[3]):
                xx = era[:,i,j]
                yy = mod[ru,:,i,j]
                na = np.logical_or(np.isnan(xx),np.isnan(yy))
                corrm[ru,i,j],pvalue[ru,i,j] = sts.pearsonr(xx[~na],yy[~na])
    
    ### Significant at 95% confidence level
    pvalue[np.where(pvalue >= 0.05)] = np.nan
    pvalue[np.where(pvalue < 0.05)] = 1.
    
    return corrm,pvalue

### Call functions
# for vv in range(len(variables)):
#     for mo in range(len(monthlychoiceq)):
for vv in range(1):
    for mo in range(1):
#         variq = variables[vv]
#         monthlychoice = monthlychoiceq[mo]
#         directoryfigure = '/Users/zlabe/Documents/Projects/ModelBiasesANN/Dark_Figures/'
#         saveData =  monthlychoice + '_' + variq + '_' + reg_name + '_' + dataset_obs
#         print('*Filename == < %s >' % saveData) 
    
#         ### Read data
#         models,lats,lons = read_primary_dataset(variq,dataset,monthlychoice,numOfEns,
#                                                 lensalso,randomalso,ravelyearsbinary,
#                                                 ravelbinary,shuffletype,timeper,
#                                                 lat_bounds,lon_bounds)
#         obs,lats_obs,lons_obs = read_obs_dataset(variq,dataset_obs,numOfEns,lensalso,randomalso,ravelyearsbinary,ravelbinary,shuffletype,lat_bounds=lat_bounds,lon_bounds=lon_bounds)
#         modelanom,obsanom = dSS.calculate_anomalies(models,obs,lats,lons,baseline,yearsall)
#         modelannmean,obsannmean = dSS.remove_annual_mean(models,obs,lats,lons,lats_obs,lons_obs)

#         ### Add mmean
#         mmmean = np.nanmean(models,axis=0)[np.newaxis,:,:,:,:]
#         models = np.append(models,mmmean,axis=0)
        
# ###############################################################################        
#         ### Calculate statistics
#         clim = np.nanmean(models[:,:,:,:,:],axis=2)
#         climobs = np.nanmean(obs[:,:,:],axis=0)
        
#         std = np.nanstd(models[:,:,:,:,:],axis=2)
#         stdobs = np.nanstd(obs[:,:,:],axis=0)
        
#         bias = models - obs
#         biasmean = np.nanmean(bias[:,:,:,:,:],axis=2)

# ###############################################################################          
#         ### Calculate ensemble spread statistics
#         maxens = np.nanmax(models[:,:,:,:,:],axis=1)
#         minens = np.nanmin(models[:,:,:,:,:],axis=1)
#         spread = maxens - minens
#         spreadmean = np.nanmean(spread[:,:,:,:],axis=1) # across all years

# ###############################################################################          
#         ### Ensemble mean
#         climens = np.nanmean(clim[:,:,:,:],axis=1)
#         stdens = np.nanmean(std[:,:,:,:],axis=1)
#         biasens = np.nanmean(biasmean[:,:,:,:],axis=1)

# ###############################################################################          
#         ### Assemble all data for plotting
#         stdall = np.append(stdobs[np.newaxis,:,:],stdens,axis=0)
#         climall = np.append(climobs[np.newaxis,:,:],climens,axis=0)
#         biasall = biasens
#         spreadall = spreadmean
        
# ###############################################################################
# ###############################################################################
# ###############################################################################     
#         #######################################################################
#         #######################################################################
#         #######################################################################
#         ### Plot subplot of mean climate
        if variq == 'T2M':
            limit = np.arange(-35,30.01,0.5)
            barlim = np.round(np.arange(-35,31,65),2)
            cmap = plt.cm.CMRmap
            label = r'\textbf{%s -- [$^{\circ}$C] -- 1950-2019}' % variq
        elif variq == 'P':
            limit = np.arange(0,10.01,0.01)
            barlim = np.round(np.arange(-0,10.01,2),2)
            cmap = cmocean.cm.rain                                                                                                                                 
            label = r'\textbf{%s -- [mm/day mean] -- 1950-2019}' % variq
        elif variq == 'SLP':
            limit = np.arange(990,1020.01,0.5)
            barlim = np.round(np.arange(990,1021,10),2)
            cmap = sss.Nuuk_20.mpl_colormap 
            label = r'\textbf{%s -- [hPa : mean] -- 1950-2019}' % variq
        
        # fig = plt.figure(figsize=(8,4))
        # for r in range(0,len(allDataLabels)):
        #     var = climall[r]
            
        #     ax1 = plt.subplot(2,4,r+1)
        #     m = Basemap(projection='moll',lon_0=0,resolution='l',area_thresh=10000)
        #     m.drawcoastlines(color='dimgrey',linewidth=0.27)
                
        #     var, lons_cyclic = addcyclic(var, lons)
        #     var, lons_cyclic = shiftgrid(180., var, lons_cyclic, start=False)
        #     lon2d, lat2d = np.meshgrid(lons_cyclic, lats)
        #     x, y = m(lon2d, lat2d)
               
        #     circle = m.drawmapboundary(fill_color='k',color='k',
        #                       linewidth=0.7)
        #     circle.set_clip_on(False)
            
        #     if variq == 'P':
        #         xx = 'max'
        #     else:
        #         xx = 'both'
        #     cs1 = m.contourf(x,y,var,limit,extend=xx)
        #     cs1.set_cmap(cmap) 
                    
        #     ax1.annotate(r'\textbf{%s}' % allDataLabels[r],xy=(0,0),xytext=(0.5,1.10),
        #                   textcoords='axes fraction',color='dimgrey',fontsize=10,
        #                   rotation=0,ha='center',va='center')
            
        # ###############################################################################
        # cbar_ax1 = fig.add_axes([0.36,0.11,0.3,0.03])                
        # cbar1 = fig.colorbar(cs1,cax=cbar_ax1,orientation='horizontal',
        #                     extend=xx,extendfrac=0.07,drawedges=False)
        # cbar1.set_label(label,fontsize=9,color='k',labelpad=1.4)  
        # cbar1.set_ticks(barlim)
        # cbar1.set_ticklabels(list(map(str,barlim)))
        # cbar1.ax.tick_params(axis='x', size=.01,labelsize=5)
        # cbar1.outline.set_edgecolor('darkgrey')
        
        # plt.tight_layout()
        # plt.subplots_adjust(top=0.85,wspace=0.02,hspace=0.00,bottom=0.14)
        
        # plt.savefig(directoryfigure + 'Mean-%s_CLEAR8.png' % saveData,dpi=1000)
            
        fig = plt.figure(figsize=(2,8))
        for r in range(0,len(allDataLabels)):
            var = climall[r]
            
            ax1 = plt.subplot(8,1,r+1)
            m = Basemap(projection='moll',lon_0=0,resolution='l',area_thresh=10000)
            m.drawcoastlines(color='dimgrey',linewidth=0.27)
                
            var, lons_cyclic = addcyclic(var, lons)
            var, lons_cyclic = shiftgrid(180., var, lons_cyclic, start=False)
            lon2d, lat2d = np.meshgrid(lons_cyclic, lats)
            x, y = m(lon2d, lat2d)
               
            circle = m.drawmapboundary(fill_color='k',color='k',
                              linewidth=0.7)
            circle.set_clip_on(False)
            
            if variq == 'P':
                xx = 'max'
            else:
                xx = 'both'
            cs1 = m.contourf(x,y,var,limit,extend=xx)
            cs1.set_cmap(cmap) 
                    
            ax1.annotate(r'\textbf{%s}' % allDataLabels[r],xy=(0,0),xytext=(1.04,0.5),
                          textcoords='axes fraction',color='w',fontsize=10,
                          rotation=270,ha='center',va='center')
            
        ###############################################################################
        cbar_ax1 = fig.add_axes([0.20,0.09,0.6,0.009])                
        cbar1 = fig.colorbar(cs1,cax=cbar_ax1,orientation='horizontal',
                            extend=xx,extendfrac=0.07,drawedges=False)
        cbar1.set_label(label,fontsize=9,color='darkgrey',labelpad=1.4)  
        cbar1.set_ticks(barlim)
        cbar1.set_ticklabels(list(map(str,barlim)))
        cbar1.ax.tick_params(axis='x', size=.01,labelsize=4.5,pad=2)
        cbar1.outline.set_edgecolor('darkgrey')
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.11,hspace=0.00)
        plt.savefig(directoryfigure + 'Mean-%s_DARK8_Vertical.png' % saveData,dpi=1000)
        