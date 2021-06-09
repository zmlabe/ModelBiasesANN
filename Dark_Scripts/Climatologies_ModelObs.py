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

### Plotting defaults 
matplotlib.rc('savefig', facecolor='black')
matplotlib.rc('axes', edgecolor='darkgrey')
matplotlib.rc('xtick', color='darkgrey')
matplotlib.rc('ytick', color='darkgrey')
matplotlib.rc('axes', labelcolor='darkgrey')
matplotlib.rc('axes', facecolor='black')
plt.rc('text',usetex=True)
plt.rc('font',**{'family':'sans-serif','sans-serif':['Avant Garde']})

###############################################################################
###############################################################################
###############################################################################
### Data preliminaries 
modelGCMs = ['CanESM2','MPI','CSIRO-MK3.6','KNMI-ecearth','GFDL-CM3','GFDL-ESM2M','LENS']
dataset_obs = 'ERA5BE'
allDataLabels = [dataset_obs,'CanESM2','MPI','CSIRO-MK3.6','KNMI-ecearth','GFDL-CM3','GFDL-ESM2M','LENS']
letters = ["a","b","c","d","e","f","g","h","i","j","k","l","m"]
datasetsingle = ['SMILE']
monthlychoiceq = ['JFM','AMJ','JAS','OND','annual']
monthlychoiceq = ['annual']
variables = ['T2M','P','SLP']
# variables = ['T2M']
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
        variq = variables[vv]
        monthlychoice = monthlychoiceq[mo]
        directoryfigure = '/Users/zlabe/Documents/Projects/ModelBiasesANN/Dark_Figures/'
        saveData =  monthlychoice + '_' + variq + '_' + reg_name + '_' + dataset_obs
        print('*Filename == < %s >' % saveData) 
    
        ### Read data
        models,lats,lons = read_primary_dataset(variq,dataset,monthlychoice,numOfEns,
                                                lensalso,randomalso,ravelyearsbinary,
                                                ravelbinary,shuffletype,timeper,
                                                lat_bounds,lon_bounds)
        obs,lats_obs,lons_obs = read_obs_dataset(variq,dataset_obs,numOfEns,lensalso,randomalso,ravelyearsbinary,ravelbinary,shuffletype,lat_bounds=lat_bounds,lon_bounds=lon_bounds)
        modelanom,obsanom = dSS.calculate_anomalies(models,obs,lats,lons,baseline,yearsall)
        modelannmean,obsannmean = dSS.remove_annual_mean(models,obs,lats,lons,lats_obs,lons_obs)

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
        ### Calculate decadal trends
        dectrendmodels = np.empty((models.shape[0],models.shape[1],models.shape[3],models.shape[4]))
        for i in range(models.shape[0]):
            trendmodel = UT.linearTrend(models[i,:,:,:,:],yearsall,level,yearsall.min(),yearsall.max())
            dectrendmodels[i,:,:,:] = trendmodel * 10.
            print('Completed: trends for model: %s!' % modelGCMs[i])
            
        trendsobs = UT.linearTrendR(obs,yearsall,level,yearsall.min(),yearsall.max())
        dectrendobs = trendsobs * 10.
        
###############################################################################          
        ### Calculate spatial correlations
        corrall = np.empty((models.shape[0],models.shape[1],models.shape[3],models.shape[4]))
        pall = np.empty((models.shape[0],models.shape[1],models.shape[3],models.shape[4]))
        for i in range(models.shape[0]):
            corrall[i,:,:,:],pall[i,:,:,:] = calcCorrs(obs[:,:,:],models[i,:,:,:,:])
            print('Completed: correlations for model: %s!' % modelGCMs[i])
            
        corrmean = np.nanmean(corrall,axis=1) # ensemble mean
        pmean = np.nanmean(pall,axis=1) # ensemble mean
 
###############################################################################         
        ### Calculate mean model trend
        meantrend = np.nanmean(dectrendmodels[:,:,:,:],axis=1)

###############################################################################          
        ### Assemble all data for plotting
        stdall = np.append(stdobs[np.newaxis,:,:],stdens,axis=0)
        climall = np.append(climobs[np.newaxis,:,:],climens,axis=0)
        trendall = np.append(dectrendobs[np.newaxis,:,:],meantrend,axis=0)
        biasall = biasens
        spreadall = spreadmean
        corrall = corrmean
        pall = pmean
        
        #######################################################################
        #######################################################################
        #######################################################################
        ### Plot subplot of model standard deviation
        plt.rc('text',usetex=True)
        plt.rc('font',**{'family':'sans-serif','sans-serif':['Avant Garde']}) 
        
        if variq == 'T2M':
            limit = np.arange(0,3.01,0.1)
            barlim = np.round(np.arange(-0,3.01,1),2)
            cmap = sss.Batlow_20.mpl_colormap  
            label = r'\textbf{%s -- [$^{\circ}$C stdev.] -- 1950-2019}' % variq
        elif variq == 'P':
            limit = np.arange(0,3.01,0.1)
            barlim = np.round(np.arange(-0,3.01,1),2)
            cmap = sss.Batlow_20.mpl_colormap                                                                                                                                       
            label = r'\textbf{%s -- [mm/day stdev.] -- 1950-2019}' % variq
        elif variq == 'SLP':
            limit = np.arange(0,3.01,0.2)
            barlim = np.round(np.arange(0,3.01,1),2)
            cmap = sss.Batlow_20.mpl_colormap  
            label = r'\textbf{%s -- [hPa : stdev.] -- 1950-2019}' % variq
        
        fig = plt.figure(figsize=(8,4))
        for r in range(len(allDataLabels)):
            var = stdall[r]
            
            ax1 = plt.subplot(2,4,r+1)
            m = Basemap(projection='moll',lon_0=0,resolution='l',area_thresh=10000)
            m.drawcoastlines(color='dimgrey',linewidth=0.27)
                
            var, lons_cyclic = addcyclic(var, lons)
            var, lons_cyclic = shiftgrid(180., var, lons_cyclic, start=False)
            lon2d, lat2d = np.meshgrid(lons_cyclic, lats)
            x, y = m(lon2d, lat2d)
               
            circle = m.drawmapboundary(fill_color='dimgrey',color='dimgray',
                              linewidth=0.7)
            circle.set_clip_on(False)
            
            cs1 = m.contourf(x,y,var,limit,extend='max')
            cs1.set_cmap(cmap) 
                    
            ax1.annotate(r'\textbf{%s}' % allDataLabels[r],xy=(0,0),xytext=(0.5,1.10),
                          textcoords='axes fraction',color='w',fontsize=8,
                          rotation=0,ha='center',va='center')
            ax1.annotate(r'\textbf{[%s]}' % letters[r],xy=(0,0),xytext=(0.86,0.97),
                          textcoords='axes fraction',color='k',fontsize=6,
                          rotation=330,ha='center',va='center')
            
        ###############################################################################
        cbar_ax1 = fig.add_axes([0.36,0.11,0.3,0.03])                
        cbar1 = fig.colorbar(cs1,cax=cbar_ax1,orientation='horizontal',
                            extend='max',extendfrac=0.07,drawedges=False)
        cbar1.set_label(label,fontsize=9,color='dimgrey',labelpad=1.4)  
        cbar1.set_ticks(barlim)
        cbar1.set_ticklabels(list(map(str,barlim)))
        cbar1.ax.tick_params(axis='x', size=.01,labelsize=5)
        cbar1.outline.set_edgecolor('dimgrey')
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.85,wspace=0.02,hspace=0.00,bottom=0.14)
        
        plt.savefig(directoryfigure + 'Stdev-%s.png' % saveData,dpi=300)
        
###############################################################################
###############################################################################
###############################################################################     
        #######################################################################
        #######################################################################
        #######################################################################
        ### Plot subplot of mean climate
        if variq == 'T2M':
            limit = np.arange(-35,35.01,0.5)
            barlim = np.round(np.arange(-35,36,5),2)
            cmap = plt.cm.CMRmap_r
            label = r'\textbf{%s -- [$^{\circ}$C mean] -- 1950-2019}' % variq
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
        
        fig = plt.figure(figsize=(8,4))
        for r in range(len(allDataLabels)):
            var = climall[r]
            
            ax1 = plt.subplot(2,4,r+1)
            m = Basemap(projection='moll',lon_0=0,resolution='l',area_thresh=10000)
            m.drawcoastlines(color='dimgrey',linewidth=0.27)
                
            var, lons_cyclic = addcyclic(var, lons)
            var, lons_cyclic = shiftgrid(180., var, lons_cyclic, start=False)
            lon2d, lat2d = np.meshgrid(lons_cyclic, lats)
            x, y = m(lon2d, lat2d)
               
            if r == 0:
                circle = m.drawmapboundary(fill_color='k',color='k',
                                  linewidth=3)
                circle.set_clip_on(False)
            else:
                circle = m.drawmapboundary(fill_color='k',color='k',
                                  linewidth=0.7)
                circle.set_clip_on(False)
            
            if variq == 'P':
                xx = 'max'
            else:
                xx = 'both'
            cs1 = m.contourf(x,y,var,limit,extend=xx)
            cs1.set_cmap(cmap) 
                    
            ax1.annotate(r'\textbf{%s}' % allDataLabels[r],xy=(0,0),xytext=(0.5,1.10),
                          textcoords='axes fraction',color='w',fontsize=8,
                          rotation=0,ha='center',va='center')
            ax1.annotate(r'\textbf{[%s]}' % letters[r],xy=(0,0),xytext=(0.86,0.97),
                          textcoords='axes fraction',color='k',fontsize=6,
                          rotation=330,ha='center',va='center')
            
        ###############################################################################
        cbar_ax1 = fig.add_axes([0.36,0.11,0.3,0.03])                
        cbar1 = fig.colorbar(cs1,cax=cbar_ax1,orientation='horizontal',
                            extend=xx,extendfrac=0.07,drawedges=False)
        cbar1.set_label(label,fontsize=9,color='darkgrey',labelpad=1.4)  
        cbar1.set_ticks(barlim)
        cbar1.set_ticklabels(list(map(str,barlim)))
        cbar1.ax.tick_params(axis='x', size=.01,labelsize=5)
        cbar1.outline.set_edgecolor('darkgrey')
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.85,wspace=0.02,hspace=0.00,bottom=0.14)
        
        plt.savefig(directoryfigure + 'Mean-%s.png' % saveData,dpi=300)

###############################################################################
###############################################################################
###############################################################################     
        #######################################################################
        #######################################################################
        #######################################################################
        ### Plot subplot of model trends        
        if variq == 'T2M':
            limit = np.arange(-0.50,0.501,0.025)
            barlim = np.round(np.arange(-0.50,0.501,0.25),2)
            cmap = cmocean.cm.balance
            label = r'\textbf{%s -- [$^{\circ}$C PER DECADE] -- 1950-2019}' % variq
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
            
            ax1 = plt.subplot(2,4,r+1)
            m = Basemap(projection='moll',lon_0=0,resolution='l',area_thresh=10000)
            m.drawcoastlines(color='dimgrey',linewidth=0.27)
                
            var, lons_cyclic = addcyclic(var, lons)
            var, lons_cyclic = shiftgrid(180., var, lons_cyclic, start=False)
            lon2d, lat2d = np.meshgrid(lons_cyclic, lats)
            x, y = m(lon2d, lat2d)
               
            if r == 0:
                circle = m.drawmapboundary(fill_color='k',color='k',
                                  linewidth=3)
                circle.set_clip_on(False)
            else:
                circle = m.drawmapboundary(fill_color='k',color='k',
                                  linewidth=0.7)
                circle.set_clip_on(False)
            
            cs1 = m.contourf(x,y,var,limit,extend='both')
            cs1.set_cmap(cmap) 
                    
            ax1.annotate(r'\textbf{%s}' % allDataLabels[r],xy=(0,0),xytext=(0.5,1.10),
                          textcoords='axes fraction',color='w',fontsize=8,
                          rotation=0,ha='center',va='center')
            ax1.annotate(r'\textbf{[%s]}' % letters[r],xy=(0,0),xytext=(0.86,0.97),
                          textcoords='axes fraction',color='k',fontsize=6,
                          rotation=330,ha='center',va='center')
            
        ###############################################################################
        cbar_ax1 = fig.add_axes([0.36,0.11,0.3,0.03])                
        cbar1 = fig.colorbar(cs1,cax=cbar_ax1,orientation='horizontal',
                            extend='both',extendfrac=0.07,drawedges=False)
        cbar1.set_label(label,fontsize=9,color='darkgrey',labelpad=1.4)  
        cbar1.set_ticks(barlim)
        cbar1.set_ticklabels(list(map(str,barlim)))
        cbar1.ax.tick_params(axis='x', size=.01,labelsize=5)
        cbar1.outline.set_edgecolor('darkgrey')
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.85,wspace=0.02,hspace=0.00,bottom=0.14)
        
        plt.savefig(directoryfigure + 'Trend-%s.png' % saveData,dpi=300)
        
###############################################################################
###############################################################################
###############################################################################     
        #######################################################################
        #######################################################################
        #######################################################################
        ### Plot subplot of model trends        
        if variq == 'T2M':
            limit = np.arange(-1,1.01,0.05)
            barlim = np.round(np.arange(-1,1.01,0.25),2)
            cmap = cmocean.cm.balance
            label = r'\textbf{%s -- [$^{\circ}$C PER DECADE] -- 1950-2019}' % variq
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
            
            ax1 = plt.subplot(2,4,r+1)
            m = Basemap(projection='npstere',boundinglat=65,lon_0=0,
                        resolution='l',round =True,area_thresh=10000)
            m.drawcoastlines(color='dimgrey',linewidth=0.27)
                
            var, lons_cyclic = addcyclic(var, lons)
            var, lons_cyclic = shiftgrid(180., var, lons_cyclic, start=False)
            lon2d, lat2d = np.meshgrid(lons_cyclic, lats)
            x, y = m(lon2d, lat2d)
               
            if r == 0:
                circle = m.drawmapboundary(fill_color='k',color='k',
                                  linewidth=3)
                circle.set_clip_on(False)
            else:
                circle = m.drawmapboundary(fill_color='k',color='k',
                                  linewidth=0.7)
                circle.set_clip_on(False)
            
            cs1 = m.contourf(x,y,var,limit,extend='both')
            cs1.set_cmap(cmap) 
                    
            ax1.annotate(r'\textbf{%s}' % allDataLabels[r],xy=(0,0),xytext=(0.5,1.10),
                          textcoords='axes fraction',color='w',fontsize=8,
                          rotation=0,ha='center',va='center')
            ax1.annotate(r'\textbf{[%s]}' % letters[r],xy=(0,0),xytext=(0.86,0.97),
                          textcoords='axes fraction',color='k',fontsize=6,
                          rotation=330,ha='center',va='center')
            
        ###############################################################################
        cbar_ax1 = fig.add_axes([0.36,0.11,0.3,0.03])                
        cbar1 = fig.colorbar(cs1,cax=cbar_ax1,orientation='horizontal',
                            extend='both',extendfrac=0.07,drawedges=False)
        cbar1.set_label(label,fontsize=9,color='darkgrey',labelpad=1.4)  
        cbar1.set_ticks(barlim)
        cbar1.set_ticklabels(list(map(str,barlim)))
        cbar1.ax.tick_params(axis='x', size=.01,labelsize=5)
        cbar1.outline.set_edgecolor('darkgrey')
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.85,wspace=0.02,hspace=0.00,bottom=0.14)
        
        plt.savefig(directoryfigure + 'Trend-Arctic-%s.png' % saveData,dpi=300)
        
###############################################################################
###############################################################################
###############################################################################     
        #######################################################################
        #######################################################################
        #######################################################################
        ### Plot subplot of model biases      
        if variq == 'T2M':
            limit = np.arange(-6,6.01,0.25)
            barlim = np.round(np.arange(-6,7,2),2)
            cmap = cmocean.cm.balance
            label = r'\textbf{%s -- [$^{\circ}$C bias] -- 1950-2019}' % variq
        elif variq == 'P':
            limit = np.arange(-3,3.01,0.01)
            barlim = np.round(np.arange(-3,3.1,1),2)
            cmap = cmocean.cm.tarn                                                                                                                                  
            label = r'\textbf{%s -- [mm/day bias] -- 1950-2019}' % variq
        elif variq == 'SLP':
            limit = np.arange(-5,5.1,0.25)
            barlim = np.round(np.arange(-5,6,1),2)
            cmap = cmocean.cm.diff
            label = r'\textbf{%s -- [hPa bias] -- 1950-2019}' % variq
        
        fig = plt.figure(figsize=(8,4))
        for r in range(len(biasall)):
            var = biasall[r]
            
            ax1 = plt.subplot(2,4,r+2)
            m = Basemap(projection='moll',lon_0=0,resolution='l',area_thresh=10000)
            m.drawcoastlines(color='dimgrey',linewidth=0.27)
                
            var, lons_cyclic = addcyclic(var, lons)
            var, lons_cyclic = shiftgrid(180., var, lons_cyclic, start=False)
            lon2d, lat2d = np.meshgrid(lons_cyclic, lats)
            x, y = m(lon2d, lat2d)
            circle = m.drawmapboundary(fill_color='white',color='darkgray',
                              linewidth=0.7)
            circle.set_clip_on(False)
            
            cs1 = m.contourf(x,y,var,limit,extend='both')
            cs1.set_cmap(cmap) 
                    
            ax1.annotate(r'\textbf{%s}' % modelGCMs[r],xy=(0,0),xytext=(0.5,1.10),
                          textcoords='axes fraction',color='w',fontsize=8,
                          rotation=0,ha='center',va='center')
            ax1.annotate(r'\textbf{[%s]}' % letters[r],xy=(0,0),xytext=(0.86,0.97),
                          textcoords='axes fraction',color='k',fontsize=6,
                          rotation=330,ha='center',va='center')
            
        ###############################################################################
        cbar_ax1 = fig.add_axes([0.36,0.11,0.3,0.03])                
        cbar1 = fig.colorbar(cs1,cax=cbar_ax1,orientation='horizontal',
                            extend='both',extendfrac=0.07,drawedges=False)
        cbar1.set_label(label,fontsize=9,color='darkgrey',labelpad=1.4)  
        cbar1.set_ticks(barlim)
        cbar1.set_ticklabels(list(map(str,barlim)))
        cbar1.ax.tick_params(axis='x', size=.01,labelsize=5)
        cbar1.outline.set_edgecolor('darkgrey')
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.85,wspace=0.02,hspace=0.00,bottom=0.14)
        
        plt.savefig(directoryfigure + 'Bias-%s.png' % saveData,dpi=300)
        
###############################################################################
###############################################################################
###############################################################################     
        #######################################################################
        #######################################################################
        #######################################################################
        ### Plot subplot of model ensemble spread    
        if variq == 'T2M':
            limit = np.arange(0,6.01,0.25)
            barlim = np.round(np.arange(0,7,2),2)
            cmap = cmocean.cm.ice_r
            label = r'\textbf{%s -- [$^{\circ}$C spread] -- 1950-2019}' % variq
        elif variq == 'P':
            limit = np.arange(0,5.01,0.01)
            barlim = np.round(np.arange(0,5.1,1),2)
            cmap = cmocean.cm.ice_r                                                                                                                                 
            label = r'\textbf{%s -- [mm/day spread] -- 1950-2019}' % variq
        elif variq == 'SLP':
            limit = np.arange(0,10.1,0.25)
            barlim = np.round(np.arange(0,11,1),2)
            cmap = cmocean.cm.ice_r
            label = r'\textbf{%s -- [hPa spread] -- 1950-2019}' % variq
        
        fig = plt.figure(figsize=(8,4))
        for r in range(len(spreadall)):
            var = spreadall[r]
            
            ax1 = plt.subplot(2,4,r+2)
            m = Basemap(projection='moll',lon_0=0,resolution='l',area_thresh=10000)
            m.drawcoastlines(color='dimgrey',linewidth=0.27)
                
            var, lons_cyclic = addcyclic(var, lons)
            var, lons_cyclic = shiftgrid(180., var, lons_cyclic, start=False)
            lon2d, lat2d = np.meshgrid(lons_cyclic, lats)
            x, y = m(lon2d, lat2d)
            circle = m.drawmapboundary(fill_color='white',color='k',
                              linewidth=0.7)
            circle.set_clip_on(False)
            
            cs1 = m.contourf(x,y,var,limit,extend='max')
            cs1.set_cmap(cmap) 
                    
            ax1.annotate(r'\textbf{%s}' % modelGCMs[r],xy=(0,0),xytext=(0.5,1.10),
                          textcoords='axes fraction',color='w',fontsize=8,
                          rotation=0,ha='center',va='center')
            ax1.annotate(r'\textbf{[%s]}' % letters[r],xy=(0,0),xytext=(0.86,0.97),
                          textcoords='axes fraction',color='k',fontsize=6,
                          rotation=330,ha='center',va='center')
            
        ###############################################################################
        cbar_ax1 = fig.add_axes([0.36,0.11,0.3,0.03])                
        cbar1 = fig.colorbar(cs1,cax=cbar_ax1,orientation='horizontal',
                            extend='max',extendfrac=0.07,drawedges=False)
        cbar1.set_label(label,fontsize=9,color='darkgrey',labelpad=1.4)  
        cbar1.set_ticks(barlim)
        cbar1.set_ticklabels(list(map(str,barlim)))
        cbar1.ax.tick_params(axis='x', size=.01,labelsize=5)
        cbar1.outline.set_edgecolor('darkgrey')
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.85,wspace=0.02,hspace=0.00,bottom=0.14)
        
        plt.savefig(directoryfigure + 'Spread-%s.png' % saveData,dpi=300)
        
###############################################################################
###############################################################################
###############################################################################     
        #######################################################################
        #######################################################################
        #######################################################################
        ### Plot subplot of model ensemble spread    
        if variq == 'T2M':
            limit = np.arange(-1,1.01,0.05)
            barlim = np.round(np.arange(-1,1.1,0.25),2)
            cmap = cmocean.cm.balance
            label = r'\textbf{%s -- [$^{\circ}$C correlation] -- 1950-2019}' % variq
        elif variq == 'P':
            limit = np.arange(-1,1.01,0.05)
            barlim = np.round(np.arange(-1,1.1,0.25),2)
            cmap = cmocean.cm.balance                                                                                                                                
            label = r'\textbf{%s -- [mm/day correlation] -- 1950-2019}' % variq
        elif variq == 'SLP':
            limit = np.arange(-1,1.01,0.05)
            barlim = np.round(np.arange(-1,1.1,0.25),2)
            cmap = cmocean.cm.balance
            label = r'\textbf{%s -- [hPa correlation] -- 1950-2019}' % variq
        
        fig = plt.figure(figsize=(8,4))
        for r in range(len(corrall)):
            var = corrall[r]
            varp = pall[r]
            
            ax1 = plt.subplot(2,4,r+2)
            m = Basemap(projection='moll',lon_0=0,resolution='l',area_thresh=10000)
            m.drawcoastlines(color='navy',linewidth=0.34)
                
            var, lons_cyclic = addcyclic(var, lons)
            var, lons_cyclic = shiftgrid(180., var, lons_cyclic, start=False)
            varp, lons_cyclic = addcyclic(varp, lons)
            varp, lons_cyclic = shiftgrid(180., varp, lons_cyclic, start=False)
            lon2d, lat2d = np.meshgrid(lons_cyclic, lats)
            x, y = m(lon2d, lat2d)
            circle = m.drawmapboundary(fill_color='k',color='k',
                              linewidth=0.7)
            circle.set_clip_on(False)
            
            cs1 = m.contourf(x,y,var,limit,extend='neither')
            cs2 = m.contourf(x,y,varp,colors='None',hatches=['.....'])
            cs1.set_cmap(cmap) 
                    
            ax1.annotate(r'\textbf{%s}' % modelGCMs[r],xy=(0,0),xytext=(0.5,1.10),
                          textcoords='axes fraction',color='w',fontsize=8,
                          rotation=0,ha='center',va='center')
            ax1.annotate(r'\textbf{[%s]}' % letters[r],xy=(0,0),xytext=(0.86,0.97),
                          textcoords='axes fraction',color='k',fontsize=6,
                          rotation=330,ha='center',va='center')
            
        ###############################################################################
        cbar_ax1 = fig.add_axes([0.36,0.11,0.3,0.03])                
        cbar1 = fig.colorbar(cs1,cax=cbar_ax1,orientation='horizontal',
                            extend='neither',extendfrac=0.07,drawedges=False)
        cbar1.set_label(label,fontsize=9,color='darkgrey',labelpad=1.4)  
        cbar1.set_ticks(barlim)
        cbar1.set_ticklabels(list(map(str,barlim)))
        cbar1.ax.tick_params(axis='x', size=.01,labelsize=5)
        cbar1.outline.set_edgecolor('darkgrey')
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.85,wspace=0.02,hspace=0.00,bottom=0.14)
        
        plt.savefig(directoryfigure + 'Corr-%s.png' % saveData,dpi=300)