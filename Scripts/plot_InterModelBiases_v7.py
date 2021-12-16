"""
Script for plotting differences in models for select variables over the 
1950-2019 period

Author     : Zachary M. Labe
Date       : 15 December 2021
Version    : 7 - adds validation data for early stopping
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
modelGCMs = ['CanESM2','MPI','CSIRO-MK3.6','KNMI-ecearth','GFDL-CM3','GFDL-ESM2M','LENS']
modelGCMsNames = ['CanESM2','MPI','CSIRO-MK3.6','KNMI-ecearth','GFDL-CM3','GFDL-ESM2M','LENS','MMmean']
letters = ["a","b","c","d","e","f","g","h","i","j","k","l","m"]
datasetsingle = ['SMILE']
monthlychoiceq = ['JFM','AMJ','JAS','OND','annual']
variables = ['T2M','P','SLP']
reg_name = 'SMILEGlobe'
level = 'surface'
monthlychoiceq = ['annual']
variables = ['T2M']
timeper = 'historical'
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
randomalso = False
shuffletype = 'none'
###############################################################################
###############################################################################
###############################################################################
###############################################################################
### Read in model data
def read_primary_dataset(variq,dataset,monthlychoice,numOfEns,lensalso,randomalso,ravelyearsbinary,ravelbinary,shuffletype,timeper,lat_bounds=lat_bounds,lon_bounds=lon_bounds):
    data,lats,lons = df.readFiles(variq,dataset,monthlychoice,numOfEns,lensalso,randomalso,ravelyearsbinary,ravelbinary,shuffletype,timeper)
    datar,lats,lons = df.getRegion(data,lats,lons,lat_bounds,lon_bounds)
    print('\nOur dataset: ',dataset,' is shaped',data.shape)
    return datar,lats,lons

### Call functions
for vv in range(len(variables)):
    for mo in range(len(monthlychoiceq)):
        variq = variables[vv]
        monthlychoice = monthlychoiceq[mo]
        directoryfigure = '/Users/zlabe/Desktop/ModelComparison_v1/Climatologies/interModel/%s/' % variq
        saveData =  monthlychoice + '_' + variq + '_' + reg_name
        print('*Filename == < %s >' % saveData) 
    
        ### Read data
        models,lats,lons = read_primary_dataset(variq,dataset,monthlychoice,numOfEns,
                                                lensalso,randomalso,ravelyearsbinary,
                                                ravelbinary,shuffletype,timeper,
                                                lat_bounds,lon_bounds)
        
        ### Calculate ensemble mean
        ensmean = np.nanmean(models[:,:,:,:,:],axis=1)
        
        ### Calculate multimodel mean
        modmean = np.nanmean(models[:,:,:,:,:],axis=0)
        
        ### Calculate difference from multimodelmean
        diffmod = models - modmean
        diffmodensm = np.nanmean(diffmod[:,:,:,:,:],axis=1)
        diffmodmean = np.nanmean(diffmodensm[:,:,:,:],axis=1)
        
        ### Calculate different between each model
        # intermodel = np.empty((models.shape[0],models.shape[0],models.shape[1],
        #                         models.shape[2],models.shape[3],models.shape[4]))
        # for mm in range(models.shape[0]):
        #     for ea in range(models.shape[0]):
        #         intermodel[mm,ea,:,:,:,:] = models[mm,:,:,:,:] - models[ea,:,:,:,:]
        # ensmeanintermodel = np.nanmean(intermodel[:,:,:,:,:,:],axis=2)
        # timeensmeanintermodel = np.nanmean(ensmeanintermodel[:,:,:,:,:],axis=2)
                              
###############################################################################
###############################################################################
###############################################################################     
        #######################################################################
        #######################################################################
        #######################################################################
        ### Plot subplot of different from multimodel mean  
        if variq == 'T2M':
            limit = np.arange(-6,6.01,0.25)
            barlim = np.round(np.arange(-6,7,2),2)
            cmap = cmocean.cm.balance
            label = r'\textbf{%s -- [$^{\circ}$C MMmean difference] -- 1950-2019}' % variq
        elif variq == 'P':
            limit = np.arange(-3,3.01,0.01)
            barlim = np.round(np.arange(-3,3.1,1),2)
            cmap = cmocean.cm.tarn                                                                                                                                  
            label = r'\textbf{%s -- [mm/day MMmean difference] -- 1950-2019}' % variq
        elif variq == 'SLP':
            limit = np.arange(-5,5.1,0.25)
            barlim = np.round(np.arange(-5,6,1),2)
            cmap = cmocean.cm.diff
            label = r'\textbf{%s -- [hPa MMmean difference] -- 1950-2019}' % variq
        
        fig = plt.figure(figsize=(8,4))
        for r in range(len(diffmodmean)):
            var = diffmodmean[r]
            
            ax1 = plt.subplot(2,4,r+2)
            m = Basemap(projection='moll',lon_0=0,resolution='l',area_thresh=10000)
            m.drawcoastlines(color='dimgrey',linewidth=0.27)
                
            var, lons_cyclic = addcyclic(var, lons)
            var, lons_cyclic = shiftgrid(180., var, lons_cyclic, start=False)
            lon2d, lat2d = np.meshgrid(lons_cyclic, lats)
            x, y = m(lon2d, lat2d)
            circle = m.drawmapboundary(fill_color='white',color='dimgray',
                              linewidth=0.7)
            circle.set_clip_on(False)
            
            cs1 = m.contourf(x,y,var,limit,extend='both')
            cs1.set_cmap(cmap) 
                    
            ax1.annotate(r'\textbf{%s}' % modelGCMs[r],xy=(0,0),xytext=(0.5,1.10),
                          textcoords='axes fraction',color='dimgrey',fontsize=8,
                          rotation=0,ha='center',va='center')
            ax1.annotate(r'\textbf{[%s]}' % letters[r],xy=(0,0),xytext=(0.86,0.97),
                          textcoords='axes fraction',color='k',fontsize=6,
                          rotation=330,ha='center',va='center')
            
        ###############################################################################
        cbar_ax1 = fig.add_axes([0.36,0.11,0.3,0.03])                
        cbar1 = fig.colorbar(cs1,cax=cbar_ax1,orientation='horizontal',
                            extend='both',extendfrac=0.07,drawedges=False)
        cbar1.set_label(label,fontsize=9,color='dimgrey',labelpad=1.4)  
        cbar1.set_ticks(barlim)
        cbar1.set_ticklabels(list(map(str,barlim)))
        cbar1.ax.tick_params(axis='x', size=.01,labelsize=5)
        cbar1.outline.set_edgecolor('dimgrey')
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.85,wspace=0.02,hspace=0.00,bottom=0.14)
        
        plt.savefig(directoryfigure + 'MultiModelBias-%s_ALL.png' % saveData,dpi=300)
        directorydataMS = '/Users/zlabe/Documents/Research/ModelComparison/Data/RevisitResults_v7/'
        np.save(directorydataMS + 'MMMeandifferences_7models.npy',diffmodmean)
        
        ###############################################################################
        ###############################################################################
        ###############################################################################
        fig = plt.figure(figsize=(10,2))
        for r in range(len(diffmodmean)+1):
            if r < 7:
                var = diffmodmean[r]
            else:
                var = np.empty((lats.shape[0],lons.shape[0]))
                var[:] = np.nan
            
            ax1 = plt.subplot(1,len(diffmodmean)+1,r+1)
            m = Basemap(projection='npstere',boundinglat=65,lon_0=0,
                        resolution='l',round =True,area_thresh=10000)
            m.drawcoastlines(color='darkgrey',linewidth=0.27)
                
            var, lons_cyclic = addcyclic(var, lons)
            var, lons_cyclic = shiftgrid(180., var, lons_cyclic, start=False)
            lon2d, lat2d = np.meshgrid(lons_cyclic, lats)
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
        cbar_ax1 = fig.add_axes([0.36,0.13,0.3,0.03])                
        cbar1 = fig.colorbar(cs1,cax=cbar_ax1,orientation='horizontal',
                            extend='both',extendfrac=0.07,drawedges=False)
        cbar1.set_label(label,fontsize=9,color='dimgrey',labelpad=1.4)  
        cbar1.set_ticks(barlim)
        cbar1.set_ticklabels(list(map(str,barlim)))
        cbar1.ax.tick_params(axis='x', size=.01,labelsize=5)
        cbar1.outline.set_edgecolor('dimgrey')
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.85,wspace=0.02,hspace=0.02,bottom=0.14)
        
        plt.savefig(directoryfigure + 'MultiModelBias-%s_ALL-Arctic.png' % saveData,dpi=300)
        
        ###############################################################################
        ###############################################################################
        ###############################################################################
        if variq == 'T2M':
            limit = np.arange(-3,3.01,0.2)
            barlim = np.round(np.arange(-3,4,1),2)
            cmap = cmocean.cm.balance
            label = r'\textbf{%s -- [$^{\circ}$C MMmean difference] -- 1950-2019}' % variq
        elif variq == 'P':
            limit = np.arange(-3,3.01,0.01)
            barlim = np.round(np.arange(-3,3.1,1),2)
            cmap = cmocean.cm.tarn                                                                                                                                  
            label = r'\textbf{%s -- [mm/day MMmean difference] -- 1950-2019}' % variq
        elif variq == 'SLP':
            limit = np.arange(-5,5.1,0.25)
            barlim = np.round(np.arange(-5,6,1),2)
            cmap = cmocean.cm.diff
            label = r'\textbf{%s -- [hPa MMmean difference] -- 1950-2019}' % variq
        
        fig = plt.figure(figsize=(10,2))
        for r in range(len(diffmodmean)+1):
            if r < 7:
                var = diffmodmean[r]
            else:
                var = np.empty((lats.shape[0],lons.shape[0]))
                var[:] = np.nan
            
            ax1 = plt.subplot(1,len(diffmodmean)+1,r+1)
            m = Basemap(projection='moll',lon_0=0,resolution='l',area_thresh=10000)
            m.drawcoastlines(color='darkgrey',linewidth=0.27)
                
            var, lons_cyclic = addcyclic(var, lons)
            var, lons_cyclic = shiftgrid(180., var, lons_cyclic, start=False)
            lon2d, lat2d = np.meshgrid(lons_cyclic, lats)
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
        cbar_ax1 = fig.add_axes([0.36,0.13,0.3,0.03])                
        cbar1 = fig.colorbar(cs1,cax=cbar_ax1,orientation='horizontal',
                            extend='both',extendfrac=0.07,drawedges=False)
        cbar1.set_label(label,fontsize=9,color='dimgrey',labelpad=1.4)  
        cbar1.set_ticks(barlim)
        cbar1.set_ticklabels(list(map(str,barlim)))
        cbar1.ax.tick_params(axis='x', size=.01,labelsize=5)
        cbar1.outline.set_edgecolor('dimgrey')
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.85,wspace=0.02,hspace=0.02,bottom=0.14)
        
        plt.savefig(directoryfigure + 'MultiModelBias-%s_ALL-StyleGlobe.png' % saveData,dpi=300)
        
        ###############################################################################
        ###############################################################################
        ###############################################################################
        if variq == 'T2M':
            limit = np.arange(-3,3.01,0.2)
            barlim = np.round(np.arange(-3,4,1),2)
            cmap = cmocean.cm.balance
            label = r'\textbf{%s -- [$^{\circ}$C MMmean difference] -- 1950-2019}' % variq
        elif variq == 'P':
            limit = np.arange(-3,3.01,0.01)
            barlim = np.round(np.arange(-3,3.1,1),2)
            cmap = cmocean.cm.tarn                                                                                                                                  
            label = r'\textbf{%s -- [mm/day MMmean difference] -- 1950-2019}' % variq
        elif variq == 'SLP':
            limit = np.arange(-5,5.1,0.25)
            barlim = np.round(np.arange(-5,6,1),2)
            cmap = cmocean.cm.diff
            label = r'\textbf{%s -- [hPa MMmean difference] -- 1950-2019}' % variq
        
        fig = plt.figure(figsize=(10,2))
        for r in range(len(diffmodmean)+1):
            if r < 7:
                var = diffmodmean[r]
            else:
                var = np.empty((lats.shape[0],lons.shape[0]))
                var[:] = np.nan
                
            latq = np.where((lats >= -20) & (lats <= 20))[0]
            latsqq = lats[latq]
            var = var[latq,:]
            
            ax1 = plt.subplot(1,len(diffmodmean)+1,r+1)
            m = Basemap(projection='moll',lon_0=0,resolution='l',area_thresh=10000)
            m.drawcoastlines(color='darkgrey',linewidth=0.27)
                
            var, lons_cyclic = addcyclic(var, lons)
            var, lons_cyclic = shiftgrid(180., var, lons_cyclic, start=False)
            lon2d, lat2d = np.meshgrid(lons_cyclic, latsqq)
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
        cbar_ax1 = fig.add_axes([0.36,0.13,0.3,0.03])                
        cbar1 = fig.colorbar(cs1,cax=cbar_ax1,orientation='horizontal',
                            extend='both',extendfrac=0.07,drawedges=False)
        cbar1.set_label(label,fontsize=9,color='dimgrey',labelpad=1.4)  
        cbar1.set_ticks(barlim)
        cbar1.set_ticklabels(list(map(str,barlim)))
        cbar1.ax.tick_params(axis='x', size=.01,labelsize=5)
        cbar1.outline.set_edgecolor('dimgrey')
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.85,wspace=0.02,hspace=0.02,bottom=0.14)
        
        plt.savefig(directoryfigure + 'MultiModelBias-%s_ALL-Tropics.png' % saveData,dpi=300)
        
###############################################################################
###############################################################################
###############################################################################     
        #######################################################################
        #######################################################################
        #######################################################################
        ### Plot subplot of different from multimodel mean  
        if variq == 'T2M':
            limit = np.arange(-6,6.01,0.25)
            barlim = np.round(np.arange(-6,7,2),2)
            cmap = cmocean.cm.balance
            label = r'\textbf{%s -- [$^{\circ}$C difference] -- 1950-2019}' % variq
        elif variq == 'P':
            limit = np.arange(-3,3.01,0.01)
            barlim = np.round(np.arange(-3,3.1,1),2)
            cmap = cmocean.cm.tarn                                                                                                                                  
            label = r'\textbf{%s -- [mm/day difference] -- 1950-2019}' % variq
        elif variq == 'SLP':
            limit = np.arange(-5,5.1,0.25)
            barlim = np.round(np.arange(-5,6,1),2)
            cmap = cmocean.cm.diff
            label = r'\textbf{%s -- [hPa difference] -- 1950-2019}' % variq
        
        for diff in range(timeensmeanintermodel.shape[0]):
            fig = plt.figure(figsize=(8,4))
            for r in range(timeensmeanintermodel.shape[1]+1):
                var = timeensmeanintermodel[diff,r-1,:,:]
                
                ax1 = plt.subplot(2,4,r+1)
                if r == 0:
                    varc = np.nanmean(ensmean[diff],axis=0) # average over years
                    latsc = lats.copy()
                    lonsc = lons.copy()
                    if variq == 'T2M':
                        limitc = np.arange(-35,35.01,0.5)
                        barlimc = np.round(np.arange(-35,36,5),2)
                        cmapc = plt.cm.CMRmap_r
                        labelc = r'\textbf{%s -- [$^{\circ}$C mean] -- 1950-2019}' % variq
                    elif variq == 'P':
                        limitc = np.arange(0,10.01,0.01)
                        barlimc = np.round(np.arange(-0,10.01,2),2)
                        cmapc = cmocean.cm.rain                                                                                                                                 
                        labelc = r'\textbf{%s -- [mm/day mean] -- 1950-2019}' % variq
                    elif variq == 'SLP':
                        limitc = np.arange(990,1020.01,0.5)
                        barlimc = np.round(np.arange(990,1021,10),2)
                        cmapc = sss.Nuuk_20.mpl_colormap 
                        labelc = r'\textbf{%s -- [hPa : mean] -- 1950-2019}' % variq
                        
                    m = Basemap(projection='moll',lon_0=0,resolution='l',area_thresh=10000)
                    m.drawcoastlines(color='dimgrey',linewidth=0.27)
                        
                    varc, lons_cyclicc = addcyclic(varc, lonsc)
                    varc, lons_cyclicc = shiftgrid(180., varc, lons_cyclicc, start=False)
                    lon2dc, lat2dc = np.meshgrid(lons_cyclicc, latsc)
                    xc, yc = m(lon2dc, lat2dc)
                    circle = m.drawmapboundary(fill_color='white',color='dimgray',
                                      linewidth=0.7)
                    circle.set_clip_on(False)
                    
                    cs1 = m.contourf(xc,yc,varc,limitc,extend='both')
                    cs1.set_cmap(cmapc) 
                            
                    ax1.annotate(r'\textbf{%s Climatology}' % modelGCMs[diff],xy=(0,0),xytext=(0.5,1.10),
                                  textcoords='axes fraction',color='dimgrey',fontsize=8,
                                  rotation=0,ha='center',va='center')
                    ax1.annotate(r'\textbf{[%s]}' % letters[r],xy=(0,0),xytext=(0.86,0.97),
                                  textcoords='axes fraction',color='k',fontsize=6,
                                  rotation=330,ha='center',va='center')
                else:
                    m = Basemap(projection='moll',lon_0=0,resolution='l',area_thresh=10000)
                    m.drawcoastlines(color='dimgrey',linewidth=0.27)
                        
                    var, lons_cyclic = addcyclic(var, lons)
                    var, lons_cyclic = shiftgrid(180., var, lons_cyclic, start=False)
                    lon2d, lat2d = np.meshgrid(lons_cyclic, lats)
                    x, y = m(lon2d, lat2d)
                    circle = m.drawmapboundary(fill_color='white',color='dimgray',
                                      linewidth=0.7)
                    circle.set_clip_on(False)
                    
                    cs1 = m.contourf(x,y,var,limit,extend='both')
                    cs1.set_cmap(cmap) 
                            
                    ax1.annotate(r'\textbf{%s}' % modelGCMs[r-1],xy=(0,0),xytext=(0.5,1.10),
                                  textcoords='axes fraction',color='dimgrey',fontsize=8,
                                  rotation=0,ha='center',va='center')
                    ax1.annotate(r'\textbf{[%s]}' % letters[r],xy=(0,0),xytext=(0.86,0.97),
                                  textcoords='axes fraction',color='k',fontsize=6,
                                  rotation=330,ha='center',va='center')
            
            
            ###############################################################################
            fig.suptitle(r'\textbf{%s minus each SMILE}' % modelGCMs[diff],color='k',
                      fontsize=15)
                
            cbar_ax1 = fig.add_axes([0.36,0.11,0.3,0.03])                
            cbar1 = fig.colorbar(cs1,cax=cbar_ax1,orientation='horizontal',
                                extend='both',extendfrac=0.07,drawedges=False)
            cbar1.set_label(label,fontsize=9,color='dimgrey',labelpad=1.4)  
            cbar1.set_ticks(barlim)
            cbar1.set_ticklabels(list(map(str,barlim)))
            cbar1.ax.tick_params(axis='x', size=.01,labelsize=5)
            cbar1.outline.set_edgecolor('dimgrey')
            
            plt.tight_layout()
            plt.subplots_adjust(top=0.85,wspace=0.02,hspace=0.00,bottom=0.14)
            
            plt.savefig(directoryfigure + 'InterBias-%s_%s.png' % (saveData,modelGCMs[diff]),dpi=300)
        