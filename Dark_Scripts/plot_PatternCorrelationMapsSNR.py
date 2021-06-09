"""
Script for plotting pattern correlation maps between models and observations
of SNR

Author     : Zachary M. Labe
Date       : 10 May 2021
Version    : 1
"""

### Import packages
import sys
import matplotlib.pyplot as plt
import numpy as np
import palettable.cubehelix as cm
import palettable.scientific.sequential as sss
import palettable.cartocolors.qualitative as cc
import cmasher as cmr
import cmocean as cmocean
from mpl_toolkits.basemap import Basemap, addcyclic, shiftgrid
import calc_Utilities as UT
import scipy.stats as sts

### Plotting defaults 
plt.rc('text',usetex=True)
plt.rc('font',**{'family':'sans-serif','sans-serif':['Avant Garde']}) 

### Parameters
letters = ["a","b","c","d","e","f","g","h","i","j","k","l","m"]
datasetsingle = ['SMILE']
dataset_obs = ['ERA5BE']
monthlychoiceq = ['annual','JFM','AMJ','JAS','OND']
typeOfCorr = ['R']
variables = ['T2M','P']
reg_name = 'SMILEGlobe'
level = 'surface'
timeper = 'historical'

### Read in data
for vv in range(len(variables)):
    for mo in range(len(monthlychoiceq)):
# for vv in range(1):
#     for mo in range(1):
        variq = variables[vv]
        monthlychoice = monthlychoiceq[mo]
        directorydata = '/Users/zlabe/Documents/Research/ModelComparison/Data/Climatologies/'
        directoryfigure = '/Users/zlabe/Desktop/ModelComparison_v1/Climatologies/SNR/'
        saveData =  monthlychoice + '_' + variq + '_' + reg_name + '_' + timeper
        saveDatal = 'annual_T2M_SMILEGlobe_historical' # for lat/lon data files
        print('*Filename == < %s >' % saveData) 
        
        snr = np.load(directorydata + saveData + '_modelsSNR.npz')['arr_0']
        snrobs = np.load(directorydata + saveData + '_obsSNR.npz')['arr_0']
        lats = np.load(directorydata + saveDatal + '_PointByPoint_lats.npz')['arr_0']
        lons = np.load(directorydata + saveDatal + '_PointByPoint_lons.npz')['arr_0']
        modelGCMs = ['CanESM2','MPI','CSIRO-MK3.6','KNMI-ecearth','GFDL-CM3','GFDL-ESM2M','LENS','MMmean','OBS']
        
        ### Calculate ensemble mean
        snrmean = np.nanmean(snr,axis=1)
        
        ### Add all data
        allsnr = np.append(snrmean,snrobs[np.newaxis,:,:],axis=0)

        ###############################################################################
        ###############################################################################
        ###############################################################################     
        #######################################################################
        #######################################################################
        #######################################################################
        ### Plot subplot of different from observations
        if variq == 'T2M':
            limit = np.arange(0,4.01,0.25)
            barlim = np.round(np.arange(0,5,1),2)
            cmap = cmr.eclipse    
            if timeper == 'historical':
                label = r'\textbf{%s -- [SNR] -- 1950-2019 - %s}' % (variq,monthlychoice)
            elif timeper == 'future':
                label = r'\textbf{%s -- [SNR] -- 2020-2089 - %s}' % (variq,monthlychoice)
        elif variq == 'P':
            limit = np.arange(0,5.01,0.1)
            barlim = np.round(np.arange(0,6,1),2)
            cmap = cmr.eclipse         
            if timeper == 'historical':                                                                                                                    
                label = r'\textbf{%s -- [SNR] -- 1950-2019 - %s}' % (variq,monthlychoice)
            elif timeper == 'future':
                label = r'\textbf{%s -- [SNR] -- 2020-2089 - %s}' % (variq,monthlychoice)
        
        fig = plt.figure(figsize=(8,4))
        for r in range(len(allsnr)):
            var = allsnr[r]
            
            ax1 = plt.subplot(3,3,r+1)
            m = Basemap(projection='moll',lon_0=0,resolution='l',area_thresh=10000)
            m.drawcoastlines(color='dimgrey',linewidth=0.27)
                
            var, lons_cyclic = addcyclic(var, lons)
            var, lons_cyclic = shiftgrid(180., var, lons_cyclic, start=False)
            lon2d, lat2d = np.meshgrid(lons_cyclic, lats)
            x, y = m(lon2d, lat2d)
            circle = m.drawmapboundary(fill_color='white',color='dimgray',
                              linewidth=0.7)
            circle.set_clip_on(False)
            if r == 0:
                circle = m.drawmapboundary(fill_color='white',color='k',
                                           linewidth=2)
                circle.set_clip_on(False)
            
            cs1 = m.contourf(x,y,var,limit,extend='max')
            cs1.set_cmap(cmap) 
                    
            ax1.annotate(r'\textbf{%s}' % modelGCMs[r],xy=(0,0),xytext=(0.04,0.84),
                          textcoords='axes fraction',color='dimgrey',fontsize=7,
                          rotation=40,ha='center',va='center')
            ax1.annotate(r'\textbf{[%s]}' % letters[r],xy=(0,0),xytext=(0.86,0.97),
                          textcoords='axes fraction',color='k',fontsize=6,
                          rotation=0,ha='center',va='center')
            
        ###############################################################################
        cbar_ax1 = fig.add_axes([0.36,0.09,0.3,0.03])                
        cbar1 = fig.colorbar(cs1,cax=cbar_ax1,orientation='horizontal',
                            extend='max',extendfrac=0.07,drawedges=False)
        cbar1.set_label(label,fontsize=9,color='dimgrey',labelpad=1.4)  
        cbar1.set_ticks(barlim)
        cbar1.set_ticklabels(list(map(str,barlim)))
        cbar1.ax.tick_params(axis='x', size=.01,labelsize=5)
        cbar1.outline.set_edgecolor('dimgrey')
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.85,wspace=0.00,hspace=0.02,bottom=0.14)
        
        plt.savefig(directoryfigure + 'SNR-%s_ALL.png' % saveData,dpi=300)
        
        ###############################################################################
        ###############################################################################
        ###############################################################################     
        #######################################################################
        #######################################################################
        #######################################################################
        ### Plot subplot of different from observations
        if variq == 'T2M':
            limit = np.arange(-2,2.1,0.1)
            barlim = np.round(np.arange(-2,2.1,0.5),2)
            cmap = cmr.fusion_r
            if timeper == 'historical':
                label = r'\textbf{%s -- [MMmean SNR Difference] -- 1950-2019 - %s}' % (variq,monthlychoice)
            elif timeper == 'future':
                label = r'\textbf{%s -- [MMmean SNR Difference] -- 2020-2089 - %s}' % (variq,monthlychoice)
        elif variq == 'P':
            limit = np.arange(-2,2.1,0.1)
            barlim = np.round(np.arange(-2,2.1,0.5),2)
            cmap = cmr.fusion_r        
            if timeper == 'historical':                                                                                                                    
                label = r'\textbf{%s -- [MMmean SNR Difference] -- 1950-2019 - %s}' % (variq,monthlychoice)
            elif timeper == 'future':
                label = r'\textbf{%s -- [MMmean SNR Difference] -- 2020-2089 - %s}' % (variq,monthlychoice)
        elif variq == 'SLP':
            limit = np.arange(-2,2.1,0.1)
            barlim = np.round(np.arange(-2,2.1,0.5),2)
            cmap = cmr.fusion_r
            if timeper == 'historical':
                label = r'\textbf{%s -- [MMmean SNR Difference] -- 1950-2019 - %s}' % (variq,monthlychoice)
            elif timeper == 'future':
                label = r'\textbf{%s -- [MMmean SNR Difference] -- 2020-2089 - %s}' % (variq,monthlychoice)
        
        fig = plt.figure(figsize=(8,4))
        for r in range(len(allsnr)-1):
            varq = allsnr[r]
            var = allsnr[-2,:,:] - varq
            
            ax1 = plt.subplot(2,4,r+1)
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
        
        plt.savefig(directoryfigure + 'MMDifference-PatternCorrelationsSNR-%s_ALL.png' % saveData,dpi=300)