"""
Script for creating composites to compare the points correlations of the MMLEA

Author     : Zachary M. Labe
Date       : 4 March 2022
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
import matplotlib
import cmasher as cmr

### Plotting defaults 
plt.rc('text',usetex=True)
plt.rc('font',**{'family':'sans-serif','sans-serif':['Avant Garde']}) 

###############################################################################
###############################################################################
###############################################################################
### Data preliminaries 
modelGCMs = ['CanESM2','MPI','CSIRO-MK3.6','KNMI-ecearth','GFDL-CM3','GFDL-ESM2M','LENS']
dataset_obs = 'ERA5BE'
allDataLabels = ['CanESM2','MPI','CSIRO-MK3.6','EC-EARTH','GFDL-CM3','GFDL-ESM2M','LENS']
letters = ["a","b","c","d","e","f","g","h","i","j","k","l","m","n"]
datasetsingle = ['SMILE']
monthlychoiceq = ['annual']
variables = ['T2M']
reg_name = 'Arctic'
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
for vv in range(1):
    for mo in range(1):
        variq = variables[vv]
        monthlychoice = monthlychoiceq[mo]
        directorydata = '/Users/zlabe/Documents/Research/ModelComparison/Data/Climatologies/'
        directoryfigure = '/Users/zlabe/Desktop/ModelComparison_v1/v7/'
        saveData =  monthlychoice + '_' + variq + '_' + reg_name + '_' + timeper
        print('*Filename == < %s >' % saveData) 
        
        corr = np.load(directorydata + saveData + '_PointByPoint_corrs.npz')['arr_0']
        lats = np.load(directorydata + saveData + '_PointByPoint_lats.npz')['arr_0']
        lons = np.load(directorydata + saveData + '_PointByPoint_lons.npz')['arr_0']

###############################################################################
###############################################################################
###############################################################################     
        #######################################################################
        #######################################################################
        #######################################################################
        ### Plot subplot of mean climate
        if variq == 'T2M':
            limit = np.arange(-1,1.01,0.1)
            barlim = np.round(np.arange(-1,1.1,0.5),2)
            cmap = cmocean.cm.balance
            label = r'\textbf{Correlation Coefficient}'
            
        fig = plt.figure(figsize=(8,2))
        for r in range(0,len(allDataLabels)):
            var = corr[r]
            
            ax1 = plt.subplot(1,7,r+1)
            m = Basemap(projection='npstere',boundinglat=61.5,lon_0=0,
                        resolution='l',round =True,area_thresh=10000)
            m.drawcoastlines(color='dimgrey',linewidth=0.5)
                
            var, lons_cyclic = addcyclic(var, lons)
            var, lons_cyclic = shiftgrid(180., var, lons_cyclic, start=False)
            lon2d, lat2d = np.meshgrid(lons_cyclic, lats)
            x, y = m(lon2d, lat2d)
               
            circle = m.drawmapboundary(fill_color='white',color='dimgrey',linewidth=0.7)
            circle.set_clip_on(False)
            
            if variq == 'P':
                xx = 'max'
            else:
                xx = 'both'
            cs1 = m.contourf(x,y,var,limit,extend=xx)
            cs1.set_cmap(cmap) 
                    
            ax1.annotate(r'\textbf{%s}' % allDataLabels[r],xy=(0,0),xytext=(0.5,1.15),
                          textcoords='axes fraction',color='k',fontsize=9,
                          rotation=0,ha='center',va='center')
            ax1.annotate(r'\textbf{[%s]}' % letters[r],xy=(0,0),
                    xytext=(0.85,0.89),xycoords='axes fraction',rotation=330,
                    color='dimgrey',fontsize=6)
            
        ###############################################################################
        cbar_ax1 = fig.add_axes([0.395,0.15,0.2,0.035])                
        cbar1 = fig.colorbar(cs1,cax=cbar_ax1,orientation='horizontal',
                            extend=xx,extendfrac=0.07,drawedges=False)
        cbar1.set_label(label,fontsize=9,color='k',labelpad=1.4)  
        cbar1.set_ticks(barlim)
        cbar1.set_ticklabels(list(map(str,barlim)))
        cbar1.ax.tick_params(axis='x', size=.001,labelsize=7,pad=2)
        cbar1.outline.set_edgecolor('dimgrey')
        
        plt.tight_layout()
        plt.savefig(directoryfigure + 'PointCorrelations-%s_SUPPLEMENT.png' % saveData,dpi=900)
        