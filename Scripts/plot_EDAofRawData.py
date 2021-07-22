"""
Script for playing with the raw data of obs and climate models

Author     : Zachary M. Labe
Date       : 29 April 2021
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
modelGCMs = ['CanESM2','MPI','CSIRO-MK3.6','KNMI-ecearth','GFDL-CM3','GFDL-ESM2M','LENS']
modelGCMsMM = ['CanESM2','MPI','CSIRO-MK3.6','KNMI-ecearth','GFDL-CM3','GFDL-ESM2M','LENS','MMean']
modelGCMsMM2 = ['ERA5BE','CanESM2','MPI','CSIRO-MK3.6','KNMI-ecearth','GFDL-CM3','GFDL-ESM2M','LENS','MMmean','LENSmean']
letters = ["a","b","c","d","e","f","g","h","i","j","k","l","m"]
datasetsingle = ['SMILE']
dataset_obs = 'ERA5BE'
monthlychoiceq = ['annual','JFM','AMJ','JAS','OND']
variables = ['T2M','P','SLP']
reg_name = 'LowerArctic'
level = 'surface'
timeper = 'historical'
directoryfigure = '/Users/zlabe/Desktop/ModelComparison_v1/Climatologies/interModel/%s/' % variables[0]
###############################################################################
###############################################################################
land_only = False
ocean_only = False
###############################################################################
###############################################################################
if timeper == 'historical': 
    baseline = np.arange(1951,1980+1,1)
elif timeper == 'future':
    baseline = np.arange(2021,2050+1,1)
###############################################################################
###############################################################################
window = 0
if timeper == 'historical': 
    yearsall = np.arange(1950+window,2019+1,1)
elif timeper == 'future':
    yearsall = np.arange(2020+window,2099+1,1)
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
def read_obs_dataset(variq,dataset_obs,numOfEns,lensalso,randomalso,ravelyearsbinary,ravelbinary,shuffletype,lat_bounds=lat_bounds,lon_bounds=lon_bounds):
    data_obs,lats_obs,lons_obs = df.readFiles(variq,dataset_obs,monthlychoice,numOfEns,lensalso,randomalso,ravelyearsbinary,ravelbinary,shuffletype,timeper)
    data_obs,lats_obs,lons_obs = df.getRegion(data_obs,lats_obs,lons_obs,
                                            lat_bounds,lon_bounds)    
    print('our OBS dataset: ',dataset_obs,' is shaped',data_obs.shape)
    return data_obs,lats_obs,lons_obs
def calcTrend(data):
    slopes = np.empty((data.shape[1],data.shape[2]))
    x = np.arange(data.shape[0])
    for i in range(data.shape[1]):
        for j in range(data.shape[2]):
            mask = np.isfinite(data[:,i,j])
            y = data[:,i,j]
            
            if np.sum(mask) == y.shape[0]:
                xx = x
                yy = y
            else:
                xx = x[mask]
                yy = y[mask]      
            if np.isfinite(np.nanmean(yy)):
                slopes[i,j],intercepts, \
                r_value,p_value,std_err = sts.linregress(xx,yy)
            else:
                slopes[i,j] = np.nan
    
    dectrend = slopes * 10.   
    print('Completed: Finished calculating trends!')      
    return dectrend
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

###############################################################################
###############################################################################
###############################################################################
###############################################################################
### Call functions
# variq = variables[0]
# monthlychoice = monthlychoiceq[0]
# directorydata = '/Users/zlabe/Documents/Research/ModelComparison/Data/Climatologies/'
# directoryfigure = '/Users/zlabe/Desktop/ModelComparison_v1/Climatologies/patternCorr/%s/' % variq
# saveData =  monthlychoice + '_' + variq + '_' + reg_name + '_' + timeper
# print('*Filename == < %s >' % saveData) 

# ### Read data
# models,lats,lons = read_primary_dataset(variq,dataset,monthlychoice,numOfEns,
#                                         lensalso,randomalso,ravelyearsbinary,
#                                         ravelbinary,shuffletype,timeper,
#                                         lat_bounds,lon_bounds)
# data_obs,lats_obs,lons_obs = read_obs_dataset(variq,dataset_obs,numOfEns,
#                                               lensalso,randomalso,ravelyearsbinary,
#                                               ravelbinary,shuffletype,lat_bounds,
#                                               lon_bounds)

# ### Only 70 years so through 2090 if future (2020-2089)
# if timeper == 'future':
#     models = models[:,:,:70,:,:]
#     yearsall = np.arange(2020,2089+1,1)
#     baseline = np.arange(2021,2050+1,1)

# ### Add on additional "model" which is a multi-model mean
# modelmean = np.nanmean(models,axis=0)
# modelmeanq = modelmean[np.newaxis,:,:,:,:]
# modelsall = np.append(models,modelmeanq,axis=0)

# ### Meshgrid of lat/lon
# lon2,lat2 = np.meshgrid(lons,lats)

# ### Additional subsample of a model for testing
# newmodels = models.copy()
# testens = newmodels[6,:,:,:,:] # 6 for LENS

# newmodeltest = np.empty(testens.shape)
# for sh in range(testens.shape[0]):
#     ensnum = np.arange(models.shape[1])
#     slices = np.random.choice(ensnum,size=models.shape[0],replace=False)
#     slicenewmodel = np.nanmean(testens[slices,:,:,:],axis=0)
#     newmodeltest[sh,:,:,:] = slicenewmodel

# ###############################################################################
# ###############################################################################
# ###############################################################################
# ### Calculate average standard deviation across years
# mmstd = np.nanstd(modelmean,axis=1)
# enstd = np.nanstd(newmodeltest,axis=1)
# alstd = np.nanstd(models,axis=2)
# obstdyr = np.nanstd(data_obs,axis=0)

# ### Calculate average standard deviation across ensemble members
# mmstdyr = np.nanmean(mmstd,axis=0)
# enstdyr = np.nanmean(enstd,axis=0)
# alstdyr = np.nanmean(alstd,axis=1)

# ### Global mean std across years and ensembles
# mmave = UT.calc_weightedAve(mmstdyr,lat2)
# enave = UT.calc_weightedAve(enstdyr,lat2)
# alave = UT.calc_weightedAve(alstdyr,lat2)
# obave = UT.calc_weightedAve(obstdyr,lat2)

# ### Global mean std across years for EACH ensemble
# mmaveens = UT.calc_weightedAve(mmstd,lat2)
# enaveens = UT.calc_weightedAve(enstd,lat2)
# alaveens = UT.calc_weightedAve(alstd,lat2)

# ### Maximum ensemble
# mmmaxens = np.max(mmaveens)
# enmaxens = np.max(enaveens)
# almaxens = np.max(alaveens,axis=1)
# maxall = np.concatenate((almaxens,[mmmaxens],[enmaxens]))

# ### Minimum ensemble
# mmminens = np.min(mmaveens)
# enminens = np.min(enaveens)
# alminens = np.min(alaveens,axis=1)
# minall = np.concatenate((alminens,[mmminens],[enminens]))

# ### Standard deviation of ensemble member sampling
# mmstdOfEnsOnly = np.nanstd(mmaveens)
# entdOfEnsOnly = np.nanstd(enaveens)
# alstdOfEnsOnly = np.nanstd(alaveens,axis=1)

### Add all data
stdall = np.concatenate(([obave],alave,[mmave],[enave]))

###############################################################################
###############################################################################
###############################################################################
fig = plt.figure()
ax = plt.subplot(111)
adjust_spines(ax, ['left', 'bottom'])
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')
ax.spines['left'].set_color('dimgrey')
ax.spines['bottom'].set_color('dimgrey')
ax.spines['left'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(0)
ax.tick_params('y',length=4,width=2,which='major',color='dimgrey')
ax.tick_params('x',length=0,width=0,which='major',color='dimgrey')

rects = plt.bar(np.arange(len(stdall)),stdall, align='center')
plt.axhline(y=obave,linestyle='--',linewidth=2,color='dimgrey',
            clip_on=False,zorder=100,dashes=(1,0.3))
for ii in range(1,len(stdall)-1):
    plt.errorbar(ii,stdall[ii],
                     yerr=np.array([[stdall[ii]-minall[ii-1],maxall[ii-1]-stdall[ii]]]).T,
                     color='k',linewidth=2,capthick=3,capsize=0)

### Set color
for i in range(len(stdall)):
    rects[i].set_color('deepskyblue')
    rects[-2].set_color('crimson')
    rects[i].set_edgecolor('w')

plt.xticks(np.arange(0,len(stdall)+1,1),modelGCMsMM2,size=5)
plt.yticks(np.arange(0,2.1,0.1),map(str,np.round(np.arange(0,2.1,0.1),2)),size=6)
plt.xlim([-0.5,10])   
plt.ylim([0,2])
plt.ylabel(r'\textbf{STDev-%s -- [$^{\circ}$C] -- 1950-2019}' % variq)

plt.tight_layout()
plt.savefig(directoryfigure + 'BarGraphOfSTD-%s_ALL.png' % saveData,dpi=300)

# ###############################################################################
# ###############################################################################
# ###############################################################################     
# #######################################################################
# #######################################################################
# #######################################################################
# ### Plot subplot of different from multimodel mean  
# if variq == 'T2M':
#     limit = np.arange(-0.75,0.76,0.01)
#     barlim = np.round(np.arange(-0.75,0.76,0.25),2)
#     cmap = cmocean.cm.balance
#     if timeper == 'historical':
#         label = r'\textbf{STD-%s -- [$^{\circ}$C MMmean difference] -- 1950-2019}' % variq
#     elif timeper == 'future':
#         label = r'\textbf{STD-%s -- [$^{\circ}$C MMmean difference] -- 2020-2089}' % variq
# elif variq == 'P':
#     limit = np.arange(-3,3.01,0.01)
#     barlim = np.round(np.arange(-3,3.1,1),2)
#     cmap = cmocean.cm.tarn   
#     if timeper == 'historical':                                                                                                                               
#         label = r'\textbf{STD-%s -- [mm/day MMmean difference] -- 1950-2019}' % variq
#     elif timeper == 'future':
#         label = r'\textbf{STD-%s -- [mm/day MMmean difference] -- 2020-2089}' % variq
# elif variq == 'SLP':
#     limit = np.arange(-2,2.1,0.2)
#     barlim = np.round(np.arange(-2,3,1),2)
#     cmap = cmocean.cm.diff
#     if timeper == 'historical':
#         label = r'\textbf{STD-%s -- [hPa MMmean difference] -- 1950-2019}' % variq
#     elif timeper == 'future':
#         label = r'\textbf{STD-%s -- [hPa MMmean difference] -- 2020-2089}' % variq
# fig = plt.figure(figsize=(8,4))
# for r in range(len(alstdyr)):
#     var = alstdyr[r] - mmstdyr
    
#     ax1 = plt.subplot(2,4,r+2)
#     m = Basemap(projection='moll',lon_0=0,resolution='l',area_thresh=10000)
#     m.drawcoastlines(color='dimgrey',linewidth=0.27)
        
#     var, lons_cyclic = addcyclic(var, lons)
#     var, lons_cyclic = shiftgrid(180., var, lons_cyclic, start=False)
#     lon2d, lat2d = np.meshgrid(lons_cyclic, lats)
#     x, y = m(lon2d, lat2d)
#     circle = m.drawmapboundary(fill_color='white',color='dimgray',
#                       linewidth=0.7)
#     circle.set_clip_on(False)
    
#     cs1 = m.contourf(x,y,var,limit,extend='both')
#     cs1.set_cmap(cmap) 
            
#     ax1.annotate(r'\textbf{%s}' % modelGCMs[r],xy=(0,0),xytext=(0.5,1.10),
#                   textcoords='axes fraction',color='dimgrey',fontsize=8,
#                   rotation=0,ha='center',va='center')
#     ax1.annotate(r'\textbf{[%s]}' % letters[r],xy=(0,0),xytext=(0.86,0.97),
#                   textcoords='axes fraction',color='k',fontsize=6,
#                   rotation=330,ha='center',va='center')
    
# ###############################################################################
# cbar_ax1 = fig.add_axes([0.36,0.11,0.3,0.03])                
# cbar1 = fig.colorbar(cs1,cax=cbar_ax1,orientation='horizontal',
#                     extend='both',extendfrac=0.07,drawedges=False)
# cbar1.set_label(label,fontsize=9,color='dimgrey',labelpad=1.4)  
# cbar1.set_ticks(barlim)
# cbar1.set_ticklabels(list(map(str,barlim)))
# cbar1.ax.tick_params(axis='x', size=.01,labelsize=5)
# cbar1.outline.set_edgecolor('dimgrey')

# plt.tight_layout()
# plt.subplots_adjust(top=0.85,wspace=0.02,hspace=0.00,bottom=0.14)

# plt.savefig(directoryfigure + 'STD-MultiModelBias-%s_ALL.png' % saveData,dpi=300)

# ###############################################################################
# ###############################################################################
# ###############################################################################     
# #######################################################################
# #######################################################################
# #######################################################################
# ### Plot subplot of different from observations
# if variq == 'T2M':
#     limit = np.arange(-0.75,0.76,0.01)
#     barlim = np.round(np.arange(-0.75,0.76,0.25),2)
#     cmap = cmocean.cm.balance
#     if timeper == 'historical':
#         label = r'\textbf{STD-%s -- [$^{\circ}$C OBS difference] -- 1950-2019}' % variq
#     elif timeper == 'future':
#         label = r'\textbf{STD-%s -- [$^{\circ}$C OBS difference] -- 2020-2089}' % variq
# elif variq == 'P':
#     limit = np.arange(-3,3.01,0.01)
#     barlim = np.round(np.arange(-3,3.1,1),2)
#     cmap = cmocean.cm.tarn              
#     if timeper == 'historical':                                                                                                                    
#         label = r'\textbf{STD-%s -- [mm/day OBS difference] -- 1950-2019}' % variq
#     elif timeper == 'future':
#         label = r'\textbf{STD-%s -- [mm/day OBS difference] -- 2020-2089}' % variq
# elif variq == 'SLP':
#     limit = np.arange(-2,2.1,0.2)
#     barlim = np.round(np.arange(-2,3,1),2)
#     cmap = cmocean.cm.diff
#     if timeper == 'historical':
#         label = r'\textbf{STD-%s -- [hPa OBS difference] -- 1950-2019}' % variq
#     elif timeper == 'future':
#         label = r'\textbf{STD-%s -- [hPa OBS difference] -- 2020-2089}' % variq

# totallstdyr = np.append(alstdyr,mmstdyr[np.newaxis,:,:],axis=0)

# fig = plt.figure(figsize=(8,4))
# for r in range(len(totallstdyr)):
#     var = totallstdyr[r] - obstdyr
    
#     ax1 = plt.subplot(2,4,r+1)
#     m = Basemap(projection='moll',lon_0=0,resolution='l',area_thresh=10000)
#     m.drawcoastlines(color='dimgrey',linewidth=0.27)
        
#     var, lons_cyclic = addcyclic(var, lons)
#     var, lons_cyclic = shiftgrid(180., var, lons_cyclic, start=False)
#     lon2d, lat2d = np.meshgrid(lons_cyclic, lats)
#     x, y = m(lon2d, lat2d)
#     circle = m.drawmapboundary(fill_color='white',color='dimgray',
#                       linewidth=0.7)
#     circle.set_clip_on(False)
    
#     cs1 = m.contourf(x,y,var,limit,extend='both')
#     cs1.set_cmap(cmap) 
            
#     ax1.annotate(r'\textbf{%s}' % modelGCMsMM[r],xy=(0,0),xytext=(0.5,1.10),
#                   textcoords='axes fraction',color='dimgrey',fontsize=8,
#                   rotation=0,ha='center',va='center')
#     ax1.annotate(r'\textbf{[%s]}' % letters[r],xy=(0,0),xytext=(0.86,0.97),
#                   textcoords='axes fraction',color='k',fontsize=6,
#                   rotation=330,ha='center',va='center')
    
# ###############################################################################
# cbar_ax1 = fig.add_axes([0.36,0.11,0.3,0.03])                
# cbar1 = fig.colorbar(cs1,cax=cbar_ax1,orientation='horizontal',
#                     extend='both',extendfrac=0.07,drawedges=False)
# cbar1.set_label(label,fontsize=9,color='dimgrey',labelpad=1.4)  
# cbar1.set_ticks(barlim)
# cbar1.set_ticklabels(list(map(str,barlim)))
# cbar1.ax.tick_params(axis='x', size=.01,labelsize=5)
# cbar1.outline.set_edgecolor('dimgrey')

# plt.tight_layout()
# plt.subplots_adjust(top=0.85,wspace=0.02,hspace=0.00,bottom=0.14)

# plt.savefig(directoryfigure + 'STD-MultiModelOBSBias-%s_ALL.png' % saveData,dpi=300)