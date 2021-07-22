"""
Script for comparing differences between obs and MMmean/GFDL-CM3

Author     : Zachary M. Labe
Date       : 19 July 2021
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
modelGCMsq = ['CanESM2','MPI','CSIROMK36','KNMIecearth','GFDLCM3','GFDLESM2M','LENS','MMMean']
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
glo = True
vv = 0
mo = 0
variq = variables[vv]
monthlychoice = monthlychoiceq[mo]
directoryfigure = '/Users/zlabe/Desktop/ModelComparison_v1/Climatologies/Arctic/'
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

### Add mmean
mmmean = np.nanmean(models,axis=0)[np.newaxis,:,:,:,:]
models = np.append(models,mmmean,axis=0)

### Remove annual mean from each map
if glo == True:
    models,obs = dSS.remove_annual_mean(models,obs,lats,lons,lats_obs,lons_obs)

### Pick MMmean
mmmeanq = 7
pickmodelnamemmm = modelGCMs[mmmeanq]
mmm = models[mmmeanq,:,:,:,:]

### Pick GFDL-CM3
gfcmq = 4
pickmodelnamegfcm = modelGCMs[gfcmq]
gfc = models[gfcmq,:,:,:,:]

### Assemble new anomalies
data = np.append(gfc[np.newaxis,:,:,:,:],mmm[np.newaxis,:,:,:,:],axis=0)

diff = obs[np.newaxis,np.newaxis,:,:,:] - data
ensdiff = np.nanmean(diff[:,:,:,:,:],axis=1)

### Calculate difference in biases between decade
plotdec = []
for i in range(ensdiff.shape[0]):
    for j in range(0,ensdiff.shape[1],10):
        ave = np.nanmean(ensdiff[i,j:j+10,:,:],axis=0)
        plotdec.append(ave)
        
### Difference in bias from last 10 years
last10 = np.nanmean(ensdiff[:,-10:,:,:],axis=1) - np.nanmean(ensdiff[:,:-10,:,:],axis=1)

### Take the difference between MM and GFDL
diffBetweenModels = np.nanmean(gfc[:,:,:,:],axis=0) - np.nanmean(mmm[:,:,:,:],axis=0)
diffBetweenModelsDec = []
for i in range(0,diffBetweenModels.shape[0],10):
    avetemp = np.nanmean(diffBetweenModels[i:i+10,:,:],axis=0)
    diffBetweenModelsDec.append(avetemp)

###############################################################################
###############################################################################
###############################################################################
### Plot subplot of bias for each decade
if glo == True:
    label = r'\textbf{[[GLO]T2M-BIAS of OBS minus GCM : $^{\circ}$C]}'
    limit = np.arange(-3,3.01,0.05)
    barlim = np.round(np.arange(-3,4,1),2)
    cmap = cmocean.cm.balance
else:
    label = r'\textbf{[T2M-BIAS of OBS minus GCM : $^{\circ}$C]}'
    limit = np.arange(-5,5.01,0.05)
    barlim = np.round(np.arange(-5,6,1),2)
    cmap = cmocean.cm.balance

fig = plt.figure(figsize=(10,3))
for r in range(len(plotdec)):
    var = plotdec[r]
    
    ax1 = plt.subplot(2,len(plotdec)//2,r+1)
    m = Basemap(projection='npstere',boundinglat=61.3,lon_0=0,
                resolution='l',round =True,area_thresh=10000)
    m.drawcoastlines(color='darkgrey',linewidth=0.3)
        
    var, lons_cyclic = addcyclic(var, lons)
    var, lons_cyclic = shiftgrid(180., var, lons_cyclic, start=False)
    lon2d, lat2d = np.meshgrid(lons_cyclic, lats)
    x, y = m(lon2d, lat2d)
       
    circle = m.drawmapboundary(fill_color='dimgrey',color='dimgray',
                      linewidth=0.7)
    circle.set_clip_on(False)
    
    cs1 = m.contourf(x,y,var,limit,extend='both')
    cs1.set_cmap(cmap) 
         
    if r < 7:
        ax1.annotate(r'\textbf{%s-%s}' % (yearsall[0]+(10*r),yearsall[0]+(10*(r+1))),xy=(0,0),xytext=(0.5,1.13),
                      textcoords='axes fraction',color='dimgrey',fontsize=8,
                      rotation=0,ha='center',va='center')
    ax1.annotate(r'\textbf{[%s]}' % letters[r],xy=(0,0),xytext=(0.86,0.97),
                  textcoords='axes fraction',color='k',fontsize=6,
                  rotation=330,ha='center',va='center')
    
ax1.annotate(r'\textbf{GFDL-CM3}',xy=(0,0),xytext=(-6.3,1.60),
              textcoords='axes fraction',color='k',fontsize=10,
              rotation=90,ha='center',va='center')
ax1.annotate(r'\textbf{MMMean}',xy=(0,0),xytext=(-6.3,0.5),
              textcoords='axes fraction',color='k',fontsize=10,
              rotation=90,ha='center',va='center')
    
###############################################################################
cbar_ax1 = fig.add_axes([0.36,0.11,0.3,0.03])                
cbar1 = fig.colorbar(cs1,cax=cbar_ax1,orientation='horizontal',
                    extend='both',extendfrac=0.07,drawedges=False)
cbar1.set_label(label,fontsize=9,color='dimgrey',labelpad=0.7)  
cbar1.set_ticks(barlim)
cbar1.set_ticklabels(list(map(str,barlim)))
cbar1.ax.tick_params(axis='x', size=.01,labelsize=5)
cbar1.outline.set_edgecolor('dimgrey')

plt.tight_layout()
plt.subplots_adjust(hspace=0.01,wspace=0.02,bottom=0.17,top=0.90)

if glo == True:
    plt.savefig(directoryfigure + 'DecadeBiasComparison_GFDL-MMM_GLO.png',dpi=300)
else:
    plt.savefig(directoryfigure + 'DecadeBiasComparison_GFDL-MMM.png',dpi=300)
    
###############################################################################
###############################################################################
###############################################################################
### Plot subplot of bias for each decade
if glo == True:
    label = r'\textbf{[[GLO]T2M-DIFF of GFDL minus MMM : $^{\circ}$C]}'
    limit = np.arange(-3,3.01,0.05)
    barlim = np.round(np.arange(-3,4,1),2)
    cmap = cmocean.cm.balance
else:
    label = r'\textbf{[T2M-DIFF of GFDL minus MMM : $^{\circ}$C]}'
    limit = np.arange(-5,5.01,0.05)
    barlim = np.round(np.arange(-5,6,1),2)
    cmap = cmocean.cm.balance

fig = plt.figure(figsize=(10,2))
for r in range(len(diffBetweenModelsDec)):
    var = diffBetweenModelsDec[r]
    
    ax1 = plt.subplot(1,len(diffBetweenModelsDec),r+1)
    m = Basemap(projection='npstere',boundinglat=61.3,lon_0=0,
                resolution='l',round =True,area_thresh=10000)
    m.drawcoastlines(color='darkgrey',linewidth=0.3)
        
    var, lons_cyclic = addcyclic(var, lons)
    var, lons_cyclic = shiftgrid(180., var, lons_cyclic, start=False)
    lon2d, lat2d = np.meshgrid(lons_cyclic, lats)
    x, y = m(lon2d, lat2d)
       
    circle = m.drawmapboundary(fill_color='dimgrey',color='dimgray',
                      linewidth=0.7)
    circle.set_clip_on(False)
    
    cs1 = m.contourf(x,y,var,limit,extend='both')
    cs1.set_cmap(cmap) 
         
    if r < 7:
        ax1.annotate(r'\textbf{%s-%s}' % (yearsall[0]+(10*r),yearsall[0]+(10*(r+1))),xy=(0,0),xytext=(0.5,1.13),
                      textcoords='axes fraction',color='dimgrey',fontsize=8,
                      rotation=0,ha='center',va='center')
    ax1.annotate(r'\textbf{[%s]}' % letters[r],xy=(0,0),xytext=(0.86,0.97),
                  textcoords='axes fraction',color='k',fontsize=6,
                  rotation=330,ha='center',va='center')
    
###############################################################################
cbar_ax1 = fig.add_axes([0.36,0.13,0.3,0.03])                
cbar1 = fig.colorbar(cs1,cax=cbar_ax1,orientation='horizontal',
                    extend='both',extendfrac=0.07,drawedges=False)
cbar1.set_label(label,fontsize=9,color='dimgrey',labelpad=0.7)  
cbar1.set_ticks(barlim)
cbar1.set_ticklabels(list(map(str,barlim)))
cbar1.ax.tick_params(axis='x', size=.01,labelsize=5)
cbar1.outline.set_edgecolor('dimgrey')

plt.tight_layout()
plt.subplots_adjust(hspace=0.01,wspace=0.02,bottom=0.17,top=0.90)

if glo == True:
    plt.savefig(directoryfigure + 'Decade_GFDLminusMMM_GLO.png',dpi=300)
else:
    plt.savefig(directoryfigure + 'Decade_GFDLminusMMM.png',dpi=300)
    
###############################################################################
###############################################################################
###############################################################################
### Plot difference in last 10 years of every model for the bias
if glo == True:
    label = r'\textbf{[[GLO]T2M-BIAS of OBS minus GCM - last 10 years : $^{\circ}$C]}'
    limit = np.arange(-2,2.01,0.05)
    barlim = np.round(np.arange(-2,3,1),2)
    cmap = cmocean.cm.balance
else:
    label = r'\textbf{[T2M-BIAS of OBS minus GCM - last 10 years : $^{\circ}$C]}'
    limit = np.arange(-2,2.01,0.05)
    barlim = np.round(np.arange(-2,3,1),2)
    cmap = cmocean.cm.balance

fig = plt.figure()
for r in range(len(last10)):
    var = last10[r]
    
    ax1 = plt.subplot(1,2,r+1)
    m = Basemap(projection='npstere',boundinglat=61.3,lon_0=0,
                resolution='l',round =True,area_thresh=10000)
    m.drawcoastlines(color='darkgrey',linewidth=0.3)
        
    var, lons_cyclic = addcyclic(var, lons)
    var, lons_cyclic = shiftgrid(180., var, lons_cyclic, start=False)
    lon2d, lat2d = np.meshgrid(lons_cyclic, lats)
    x, y = m(lon2d, lat2d)
       
    circle = m.drawmapboundary(fill_color='dimgrey',color='dimgray',
                      linewidth=0.7)
    circle.set_clip_on(False)
    
    cs1 = m.contourf(x,y,var,limit,extend='both')
    cs1.set_cmap(cmap) 
         
    ax1.annotate(r'\textbf{[%s]}' % letters[r],xy=(0,0),xytext=(0.86,0.97),
                  textcoords='axes fraction',color='k',fontsize=6,
                  rotation=330,ha='center',va='center')
    if r == 0:    
        ax1.annotate(r'\textbf{GFDL-CM3}',xy=(0,0),xytext=(0.5,1.05),
                  textcoords='axes fraction',color='k',fontsize=10,
                  rotation=0,ha='center',va='center')
    elif r == 1:
        ax1.annotate(r'\textbf{MMMean}',xy=(0,0),xytext=(0.5,1.05),
                      textcoords='axes fraction',color='k',fontsize=10,
                      rotation=0,ha='center',va='center')
    
###############################################################################
cbar_ax1 = fig.add_axes([0.36,0.11,0.3,0.03])                
cbar1 = fig.colorbar(cs1,cax=cbar_ax1,orientation='horizontal',
                    extend='both',extendfrac=0.07,drawedges=False)
cbar1.set_label(label,fontsize=9,color='dimgrey',labelpad=0.7)  
cbar1.set_ticks(barlim)
cbar1.set_ticklabels(list(map(str,barlim)))
cbar1.ax.tick_params(axis='x', size=.01,labelsize=5)
cbar1.outline.set_edgecolor('dimgrey')

plt.tight_layout()
plt.subplots_adjust(hspace=0.01,wspace=0.02,bottom=0.17,top=0.90)

if glo == True:
    plt.savefig(directoryfigure + 'Last10-BiasComparison_GFDL-MMM_GLO.png',dpi=300)
else:
    plt.savefig(directoryfigure + 'Last10-BiasComparison_GFDL-MMM.png',dpi=300)