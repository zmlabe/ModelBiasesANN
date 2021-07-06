"""
Script for plotting science communication graph of the global mean surface
temperature anomalies

Author     : Zachary M. Labe
Date       : 9 June 2021
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
import palettable.cartocolors.qualitative as cc
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
modelGCMs = ['CanESM2','MPI','CSIRO-MK3.6','KNMI-ECearth','GFDL-CM3','GFDL-ESM2M','LENS']
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

### Call functions
vv = 0
mo = 0
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

### Calculate global mean temperature Anomalies
lon2,lat2 = np.meshgrid(lons,lats)
modelsmanom = UT.calc_weightedAve(modelanom,lat2)
obsmanom = UT.calc_weightedAve(obsanom,lat2)

###############################################################################          
### Calculate ensemble spread statistics
meaensanom = np.nanmean(modelsmanom[:,:,:],axis=1)
maxensanom = np.nanmax(modelsmanom[:,:,:],axis=1)
minensanom = np.nanmin(modelsmanom[:,:,:],axis=1)
spreadanom = maxensanom - minensanom

### Calculate global mean temperature
lon2,lat2 = np.meshgrid(lons,lats)
modelsm = UT.calc_weightedAve(models,lat2)
obsm = UT.calc_weightedAve(obs,lat2)

###############################################################################          
### Calculate ensemble spread statistics
meaens = np.nanmean(modelsm[:,:,:],axis=1)
maxens = np.nanmax(modelsm[:,:,:],axis=1)
minens = np.nanmin(modelsm[:,:,:],axis=1)
spread = maxens - minens

###############################################################################
###############################################################################
###############################################################################
### Create time series
fig = plt.figure(figsize=(9,3))
ax = plt.subplot(111)

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

adjust_spines(ax, ['left', 'bottom'])
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')
ax.spines['left'].set_color('darkgrey')
ax.spines['bottom'].set_color('darkgrey')
ax.spines['left'].set_linewidth(0)
ax.spines['bottom'].set_linewidth(2)
ax.tick_params('both',length=4,width=2,which='major',color='darkgrey')
ax.tick_params(axis='x',labelsize=20,pad=4)

color = plt.cm.CMRmap(np.linspace(0.15,0.95,len(modelGCMs)))
for r,c in zip(range(len(modelGCMs)),color):
    if r == 6:
        c = 'w'
    else:
        c = c
    plt.plot(yearsall,meaens[r,:],'-',
                color=c,linewidth=3,clip_on=False,alpha=1)
    ax.fill_between(yearsall,minens[r,:],maxens[r,:],facecolor=c,alpha=0.25,zorder=1,clip_on=False)

# plt.plot(years,ensanom_lens,'-',
#                 color='crimson',linewidth=0.8,clip_on=False,alpha=1)
# plt.plot(yearsobs,globe_obs,color='w',linewidth=2,
#           dashes=(1,0.3),linestyle='--',label=r'\textbf{20CRv3}',zorder=11)

# plt.ylabel(r'\textbf{Temperature Anomaly [$^{\circ}$C]}',fontsize=10,color='darkgrey')
plt.yticks(np.arange(12,17,1),map(str,np.round(np.arange(13,17,1),2)))
plt.xticks(np.arange(1950,2019+1,69),map(str,np.arange(1950,2019+1,69)))
plt.xlim([1950,2019])   
plt.ylim([12.5,15.84])
ax.axes.yaxis.set_visible(False)

plt.subplots_adjust(bottom=0.15)
plt.savefig(directoryfigure + 'TimeSeries_MeanGlobalTemperature_SMILE_CLEAR.png',
            dpi=300)

###############################################################################
###############################################################################
###############################################################################
### Create time series
fig = plt.figure(figsize=(9,2))
ax = plt.subplot(111)

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

adjust_spines(ax, ['left', 'bottom'])
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')
ax.spines['left'].set_color('dimgrey')
ax.spines['bottom'].set_color('dimgrey')
ax.spines['left'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)
ax.tick_params('both',length=4,width=2,which='major',color='dimgrey')
ax.tick_params(axis='x',labelsize=15,pad=4)
ax.tick_params(axis='y',labelsize=15,pad=4)

color = plt.cm.CMRmap(np.linspace(0.18,1,len(modelGCMs)))
for r,c in zip(range(len(modelGCMs)),color):
    if r == 6:
        c = 'k'
    else:
        c = c
    plt.plot(yearsall,meaensanom[r,:],'-',
                color=c,linewidth=3,clip_on=False,alpha=1)
    ax.fill_between(yearsall,minensanom[r,:],maxensanom[r,:],facecolor=c,alpha=0.25,zorder=1,clip_on=False)

# plt.plot(years,ensanom_lens,'-',
#                 color='crimson',linewidth=0.8,clip_on=False,alpha=1)
# plt.plot(yearsobs,globe_obs,color='w',linewidth=2,
#           dashes=(1,0.3),linestyle='--',label=r'\textbf{20CRv3}',zorder=11)

# plt.ylabel(r'\textbf{Temperature Anomaly [$^{\circ}$C]}',fontsize=10,color='darkgrey')
plt.yticks(np.arange(-0.5,2.1,2.5),map(str,np.round(np.arange(-0.5,2.1,2.5),2)))
plt.xticks(np.arange(1950,2019+1,69),map(str,np.arange(1950,2019+1,69)))
plt.xlim([1950,2019])   
plt.ylim([-0.5,2])
# ax.axes.yaxis.set_visible(False)

plt.subplots_adjust(bottom=0.2)
plt.savefig(directoryfigure + 'TimeSeries_MeanGlobalTemperatureAnomalies_SMILE_CLEAR.png',
            dpi=600)