"""
Plot for comparing data sets of GMST in the Arctic

Reference  : Barnes et al. [2020, JAMES]
Author     : Zachary M. Labe
Date       : 10 February 2022
Version    : 7 - adds validation data for early stopping
"""

### Import packages
import sys
import matplotlib.pyplot as plt
import numpy as np
import calc_Utilities as UT
import calc_dataFunctions as df
import calc_Stats as dSS
import scipy.stats as sts
import calc_DetrendData as DET
from netCDF4 import Dataset
import cmasher as cmr

### Plotting defaults 
plt.rc('text',usetex=True)
plt.rc('font',**{'family':'sans-serif','sans-serif':['Avant Garde']}) 

###############################################################################
###############################################################################
###############################################################################
### Data preliminaries 
directoryfigure = '/Users/zlabe/Desktop/ModelComparison_v1/v7/'
modelGCMs = ['CanESM2','MPI','CSIRO-MK3.6','KNMI-ecearth','GFDL-CM3','GFDL-ESM2M','LENS']
letters = ["a","b","c","d","e","f","g","h","i","j","k","l","m"]
datasetsingle = ['SMILE']
dataset_obsall = ['ERA5BE','20CRv3','BEST','GISTEMP','HadCRUT']
obslabels = ['ERA5-BE','20CRv3','BEST','GISTEMPv4','HadCRUT5']
monthlychoiceq = ['annual']
variables = ['T2M']
reg_name = 'Arctic'
level = 'surface'
timeper = 'historical'
###############################################################################
###############################################################################
land_only = False
ocean_only = False
###############################################################################
###############################################################################
baseline = np.arange(1981,2010+1,1)
###############################################################################
###############################################################################
window = 0
yearsall = np.arange(1950+window,2019+1,1)
yearsobs = [np.arange(1950+window,2019+1,1),np.arange(1950+window,2015+1,1),
            np.arange(1950+window,2019+1,1),np.arange(1950+window,2019+1,1),
            np.arange(1950+window,2019+1,1)]
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

###############################################################################
###############################################################################
###############################################################################
###############################################################################
### Read in model data
variq = variables[0]
monthlychoice = monthlychoiceq[0]
models,lats,lons = read_primary_dataset(variq,dataset,monthlychoice,numOfEns,
                                        lensalso,randomalso,ravelyearsbinary,
                                        ravelbinary,shuffletype,timeper,
                                        lat_bounds,lon_bounds)

### Read lat/lon grid
lon2,lat2 = np.meshgrid(lons,lats)

meanobsq = []
for vv in range(len(dataset_obsall)):
    data_obsn,lats_obs,lons_obs = read_obs_dataset(variq,dataset_obsall[vv],numOfEns,
                                                  lensalso,randomalso,ravelyearsbinary,
                                                  ravelbinary,shuffletype,lat_bounds,
                                                  lon_bounds)
    anoms,anom_obs = dSS.calculate_anomalies(models,data_obsn,
                                              lats,lons,baseline,yearsall)
    
    ### Calculate mean GMST for Arctic
    meanobs = UT.calc_weightedAve(anom_obs,lat2)
    meanmodel = UT.calc_weightedAve(anoms,lat2)
    
    meanobsq.append(meanobs)
    
### Create large ensemble
ensembleanom = meanmodel.reshape(meanmodel.shape[0]*meanmodel.shape[1],meanmodel.shape[2])

### Calculate statistics
maxt = np.percentile(ensembleanom,95,axis=0)
mint = np.percentile(ensembleanom,5,axis=0)

##############################################################################
##############################################################################
##############################################################################
fig = plt.figure()
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
        
##############################################################################
##############################################################################
##############################################################################
ax = plt.subplot(111)
adjust_spines(ax, ['left', 'bottom'])
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')
ax.spines['left'].set_color('dimgrey')
ax.spines['bottom'].set_color('dimgrey')
ax.spines['left'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)
ax.tick_params('both',length=4,width=2,which='major',color='dimgrey')
ax.tick_params(axis='x',labelsize=7,pad=4)
ax.tick_params(axis='y',labelsize=7,pad=4)

plt.fill_between(x=yearsall,y1=mint,y2=maxt,facecolor='darkgrey',zorder=0,
         alpha=0.6,edgecolor='none',clip_on=False,label=r'\textbf{MMLEA}')

color = iter(cmr.infinity(np.linspace(0.00,1,len(meanobsq))))
for i in range(len(meanobsq)):
    if i == 0:
        cma = 'k'
        ll = 3
        aa = 1
        plt.plot(yearsobs[i],meanobsq[i],color=cma,alpha=aa,linewidth=ll,clip_on=False,
             label=r'\textbf{%s}' % obslabels[i],linestyle='--',dashes=(1,0.3),
             zorder=20)
    else:
        cma=next(color)
        ll = 2
        aa = 1
        plt.plot(yearsobs[i],meanobsq[i],color=cma,alpha=aa,linewidth=ll,clip_on=False,
             label=r'\textbf{%s}' % obslabels[i])
    
leg = plt.legend(shadow=False,fontsize=9,loc='upper center',
            bbox_to_anchor=(0.48, 1.15),fancybox=True,ncol=15,frameon=False,
            handlelength=1.5,handletextpad=0.25)
for line,text in zip(leg.get_lines(), leg.get_texts()):
    text.set_color(line.get_color())
    
plt.ylabel(r'\textbf{T2M Anomaly [$^{\circ}$C]}',fontsize=8,color='dimgrey')
plt.yticks(np.arange(-4,4.5,0.5),map(str,np.round(np.arange(-4,4.5,0.5),2)),size=9)
plt.xticks(np.arange(1950,2021,10),map(str,np.arange(1950,2021,10)),size=9)
plt.xlim([1950,2020])   
plt.ylim([-3,3])
plt.subplots_adjust(bottom=0.15)
plt.savefig(directoryfigure + 'ArcticGMST_Comparison.png',dpi=600)