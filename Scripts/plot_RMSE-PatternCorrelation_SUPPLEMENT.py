"""
Script for plotting RMSE and pattern correlations in the Arctc

Author     : Zachary M. Labe
Date       : 4 March 2022
Version    : 1
"""

### Import packages
import sys
import matplotlib.pyplot as plt
import numpy as np
import palettable.cubehelix as cm
import palettable.scientific.sequential as sss
import palettable.cartocolors.qualitative as cc
import cmocean as cmocean
import cmasher as cmr
import calc_Utilities as UT
import scipy.stats as sts

### Plotting defaults 
plt.rc('text',usetex=True)
plt.rc('font',**{'family':'sans-serif','sans-serif':['Avant Garde']}) 

### Paramters
letters = ["a","b","c","d","e","f","g","h","i","j","k","l","m"]
datasetsingle = ['SMILE']
dataset_obs = ['ERA5BE']
monthlychoiceq = ['annual']
typeOfCorr = ['R']
variables = ['T2M']
reg_name = 'Arctic'
level = 'surface'
timeper = 'historical'
option = 7
land_only = False
ocean_only = False

if timeper == 'historical':
    years = np.arange(1950,2019+1,1)
if reg_name == 'SMILEGlobe':
    region = 'Global'
elif reg_name == 'narrowTropics':
    region = 'Tropics'
elif reg_name == 'Arctic':
    region = 'Arctic'
elif reg_name == 'SouthernOcean':
    region = 'Southern Ocean'
elif reg_name == 'LowerArctic':
    region = 'Arctic'

variq = variables[0]
monthlychoice = monthlychoiceq[0]
directorydata = '/Users/zlabe/Documents/Research/ModelComparison/Data/Climatologies/'
directoryfigure = '/Users/zlabe/Desktop/ModelComparison_v1/v7/'
saveData =  monthlychoice + '_' + variq + '_' + reg_name + '_' + timeper
if land_only == True:
    saveData =  monthlychoice + '_LAND_' + variq + '_' + reg_name + '_' + timeper
    typemask = 'LAND'
elif ocean_only == True:
    saveData =  monthlychoice + '_OCEAN_' + variq + '_' + reg_name + '_' + timeper
    typemask = 'OCEAN'
else:
    typemask = 'LAND/OCEAN'
print('*Filename == < %s >' % saveData) 

corr = np.load(directorydata + saveData + '_corrs.npz')['arr_0'][:]
rmse = np.load(directorydata + saveData + '_RMSE.npz')['arr_0'][:]
modelGCMs = ['CanESM2','MPI','CSIRO-MK3.6','EC-EARTH','GFDL-CM3','GFDL-ESM2M','LENS'][:]

### Ensemble mean correlations
correns = np.nanmean(corr,axis=1)
rmseens = np.nanmean(rmse,axis=1)

### Pick highest correlation for each year
corryr = np.argmax(correns,axis=0)
rmseyr = np.argmax(rmseens,axis=0)

### Counts of data for each year
uniquecorryr,countcorryr = np.unique(corryr,return_counts=True)
uniquermseyr,countrmseyr = np.unique(rmseyr,return_counts=True)

###############################################################################
###############################################################################
###############################################################################
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
           
fig = plt.figure(figsize=(10,4))
ax = plt.subplot(121)
adjust_spines(ax, ['left', 'bottom'])
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')
ax.spines['left'].set_color('dimgrey')
ax.spines['bottom'].set_color('dimgrey')
ax.spines['left'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)
ax.tick_params('both',length=4,width=2,which='major',color='dimgrey')
ax.yaxis.grid(zorder=1,color='dimgrey',alpha=0.35,clip_on=False)

color = cmocean.cm.phase_r(np.linspace(0.00,1,len(modelGCMs)))
for i,c in zip(range(len(modelGCMs)),color):
    if i == 6:
        c = 'k'
    else:
        c = c
    plt.plot(years,correns[i],color=c,linewidth=2.3,
                label=r'\textbf{%s}' % modelGCMs[i],zorder=11,
                clip_on=False,alpha=1)

leg = plt.legend(shadow=False,fontsize=13,loc='upper center',
              bbox_to_anchor=(1.10,1.16),fancybox=True,ncol=7,frameon=False,
              handlelength=0,handletextpad=0)
for line,text in zip(leg.get_lines(), leg.get_texts()):
    text.set_color(line.get_color())
    
plt.ylabel(r'\textbf{Pattern Correlation Coefficient}',color='dimgrey',fontsize=11)
plt.text(1950,0.99,r'\textbf{[a]}',fontsize=9,color='k')

plt.xticks(np.arange(1950,2030+1,10),map(str,np.arange(1950,2030+1,10)),size=9)
plt.yticks(np.arange(0,1.01,0.05),map(str,np.round(np.arange(0,1.01,0.05),3)),size=9)
plt.xlim([1950,2020])   
plt.ylim([0.7,1.0])

###############################################################################
###############################################################################
###############################################################################   
ax = plt.subplot(122)

adjust_spines(ax, ['left', 'bottom'])
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')
ax.spines['left'].set_color('dimgrey')
ax.spines['bottom'].set_color('dimgrey')
ax.spines['left'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)
ax.tick_params('both',length=4,width=2,which='major',color='dimgrey')
ax.yaxis.grid(zorder=1,color='dimgrey',alpha=0.35,clip_on=False)

color = cmocean.cm.phase_r(np.linspace(0.00,1,len(modelGCMs)))
for i,c in zip(range(len(modelGCMs)),color):
    if i == 6:
        c = 'k'
    else:
        c = c
    plt.plot(years,rmseens[i],color=c,linewidth=2.3,
                label=r'\textbf{%s}' % modelGCMs[i],zorder=11,
                clip_on=False,alpha=1)

plt.xticks(np.arange(1950,2030+1,10),map(str,np.arange(1950,2030+1,10)),size=9)
plt.yticks(np.arange(0,11.5,1),map(str,np.round(np.arange(0,11,1),3)),size=9)
plt.xlim([1950,2020])   
plt.ylim([0,8])   

plt.ylabel(r'\textbf{RMSE [$^{\circ}$C]}',color='dimgrey',fontsize=11)    
plt.text(1950,7.73,r'\textbf{[b]}',fontsize=9,color='k')     

plt.savefig(directoryfigure + 'RMSE-PatternCorrelation_SUPPLEMENT',dpi=900)