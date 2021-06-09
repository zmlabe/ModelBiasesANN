"""
Script for plotting pattern correlations between models and observations

Author     : Zachary M. Labe
Date       : 28 April 2091
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
import calc_Utilities as UT
import scipy.stats as sts

### Plotting defaults 
plt.rc('text',usetex=True)
plt.rc('font',**{'family':'sans-serif','sans-serif':['Avant Garde']}) 

### Paramters
letters = ["a","b","c","d","e","f","g","h","i","j","k","l","m"]
datasetsingle = ['SMILE']
dataset_obs = ['ERA5BE']
monthlychoiceq = ['annual','JFM','AMJ','JAS','OND']
typeOfCorr = ['R','R-DT','R-RMGLO','R-TREND']
variables = ['T2M','P','SLP']
reg_name = 'SMILEGlobe'
level = 'surface'
timeper = 'future'
option = 2

### Read in data
for vv in range(len(variables)):
    for mo in range(len(monthlychoiceq)):
        variq = variables[vv]
        monthlychoice = monthlychoiceq[mo]
        directorydata = '/Users/zlabe/Documents/Research/ModelComparison/Data/Climatologies/'
        directoryfigure = '/Users/zlabe/Desktop/ModelComparison_v1/Climatologies/patternCorr/%s/%s/' % (variq,option)
        saveData =  monthlychoice + '_' + variq + '_' + reg_name + '_' + timeper
        print('*Filename == < %s >' % saveData) 
        
        corr = np.load(directorydata + saveData + '_corrs.npz')['arr_0'][:option]
        corrdt = np.load(directorydata + saveData + '_corrsdt.npz')['arr_0'][:option]
        corrglo = np.load(directorydata + saveData + '_corrsglo.npz')['arr_0'][:option]
        corrtrends = np.load(directorydata + saveData + '_corrstrends.npz')['arr_0'][:option]
        modelGCMs = ['CanESM2','MPI','CSIRO-MK3.6','KNMI-ecearth','GFDL-CM3','GFDL-ESM2M','LENS','MMmean'][:option]

        ### Ensemble mean correlations
        correns = np.nanmean(corr,axis=1)
        corrensdt = np.nanmean(corrdt,axis=1)
        corrensglo = np.nanmean(corrglo,axis=1)
        ensALLr = [correns,corrensdt,corrensglo]
        
        ### Pick highest correlation for each year
        corryr = np.argmax(correns,axis=0)
        corryrdt = np.argmax(corrensdt,axis=0)
        corryrglo = np.argmax(corrensglo,axis=0)
        yrALLr = [corryr,corryrdt,corryrglo]
        
        ### Counts of data for each year
        uniquecorryr,countcorryr = np.unique(corryr,return_counts=True)
        uniquecorryrdt,countcorryrdt = np.unique(corryrdt,return_counts=True)
        uniquecorryrglo,countcorryrglo = np.unique(corryrglo,return_counts=True)
        uniqueALL = [uniquecorryr,uniquecorryrdt,uniquecorryrglo]
        countALL = [countcorryr,countcorryrdt,countcorryrglo]
        
        ### Correlation for trends averaged across ensemble members
        correnstrends = np.nanmean(corrtrends,axis=1)
        
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
        
        for ii in range(len(yrALLr)):
            prediction = yrALLr[ii]
            uni = uniqueALL[ii]
            cou = countALL[ii]
                       
            fig = plt.figure(figsize=(10,5))
            ax = plt.subplot(121)
            adjust_spines(ax, ['left', 'bottom'])
            ax.spines['top'].set_color('none')
            ax.spines['right'].set_color('none')
            ax.spines['left'].set_color('dimgrey')
            ax.spines['bottom'].set_color('dimgrey')
            ax.spines['left'].set_linewidth(2)
            ax.spines['bottom'].set_linewidth(2)
            ax.tick_params('both',length=4,width=2,which='major',color='dimgrey')
            ax.yaxis.grid(zorder=1,color='dimgrey',alpha=0.35)
            
            x=np.arange(2020,2089+1,1)
            plt.scatter(x,prediction,c=prediction,s=40,clip_on=False,cmap=cc.Antique_8.mpl_colormap,
                        edgecolor='k',linewidth=0.4,zorder=10)
            
            plt.xticks(np.arange(2020,2091,5),map(str,np.arange(2020,2091,5)),size=6)
            plt.yticks(np.arange(0,12+1,1),modelGCMs,size=6)
            plt.xlim([2020,2090])   
            plt.ylim([0,option-1])
            
            plt.title(r'\textbf{FUTURE PATTERN CORRELATIONS FOR [ %s ]}' % variables[vv],color='k',fontsize=15)
            plt.xlabel(r'\textbf{highest %s - [climate model data] - %s}' % (typeOfCorr[ii],monthlychoice),
                        color='dimgrey',fontsize=8)
            
            for te in range(len(uni)):
                plt.text(2092,uni[te],r'\textbf{\#%s}' % cou[te],fontsize=7,color='k',
                          ha='center',va='center')
            
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
            ax.spines['bottom'].set_linewidth(0)
            ax.tick_params('y',length=4,width=2,which='major',color='dimgrey')
            ax.tick_params('x',length=0,width=0,which='major',color='dimgrey')
            
            xbar = np.arange(option)
            rects = plt.bar(xbar,correnstrends, align='center',zorder=2)
            
            plt.axhline(y=np.max(correnstrends),linestyle='--',linewidth=2,color='k',
                        clip_on=False,zorder=100,dashes=(1,0.3))
            plt.axhline(y=0,linestyle='-',linewidth=2,color='dimgrey',
                        clip_on=False,zorder=1)
            
            ### Set color
            colorlist = ['darkcyan']*8
            for cccc in range(option):
                rects[cccc].set_color(colorlist[cccc])
                rects[cccc].set_edgecolor(colorlist[cccc])
            
            plt.xticks(np.arange(0,option+1,1),modelGCMs,size=5.45)
            plt.yticks(np.arange(-1,1.1,0.1),map(str,np.round(np.arange(-1,1.1,0.1),2)),size=6)
            plt.xlim([-0.5,option-1+0.5])   
            plt.ylim([-0.1,1])
            
            plt.xlabel(r'\textbf{highest R - [climate model data] - %s}' % (monthlychoice),color='dimgrey',fontsize=8,labelpad=8)
            plt.title(r'\textbf{...AND FOR LINEAR TRENDS}',color='k',fontsize=15)
            
            plt.tight_layout()
            plt.savefig(directoryfigure + saveData + '_%s_RawClimateModel_PatternCorrelations_%sclasses.png' % (typeOfCorr[ii],len(corr)),dpi=300)