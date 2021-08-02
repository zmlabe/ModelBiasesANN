"""
Script for plotting RMSE between models and observations

Author     : Zachary M. Labe
Date       : 26 July 2021
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
monthlychoiceq = ['annual','JFM','AMJ','JAS','OND']
typeOfCorr = ['RMSE']
variables = ['T2M','P','SLP']
reg_name = 'LowerArctic'
level = 'surface'
timeper = 'historical'
if timeper == 'historical':
    years = np.arange(1950,2019+1,1)
    yearsglo = np.arange(1950,2019+1,1)
land_only = False
ocean_only = False

if reg_name == 'SMILEGlobe':
    region = 'Global'
elif reg_name == 'narrowTropics':
    region = 'Tropics'
elif reg_name == 'LowerArctic':
    region = 'Arctic'

### Read in data
# for vv in range(len(variables)):
#     for mo in range(len(monthlychoiceq)):
for vv in range(1):
      for mo in range(1):
        variq = variables[vv]
        monthlychoice = monthlychoiceq[mo]
        directorydata = '/Users/zlabe/Documents/Research/ModelComparison/Data/Climatologies/'
        directoryfigure = '/Users/zlabe/Desktop/ModelComparison_v1/Climatologies/RMSE/%s/' % (variq)
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
        
        rmse = np.load(directorydata + saveData + '_RMSE.npz')['arr_0'][:]
        modelGCMs = ['CanESM2','MPI','CSIRO-MK3.6','KNMI-ecearth','GFDL-CM3','GFDL-ESM2M','LENS','MMmean'][:]

        ### Ensemble mean correlations
        rmseens = np.nanmean(rmse,axis=1)
        ensALLr = [rmseens]
        
        ### Pick highest correlation for each year
        rmseyr = np.argmin(rmseens,axis=0)
        yrALLr = [rmseyr]
        
        ### Counts of data for each year
        uniquermseyr,countrmseyr = np.unique(rmseyr,return_counts=True)
        uniqueALL = [uniquermseyr]
        countALL = [countrmseyr]
        
        ###############################################################################
        ### Removed annual mean from each map
        rmseglo = np.load(directorydata + saveData + '_RMSEglo.npz')['arr_0'][:]
        modelGCMsglo = ['CanESM2','MPI','CSIRO-MK3.6','KNMI-ecearth','GFDL-CM3','GFDL-ESM2M','LENS','MMmean'][:]

        ### Ensemble mean correlations
        rmseensglo = np.nanmean(rmseglo,axis=1)
        ensALLrglo = [rmseensglo]
        
        ### Pick highest correlation for each year
        rmseyrglo = np.argmin(rmseensglo,axis=0)
        yrALLrglo = [rmseyrglo]
        
        ### Counts of data for each year
        uniquermseyrglo,countrmseyrglo = np.unique(rmseyrglo,return_counts=True)
        uniqueALLglo = [uniquermseyrglo]
        countALLglo = [countrmseyrglo]
        
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
            
            x=np.arange(1950,2019+1,1)
            plt.scatter(x,prediction,c=prediction,s=40,clip_on=False,cmap=cc.Antique_8.mpl_colormap,
                        edgecolor='k',linewidth=0.4,zorder=10)
            
            plt.xticks(np.arange(1950,2021,5),map(str,np.arange(1950,2021,5)),size=6)
            plt.yticks(np.arange(0,12+1,1),modelGCMs,size=6)
            plt.xlim([1950,2020])   
            plt.ylim([0,len(rmse)-1])
            
            plt.title(r'\textbf{RMSE FOR [ %s ] - %s}' % (variables[vv],region),color='k',fontsize=15)
            plt.xlabel(r'\textbf{lowest %s - [climate model data] - %s}' % (typeOfCorr[ii],monthlychoice),
                        color='dimgrey',fontsize=8)
            
            for te in range(len(uni)):
                plt.text(2022,uni[te],r'\textbf{\#%s}' % cou[te],fontsize=7,color='k',
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
            ax.spines['bottom'].set_linewidth(2)
            ax.tick_params('both',length=4,width=2,which='major',color='dimgrey')
            ax.yaxis.grid(zorder=1,color='dimgrey',alpha=0.35)
            
            color = cmr.infinity(np.linspace(0.00,1,len(modelGCMs)))
            for i,c in zip(range(len(modelGCMs)),color):
                if i == 7:
                    c = 'k'
                else:
                    c = c
                plt.plot(years,rmseens[i],color=c,linewidth=2.3,
                            label=r'\textbf{%s}' % modelGCMs[i],zorder=11,
                            clip_on=False,alpha=1)
            
            if variq == 'T2M':
                if reg_name == 'SMILEGlobe':
                    leg = plt.legend(shadow=False,fontsize=9,loc='upper center',
                                  bbox_to_anchor=(0.5,0.1),fancybox=True,ncol=4,frameon=False,
                                  handlelength=0,handletextpad=0)
                    for line,text in zip(leg.get_lines(), leg.get_texts()):
                        text.set_color(line.get_color())
                    
                    plt.xticks(np.arange(1950,2030+1,10),map(str,np.arange(1950,2030+1,10)),size=5.45)
                    plt.yticks(np.arange(0,11.5,0.5),map(str,np.round(np.arange(0,11,0.5),3)),size=6)
                    plt.xlim([1950,2020])   
                    plt.ylim([0,3])
                elif reg_name == 'narrowTropics':
                    leg = plt.legend(shadow=False,fontsize=9,loc='upper center',
                                  bbox_to_anchor=(0.5,0.1),fancybox=True,ncol=4,frameon=False,
                                  handlelength=0,handletextpad=0)
                    for line,text in zip(leg.get_lines(), leg.get_texts()):
                        text.set_color(line.get_color())
                    
                    plt.xticks(np.arange(1950,2030+1,10),map(str,np.arange(1950,2030+1,10)),size=5.45)
                    plt.yticks(np.arange(0,11.5,0.5),map(str,np.round(np.arange(0,11,0.5),3)),size=6)
                    plt.xlim([1950,2020])   
                    plt.ylim([0,3])
                elif reg_name == 'LowerArctic':
                    leg = plt.legend(shadow=False,fontsize=9,loc='upper center',
                                  bbox_to_anchor=(0.5,0.1),fancybox=True,ncol=4,frameon=False,
                                  handlelength=0,handletextpad=0)
                    for line,text in zip(leg.get_lines(), leg.get_texts()):
                        text.set_color(line.get_color())
                    
                    plt.xticks(np.arange(1950,2030+1,10),map(str,np.arange(1950,2030+1,10)),size=5.45)
                    plt.yticks(np.arange(0,11.5,0.5),map(str,np.round(np.arange(0,11,0.5),3)),size=6)
                    plt.xlim([1950,2020])   
                    plt.ylim([0,8])
                elif reg_name == 'Arctic':
                    if land_only == True:
                        leg = plt.legend(shadow=False,fontsize=9,loc='upper center',
                                      bbox_to_anchor=(0.5,0.1),fancybox=True,ncol=4,frameon=False,
                                      handlelength=0,handletextpad=0)
                        for line,text in zip(leg.get_lines(), leg.get_texts()):
                            text.set_color(line.get_color())
                        
                        plt.xticks(np.arange(1950,2030+1,10),map(str,np.arange(1950,2030+1,10)),size=5.45)
                        plt.yticks(np.arange(0,11.5,1),map(str,np.round(np.arange(0,11,1),3)),size=6)
                        plt.xlim([1950,2020])   
                        plt.ylim([0,5])
                    elif ocean_only == True:
                        leg = plt.legend(shadow=False,fontsize=9,loc='upper center',
                                      bbox_to_anchor=(0.5,0.1),fancybox=True,ncol=4,frameon=False,
                                      handlelength=0,handletextpad=0)
                        for line,text in zip(leg.get_lines(), leg.get_texts()):
                            text.set_color(line.get_color())
                        
                        plt.xticks(np.arange(1950,2030+1,10),map(str,np.arange(1950,2030+1,10)),size=5.45)
                        plt.yticks(np.arange(0,11.5,1),map(str,np.round(np.arange(0,11,1),3)),size=6)
                        plt.xlim([1950,2020])   
                        plt.ylim([0,8])
                    else:
                        leg = plt.legend(shadow=False,fontsize=9,loc='upper center',
                                      bbox_to_anchor=(0.5,0.1),fancybox=True,ncol=4,frameon=False,
                                      handlelength=0,handletextpad=0)
                        for line,text in zip(leg.get_lines(), leg.get_texts()):
                            text.set_color(line.get_color())
                        
                        plt.xticks(np.arange(1950,2030+1,10),map(str,np.arange(1950,2030+1,10)),size=5.45)
                        plt.yticks(np.arange(0,11.5,1),map(str,np.round(np.arange(0,11,1),3)),size=6)
                        plt.xlim([1950,2020])   
                        plt.ylim([1,9])
            else:
                leg = plt.legend(shadow=False,fontsize=9,loc='upper center',
                                  bbox_to_anchor=(0.5,0.15),fancybox=True,ncol=4,frameon=False,
                                  handlelength=0,handletextpad=0)
                for line,text in zip(leg.get_lines(), leg.get_texts()):
                    text.set_color(line.get_color())
                    
                plt.xticks(np.arange(1950,2030+1,10),map(str,np.arange(1950,2030+1,10)),size=5.45)
                plt.yticks(np.arange(0,1.01,0.1),map(str,np.round(np.arange(0,1.01,0.1),3)),size=6)
                plt.xlim([1950,2020])   
                plt.ylim([0.4,1.0])                
        
            plt.xlabel(r'\textbf{Average RMSE - [climate model data] - %s for %s for \underline{%s}}' % (monthlychoice,variq,region),color='dimgrey',fontsize=8,labelpad=8)
            plt.title(r'\textbf{RMSE PER YEAR - %s}' % typemask,color='k',fontsize=15)
        
            plt.tight_layout()
            
            plt.savefig(directoryfigure + saveData + '_RawClimateModel_RMSE.png',dpi=300)
            
###############################################################################
###############################################################################
###############################################################################
        ### Remove annual mean from each map
        for ii in range(len(yrALLrglo)):
            predictionglo = yrALLrglo[ii]
            uniglo = uniqueALLglo[ii]
            couglo = countALLglo[ii]
                       
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
            
            xglo=np.arange(1950,2019+1,1)
            plt.scatter(xglo,predictionglo,c=predictionglo,s=40,clip_on=False,cmap=cc.Antique_8.mpl_colormap,
                        edgecolor='k',linewidth=0.4,zorder=10)
            
            plt.xticks(np.arange(1950,2021,5),map(str,np.arange(1950,2021,5)),size=6)
            plt.yticks(np.arange(0,12+1,1),modelGCMsglo,size=6)
            plt.xlim([1950,2020])   
            plt.ylim([0,len(rmse)-1])
            
            plt.title(r'\textbf{RMSE FOR [ %s ] - %s}' % (variables[vv],region),color='k',fontsize=15)
            plt.xlabel(r'\textbf{lowest %s - [climate model data] - %s}' % (typeOfCorr[ii],monthlychoice),
                        color='dimgrey',fontsize=8)
            
            for te in range(len(uni)):
                plt.text(2022,uni[te],r'\textbf{\#%s}' % couglo[te],fontsize=7,color='k',
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
            ax.spines['bottom'].set_linewidth(2)
            ax.tick_params('both',length=4,width=2,which='major',color='dimgrey')
            ax.yaxis.grid(zorder=1,color='dimgrey',alpha=0.35)
            
            color = cmr.infinity(np.linspace(0.00,1,len(modelGCMsglo)))
            for i,c in zip(range(len(modelGCMsglo)),color):
                if i == 7:
                    c = 'k'
                else:
                    c = c
                plt.plot(yearsglo,rmseensglo[i],color=c,linewidth=2.3,
                            label=r'\textbf{%s}' % modelGCMsglo[i],zorder=11,
                            clip_on=False,alpha=1)
            
            if variq == 'T2M':
                if reg_name == 'SMILEGlobe':
                    leg = plt.legend(shadow=False,fontsize=9,loc='upper center',
                                  bbox_to_anchor=(0.5,0.1),fancybox=True,ncol=4,frameon=False,
                                  handlelength=0,handletextpad=0)
                    for line,text in zip(leg.get_lines(), leg.get_texts()):
                        text.set_color(line.get_color())
                    
                    plt.xticks(np.arange(1950,2030+1,10),map(str,np.arange(1950,2030+1,10)),size=5.45)
                    plt.yticks(np.arange(0,11.5,0.5),map(str,np.round(np.arange(0,11,0.5),3)),size=6)
                    plt.xlim([1950,2020])   
                    plt.ylim([0,3])
                elif reg_name == 'narrowTropics':
                    leg = plt.legend(shadow=False,fontsize=9,loc='upper center',
                                  bbox_to_anchor=(0.5,0.1),fancybox=True,ncol=4,frameon=False,
                                  handlelength=0,handletextpad=0)
                    for line,text in zip(leg.get_lines(), leg.get_texts()):
                        text.set_color(line.get_color())
                    
                    plt.xticks(np.arange(1950,2030+1,10),map(str,np.arange(1950,2030+1,10)),size=5.45)
                    plt.yticks(np.arange(0,11.5,0.5),map(str,np.round(np.arange(0,11,0.5),3)),size=6)
                    plt.xlim([1950,2020])   
                    plt.ylim([0,3])
                elif reg_name == 'LowerArctic':
                    leg = plt.legend(shadow=False,fontsize=9,loc='upper center',
                                  bbox_to_anchor=(0.5,0.1),fancybox=True,ncol=4,frameon=False,
                                  handlelength=0,handletextpad=0)
                    for line,text in zip(leg.get_lines(), leg.get_texts()):
                        text.set_color(line.get_color())
                    
                    plt.xticks(np.arange(1950,2030+1,10),map(str,np.arange(1950,2030+1,10)),size=5.45)
                    plt.yticks(np.arange(0,11.5,0.5),map(str,np.round(np.arange(0,11,0.5),3)),size=6)
                    plt.xlim([1950,2020])   
                    plt.ylim([0,8])
                elif reg_name == 'Arctic':
                    if land_only == True:
                        leg = plt.legend(shadow=False,fontsize=9,loc='upper center',
                                      bbox_to_anchor=(0.5,0.1),fancybox=True,ncol=4,frameon=False,
                                      handlelength=0,handletextpad=0)
                        for line,text in zip(leg.get_lines(), leg.get_texts()):
                            text.set_color(line.get_color())
                        
                        plt.xticks(np.arange(1950,2030+1,10),map(str,np.arange(1950,2030+1,10)),size=5.45)
                        plt.yticks(np.arange(0,11.5,1),map(str,np.round(np.arange(0,11,1),3)),size=6)
                        plt.xlim([1950,2020])   
                        plt.ylim([0,5])
                    elif ocean_only == True:
                        leg = plt.legend(shadow=False,fontsize=9,loc='upper center',
                                      bbox_to_anchor=(0.5,0.1),fancybox=True,ncol=4,frameon=False,
                                      handlelength=0,handletextpad=0)
                        for line,text in zip(leg.get_lines(), leg.get_texts()):
                            text.set_color(line.get_color())
                        
                        plt.xticks(np.arange(1950,2030+1,10),map(str,np.arange(1950,2030+1,10)),size=5.45)
                        plt.yticks(np.arange(0,11.5,1),map(str,np.round(np.arange(0,11,1),3)),size=6)
                        plt.xlim([1950,2020])   
                        plt.ylim([0,8])
                    else:
                        leg = plt.legend(shadow=False,fontsize=9,loc='upper center',
                                      bbox_to_anchor=(0.5,0.1),fancybox=True,ncol=4,frameon=False,
                                      handlelength=0,handletextpad=0)
                        for line,text in zip(leg.get_lines(), leg.get_texts()):
                            text.set_color(line.get_color())
                        
                        plt.xticks(np.arange(1950,2030+1,10),map(str,np.arange(1950,2030+1,10)),size=5.45)
                        plt.yticks(np.arange(0,11.5,1),map(str,np.round(np.arange(0,11,1),3)),size=6)
                        plt.xlim([1950,2020])   
                        plt.ylim([1,9])
            else:
                leg = plt.legend(shadow=False,fontsize=9,loc='upper center',
                                  bbox_to_anchor=(0.5,0.15),fancybox=True,ncol=4,frameon=False,
                                  handlelength=0,handletextpad=0)
                for line,text in zip(leg.get_lines(), leg.get_texts()):
                    text.set_color(line.get_color())
                    
                plt.xticks(np.arange(1950,2030+1,10),map(str,np.arange(1950,2030+1,10)),size=5.45)
                plt.yticks(np.arange(0,1.01,0.1),map(str,np.round(np.arange(0,1.01,0.1),3)),size=6)
                plt.xlim([1950,2020])   
                plt.ylim([0.4,1.0])                
        
            plt.xlabel(r'\textbf{Average RMSE - [climate model data] - %s for %s for \underline{%s}}' % (monthlychoice,variq,region),color='dimgrey',fontsize=8,labelpad=8)
            plt.title(r'\textbf{RMSE(glo) PER YEAR - %s}' % typemask,color='k',fontsize=15)
        
            plt.tight_layout()
            
            plt.savefig(directoryfigure + saveData + '_RawClimateModel_RMSEglo.png',dpi=300)