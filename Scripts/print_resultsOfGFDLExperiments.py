"""
Script for showing results of GFDL-CM3 experiments

Author     : Zachary M. Labe
Date       : 21 July 2021
Version    : 4 - subsamples random weight class (#8) for mmmean
"""

### Import packages
import sys
import matplotlib.pyplot as plt
import numpy as np
import cmocean as cmocean

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

### LRP param
DEFAULT_NUM_BWO_ITERATIONS = 200
DEFAULT_BWO_LEARNING_RATE = .001

### Plotting defaults 
plt.rc('text',usetex=True)
plt.rc('font',**{'family':'sans-serif','sans-serif':['Avant Garde']}) 

###############################################################################
###############################################################################
###############################################################################
### Data preliminaries 
directorydataLLL = '/Users/zlabe/Data/LENS/monthly'
directorydataENS = '/Users/zlabe/Data/SMILE/'
directorydataBB = '/Users/zlabe/Data/BEST/'
directorydataEE = '/Users/zlabe/Data/ERA5/'
directoryoutput = '/Users/zlabe/Documents/Research/ModelComparison/Data/'
###############################################################################
###############################################################################
modelGCMs = ['CCCma_canesm2','MPI','CSIRO_MK3.6','KNMI_ecearth',
              'GFDL_CM3','GFDL_ESM2M','lens']
datasetsingle = ['SMILE']
dataset_obs = 'ERA5BE'
seasons = ['annual']
variq = 'T2M'
reg_name = 'LowerArctic'
timeper = 'historical'
###############################################################################
###############################################################################
# pickSMILE = ['CCCma_canesm2','CSIRO_MK3.6','KNMI_ecearth',
#               'GFDL_ESM2M','lens']
# pickSMILE = ['CCCma_canesm2','MPI','lens']
pickSMILE = []
if len(pickSMILE) >= 1:
    lenOfPicks = len(pickSMILE)
else:
    lenOfPicks = len(modelGCMs)
###############################################################################
###############################################################################
land_only = False
ocean_only = False
if land_only == True:
    maskNoiseClass = 'land'
elif ocean_only == True:
    maskNoiseClass = 'ocean'
else:
    maskNoiseClass = 'none'

###############################################################################
###############################################################################
rm_merid_mean = False
rm_annual_mean = False
###############################################################################
###############################################################################
rm_ensemble_mean = False
rm_observational_mean = False
###############################################################################
###############################################################################
calculate_anomalies = False
if calculate_anomalies == True:
    if timeper == 'historical': 
        baseline = np.arange(1951,1980+1,1)
    elif timeper == 'future':
        baseline = np.arange(2021,2050+1,1)
    else:
        print(ValueError('WRONG TIMEPER!'))
###############################################################################
###############################################################################
window = 0
ensTypeExperi = 'ENS'
# shuffletype = 'TIMEENS'
# shuffletype = 'ALLENSRAND'
# shuffletype = 'ALLENSRANDrmmean'
shuffletype = 'RANDGAUSS'
sizeOfTwin = 4 # name of experiment for adding noise class #8
if sizeOfTwin > 0:
    sizeOfTwinq = 1
else:
    sizeOfTwinq = sizeOfTwin
###############################################################################
###############################################################################
if ensTypeExperi == 'ENS':
    if window == 0:
        rm_standard_dev = False
        if timeper == 'historical': 
            yearsall = np.arange(1950,2019+1,1)
        elif timeper == 'future':
            yearsall = np.arange(2020,2099+1,1)
        else:
            print(ValueError('WRONG TIMEPER!'))
            sys.exit()
        ravel_modelens = False
        ravelmodeltime = False
    else:
        rm_standard_dev = True
        if timeper == 'historical': 
            yearsall = np.arange(1950+window,2019+1,1)
        elif timeper == 'future':
            yearsall = np.arange(2020+window,2099+1,1)
        else:
            print(ValueError('WRONG TIMEPER!'))
            sys.exit()
        ravelmodeltime = False
        ravel_modelens = True
elif ensTypeExperi == 'GCM':
    if window == 0:
        rm_standard_dev = False
        yearsall = np.arange(1950,2019+1,1)
        ravel_modelens = False
        ravelmodeltime = False
    else:
        rm_standard_dev = True
        if timeper == 'historical': 
            yearsall = np.arange(1950,2019+1,1)
        elif timeper == 'future':
            yearsall = np.arange(2020,2099+1,1)
        else:
            print(ValueError('WRONG TIMEPER!'))
            sys.exit()
        ravelmodeltime = False
        ravel_modelens = True
###############################################################################
###############################################################################
numOfEns = 16
lensalso = True
if len(pickSMILE) == 0:
    if modelGCMs[-1] == 'RANDOM':
        randomalso = True
    else:
        randomalso = False
elif len(pickSMILE) != 0:
    if pickSMILE[-1] == 'RANDOM':
        randomalso = True
    else:
        randomalso = False
lentime = len(yearsall)
###############################################################################
###############################################################################
ravelyearsbinary = False
ravelbinary = False
num_of_class = lenOfPicks + sizeOfTwinq
###############################################################################
###############################################################################
lrpRule = 'z'
normLRP = True
###############################################################################
###############################################################################
###############################################################################
###############################################################################
### Picking experiment to save
typeOfAnalysis = 'issueWithExperiment'

# Experiment #1
if rm_ensemble_mean == True:
    if window > 1:
        if calculate_anomalies == False:
            if rm_merid_mean == False:
                if rm_observational_mean == False:
                    if rm_annual_mean == False:
                        typeOfAnalysis = 'Experiment-1'
# Experiment #2
if rm_ensemble_mean == True:
    if window == 0:
        if calculate_anomalies == False:
            if rm_merid_mean == False:
                if rm_observational_mean == False:
                    if rm_annual_mean == False:
                        typeOfAnalysis = 'Experiment-2'
# Experiment #3 (raw data)
if rm_ensemble_mean == False:
    if window == 0:
        if calculate_anomalies == False:
            if rm_merid_mean == False:
                if rm_observational_mean == False:
                    if rm_annual_mean == False:
                        typeOfAnalysis = 'Experiment-3'
                        if variq == 'T2M':
                            integer = 20 # random noise value to add/subtract from each grid point
                        elif variq == 'P':
                            integer = 20 # random noise value to add/subtract from each grid point
                        elif variq == 'SLP':
                            integer = 20 # random noise value to add/subtract from each grid point
# Experiment #4
if rm_ensemble_mean == False:
    if window == 0:
        if calculate_anomalies == False:
            if rm_merid_mean == False:
                if rm_observational_mean == False:
                    if rm_annual_mean == True:
                        typeOfAnalysis = 'Experiment-4'
                        if variq == 'T2M':
                            integer = 25 # random noise value to add/subtract from each grid point
                        elif variq == 'P':
                            integer = 15 # random noise value to add/subtract from each grid point
                        elif variq == 'SLP':
                            integer = 5 # random noise value to add/subtract from each grid point
# Experiment #5
if rm_ensemble_mean == False:
    if window == 0:
        if calculate_anomalies == False:
            if rm_merid_mean == False:
                if rm_observational_mean == True:
                    if rm_annual_mean == False:
                        typeOfAnalysis = 'Experiment-5'
# Experiment #6
if rm_ensemble_mean == False:
    if window == 0:
        if calculate_anomalies == False:
            if rm_merid_mean == False:
                if rm_observational_mean == True:
                    if rm_annual_mean == True:
                        typeOfAnalysis = 'Experiment-6'
# Experiment #7
if rm_ensemble_mean == False:
    if window == 0:
        if calculate_anomalies == True:
            if rm_merid_mean == False:
                if rm_observational_mean == True:
                    if rm_annual_mean == False:
                        typeOfAnalysis = 'Experiment-7'
# Experiment #8
if rm_ensemble_mean == False:
    if window == 0:
        if calculate_anomalies == True:
            if rm_merid_mean == False:
                if rm_observational_mean == False:
                    if rm_annual_mean == False:
                        typeOfAnalysis = 'Experiment-8'
                        if variq == 'T2M':
                            integer = 1 # random noise value to add/subtract from each grid point
                        elif variq == 'P':
                            integer = 1 # random noise value to add/subtract from each grid point
                        elif variq == 'SLP':
                            integer = 5 # random noise value to add/subtract from each grid point
# Experiment #9
if rm_ensemble_mean == False:
    if window > 1:
        if calculate_anomalies == True:
            if rm_merid_mean == False:
                if rm_observational_mean == False:
                    if rm_annual_mean == False:
                        typeOfAnalysis = 'Experiment-9'
                        
print('\n<<<<<<<<<<<< Analysis == %s (%s) ! >>>>>>>>>>>>>>>\n' % (typeOfAnalysis,timeper))
if typeOfAnalysis == 'issueWithExperiment':
    sys.exit('Wrong parameters selected to analyze')

###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
### Read in labels for each experiment
experinumber = 11
labels = []
for i in range(experinumber):
    factorObs = i # factor to add to obs
    
    ### Select how to save files
    if land_only == True:
        saveData = timeper + '_' + seasons[0] + '_LAND' + '_NoiseTwinSingleMODDIF4_AddingWARMTH-toGFDL%s_' % (factorObs) + typeOfAnalysis + '_' + variq + '_' + reg_name + '_' + dataset_obs + '_' + 'NumOfSMILE-' + str(num_of_class) + '_Method-' + ensTypeExperi
    elif ocean_only == True:
        saveData = timeper + '_' + seasons[0] + '_OCEAN' + '_NoiseTwinSingleMODDIF4_AddingWARMTH-toGFDL%s_' % (factorObs) + typeOfAnalysis + '_' + variq + '_' + reg_name + '_' + dataset_obs + '_' + 'NumOfSMILE-' + str(num_of_class) + '_Method-' + ensTypeExperi
    else:
        saveData = timeper + '_' + seasons[0] + '_NoiseTwinSingleMODDIF4_AddingWARMTH-toGFDL%s_' % (factorObs) + typeOfAnalysis + '_' + variq + '_' + reg_name + '_' + dataset_obs + '_' + 'NumOfSMILE-' + str(num_of_class) + '_Method-' + ensTypeExperi
    print('*Filename == < %s >' % saveData) 
    labelexperi = np.genfromtxt(directoryoutput + 'obsLabels_' + saveData + '.txt',unpack=True)
    labels.append(labelexperi)
    
labels = np.asarray(labels).astype(int)

np.set_printoptions(linewidth=np.inf)
print('\n\n\n')
print('%100s' % labels[0])
print('-----------Regular data-----------\n')

np.set_printoptions(linewidth=np.inf)
print('%100s' % labels[1])
print('-----------Warm mean state by 3C----------\n')

np.set_printoptions(linewidth=np.inf)
print('%100s' % labels[2])
print('-----------Cool mean state by 3c-----------\n')

np.set_printoptions(linewidth=np.inf)
print('%100s' % labels[3])
print('-----------Warm recent 10 years by 3c-----------\n')

np.set_printoptions(linewidth=np.inf)
print('%100s' % labels[4])
print('-----------Cool recent 10 years by 3c-----------\n')

np.set_printoptions(linewidth=np.inf)
print('%100s' % labels[5])
print('-----------Warm North Pole by 5C-----------\n')

np.set_printoptions(linewidth=np.inf)
print('%100s' % labels[6])
print('-----------Cool North Pole by 5C-----------\n')

np.set_printoptions(linewidth=np.inf)
print('%100s' % labels[7])
print('-----------Warm Lower Arctic by 5C-----------\n')

np.set_printoptions(linewidth=np.inf)
print('%100s' % labels[8])
print('-----------Cool Lower Arctic by 5C-----------\n')

np.set_printoptions(linewidth=np.inf)
print('%100s' % labels[9])
print('-----------Warm first 50 years by 3C-----------\n')

np.set_printoptions(linewidth=np.inf)
print('%100s' % labels[10])
print('-----------Cool first 50 years by 3C-----------\n')

# if factorObs == 0:
#     data = data
# elif factorObs == 1: # warm its mean state
#     GFDL = data[4,:,:,:,:]
#     GFDLwarmer = GFDL + 3
#     data[4,:,:,:,:] = GFDLwarmer
# elif factorObs == 2: # cool its mean state
#     GFDL = data[4,:,:,:,:]
#     GFDLcooler = GFDL - 3
#     data[4,:,:,:,:] = GFDLcooler
# elif factorObs == 3: # warm recent 10 years
#     GFDL = data[4,:,:,:,:] 
#     GFDLbefore = GFDL[:,:-10,:,:]
#     GFDLafter = GFDL[:,-10:,:,:] + 3
#     GFDLq = np.append(GFDLbefore,GFDLafter,axis=1)
#     data[4,:,:,:,:] = GFDLq
# elif factorObs == 4: # cool recent 10 years
#     GFDL = data[4,:,:,:,:] 
#     GFDLbefore = GFDL[:,:-10,:,:]
#     GFDLafter = GFDL[:,-10:,:,:] - 3
#     GFDLq = np.append(GFDLbefore,GFDLafter,axis=1)
#     data[4,:,:,:,:] = GFDLq         
# elif factorObs == 5: # warm the North Pole
#     sizeofNP = 10
#     GFDL = data[4,:,:,:,:] 
#     warmerNP = np.zeros((GFDL.shape[0],GFDL.shape[1],GFDL.shape[2]-sizeofNP,GFDL.shape[3])) + 5
#     addtoclimoNP = GFDL[:,:,sizeofNP:,:] + warmerNP
#     GFDL[:,:,sizeofNP:,:] = addtoclimoNP
#     data[4,:,:,:,:] = GFDL
# elif factorObs == 6: # cool the North Pole
#     sizeofNP = 10
#     GFDL = data[4,:,:,:,:] 
#     coolerNP = np.zeros((GFDL.shape[0],GFDL.shape[1],GFDL.shape[2]-sizeofNP,GFDL.shape[3])) - 5
#     addtoclimoNP = GFDL[:,:,sizeofNP:,:] + coolerNP
#     GFDL[:,:,sizeofNP:,:] = addtoclimoNP
#     data[4,:,:,:,:] = GFDL
# elif factorObs == 7: # warm the Lower Arctic
#     sizeofLA = 5
#     GFDL = data[4,:,:,:,:] 
#     warmerLA = np.zeros((GFDL.shape[0],GFDL.shape[1],sizeofLA,GFDL.shape[3])) + 5
#     addtoclimoLA = GFDL[:,:,:sizeofLA,:] + warmerLA
#     GFDL[:,:,:sizeofLA,:] = addtoclimoLA
#     data[4,:,:,:,:] = GFDL
# elif factorObs == 8: # cool the Lower Arctic
#     sizeofLA = 5
#     GFDL = data[4,:,:,:,:] 
#     coolerLA = np.zeros((GFDL.shape[0],GFDL.shape[1],sizeofLA,GFDL.shape[3])) - 5
#     addtoclimoLA = GFDL[:,:,:sizeofLA,:] + coolerLA
#     GFDL[:,:,:sizeofLA,:] = addtoclimoLA
#     data[4,:,:,:,:] = GFDL
# elif factorObs == 9: # warm early 50 years
#     GFDL = data[4,:,:,:,:] 
#     GFDLafter = GFDL[:,50:,:,:]
#     GFDLbefore = GFDL[:,:50,:,:] + 3
#     GFDLq = np.append(GFDLbefore,GFDLafter,axis=1)
#     data[4,:,:,:,:] = GFDLq
# elif factorObs == 10: # cool early 50 years
#     GFDL = data[4,:,:,:,:] 
#     GFDLafter = GFDL[:,50:,:,:]
#     GFDLbefore = GFDL[:,:50,:,:] - 3
#     GFDLq = np.append(GFDLbefore,GFDLafter,axis=1)
#     data[4,:,:,:,:] = GFDLq    