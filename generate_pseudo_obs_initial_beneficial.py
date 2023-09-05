# -*- coding: utf-8 -*-
"""
Created on Feb 18 10:08:39 2021
cnv_sim_sbi

@author: grace
"""
import random
import argparse
import numpy as np
from numpy.random import normal

from cnv_simulation_4_params import CNVsimulator_simpleWF, CNVsimulator_simpleChemo
parser = argparse.ArgumentParser()
parser.add_argument('-g', "--generation_file")
parser.add_argument('-n', "--name")
args = parser.parse_args()
g_file = str(args.generation_file)
ending = str(args.name)

#####other parameters needed for model #####
# pop size, fitness SNVs, mutation rate SNVs, number of generations
N = 3.3e8
reps=5
generation=np.genfromtxt(g_file,delimiter=',', skip_header=1,dtype="int64")



# cnv_params = np.log10(np.array([1e-1,1e-5]))
# all_params=np.tile(cnv_params, (reps,1))

# obs = CNVsimulator_simpleChemo(reps, s_snv, m_snv, generation, parameters=cnv_params, seed=None)
# all_obs = obs

# cnv_params = np.log10(np.array([1e-1,1e-7]))
# all_params=np.append(all_params,np.tile(cnv_params, (reps,1)), axis=0)

# obs = CNVsimulator_simpleChemo(reps, s_snv, m_snv, generation, parameters=cnv_params, seed=None)
# all_obs =np.append(all_obs,obs,axis=0)
        
# cnv_params = np.log10(np.array([1e-3,1e-5]))
# all_params=np.append(all_params,np.tile(cnv_params, (reps,1)), axis=0)

# obs = CNVsimulator_simpleChemo(reps, s_snv, m_snv, generation, parameters=cnv_params, seed=None)
# all_obs =np.append(all_obs,obs,axis=0)
        
# cnv_params = np.log10(np.array([1e-3,1e-7]))
# all_params=np.append(all_params,np.tile(cnv_params, (reps,1)), axis=0)

# obs = CNVsimulator_simpleChemo(reps, s_snv, m_snv, generation, parameters=cnv_params, seed=None)
# all_obs =np.append(all_obs,obs,axis=0)

# all_obs = np.append(all_obs,all_params,axis=1)

# np.savetxt("Chemo_simulated_single_observations.csv", all_obs, delimiter=',')


## WF ##
# cnv_params = np.log10(np.array([1e-1,1e-5]))
# all_params=np.tile(cnv_params, (reps,1))

# obs = CNVsimulator_simpleWF(reps, N, s_snv, m_snv, generation, parameters=cnv_params, seed=None)
# all_obs_WF = obs

# cnv_params = np.log10(np.array([1e-1,1e-7]))
# all_params=np.append(all_params,np.tile(cnv_params, (reps,1)), axis=0)

# obs = CNVsimulator_simpleWF(reps, N, s_snv, m_snv, generation, parameters=cnv_params, seed=None)
# all_obs_WF =np.append(all_obs_WF,obs,axis=0)
        
# cnv_params = np.log10(np.array([1e-3,1e-5]))
# all_params=np.append(all_params,np.tile(cnv_params, (reps,1)), axis=0)

# obs = CNVsimulator_simpleWF(reps, N, s_snv, m_snv, generation, parameters=cnv_params, seed=None)
# all_obs_WF =np.append(all_obs_WF,obs,axis=0)
        
# cnv_params = np.log10(np.array([1e-3,1e-7]))
# all_params=np.append(all_params,np.tile(cnv_params, (reps,1)), axis=0)

# obs = CNVsimulator_simpleWF(reps, N, s_snv, m_snv, generation, parameters=cnv_params, seed=None)
# all_obs_WF =np.append(all_obs_WF,obs,axis=0)

# all_obs_WF = np.append(all_obs_WF,all_params,axis=1)

# np.savetxt("WF_simulated_single_observations.csv", all_obs_WF, delimiter=',')

### additional observations after review, WF only ###



cnv_params = np.log10(np.array([10**-1,10**-(3.8), 10**-(1.40),10**-(3.6)]))
all_params=np.tile(cnv_params, (reps,1))

obs = CNVsimulator_simpleWF(reps, N, generation, parameters=cnv_params, seed=None)
all_obs_WF = obs

cnv_params = np.log10(np.array([10**-1.3,10**-5, 1e-1, 1e-6]))
all_params=np.append(all_params,np.tile(cnv_params, (reps,1)), axis=0)

obs = CNVsimulator_simpleWF(reps, N, generation, parameters=cnv_params, seed=None)
all_obs_WF =np.append(all_obs_WF,obs,axis=0)

cnv_params = np.log10(np.array([10**-0.86,1e-5, 10**-1.1, 10**-4.2]))
all_params=np.append(all_params,np.tile(cnv_params, (reps,1)), axis=0)

obs = CNVsimulator_simpleWF(reps, N, generation, parameters=cnv_params, seed=None)
all_obs_WF =np.append(all_obs_WF,obs,axis=0)

cnv_params = np.log10(np.array([10**-1.8,1e-4, 10**-2, 10**-3]))
all_params=np.append(all_params,np.tile(cnv_params, (reps,1)), axis=0)

obs = CNVsimulator_simpleWF(reps, N, generation, parameters=cnv_params, seed=None)

all_obs_WF =np.append(all_obs_WF,obs,axis=0)

cnv_params = np.log10(np.array([1e-1,1e-3, 1e-4, 1e-3]))
all_params=np.append(all_params,np.tile(cnv_params, (reps,1)), axis=0)

obs = CNVsimulator_simpleWF(reps, N, generation, parameters=cnv_params, seed=None)

all_obs_WF =np.append(all_obs_WF,obs,axis=0)

cnv_params = np.log10(np.array([1e-1,1e-9, 1e-4, 1e-9]))
all_params=np.append(all_params,np.tile(cnv_params, (reps,1)), axis=0)

obs = CNVsimulator_simpleWF(reps, N, generation, parameters=cnv_params, seed=None)

all_obs_WF =np.append(all_obs_WF,obs,axis=0)

cnv_params = np.log10(np.array([1e-3,1e-3, 1e-2, 1e-4]))
all_params=np.append(all_params,np.tile(cnv_params, (reps,1)), axis=0)

obs = CNVsimulator_simpleWF(reps, N, generation, parameters=cnv_params, seed=None)
all_obs_WF =np.append(all_obs_WF,obs,axis=0)

cnv_params = np.log10(np.array([1e-3,1e-9, 1e-3, 1e-9]))
all_params=np.append(all_params,np.tile(cnv_params, (reps,1)), axis=0)

obs = CNVsimulator_simpleWF(reps, N, generation, parameters=cnv_params, seed=None)
all_obs_WF =np.append(all_obs_WF,obs,axis=0)

all_obs_WF = np.append(all_obs_WF,all_params,axis=1)

#old_WF = np.genfromtxt("WF_simulated_single_observations.csv", delimiter=",")

#all_obs_WF = np.append(old_WF, all_obs_WF, axis = 0)

np.savetxt(f"WF_simulated_single_observations_{ending}.csv", all_obs_WF, delimiter=',')
