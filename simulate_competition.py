import numpy as np

def get_params(parameters):
    if isinstance(parameters, np.ndarray):
        s_1, s_2, m_1, m_2 = np.power(10,parameters)
    if not isinstance(parameters, np.ndarray):
        s_1 = np.power(10,parameters.s1)
        m_1 = np.power(10,parameters.m1)
        s_2 = np.power(10,parameters.s2)
        m_2 = np.power(10,parameters.m2)
        
    return s_1, s_2, m_1, m_2


## Wright Fisher stochastic discrete time model ##

def simpleWF(N, generation, s_1, s_2, m_1, m_2, seed=None):
    """ CNV competition simulator
    Simulates competition of two genotypes for x generations
    Returns proportion of the two populations for the specified generations
    
    Parameters
    -------------------
    N : int
        population size  
    s_snv : float
        fitness benefit of SNVs  
    m_snv : float 
        probability mutation to SNV   
    generation : np.array, 1d 
        with generations to output
    seed : int
    
    s_1, s_2 : float
        fitness benefits of CNVs  
    m_1, m_2 : float 
        probability mutations to CNVs
    """
    
    # SNV parameters as constants
    s_snv = 1e-3
    m_snv = 1e-5
    
    if seed is not None:
        np.random.seed(seed=seed)
    else:
        np.random.seed()

    
    assert N > 0
    N = np.uint64(N)    
    
    # Order is: wt_1, wt_2, cnv1, cnv2, snv_1, snv_2
    
    w = np.array([1,1, 1 + s_1, 1 + s_2, 1 + s_snv, 1 + s_snv], dtype='float64')
    S = np.diag(w)
    
    # make transition rate array
    M = np.array([[1 - m_1 - m_snv, 0, 0, 0, 0, 0],
    		   [1 - m_2 - m_snv, 0, 0, 0, 0, 0],
                   [m_1, 0, 1, 0, 0, 0],
                   [m_2, 0, 0, 1, 0, 0],
                   [m_snv, 0, 0, 0, 1, 0],
                   [m_snv, 0, 0, 0, 0, 1]], dtype='float64')
    # print(M.sum(axis=0))
    # assert np.allclose(M.sum(axis=0), 1)
    
    
    # mutation and selection
    E = M @ S

    # rows are genotypes
    n = np.zeros(6)
    n[0] = 0.5 # wt_1
    n[1] = 0.5 # wt_2
    
    # follow proportion of the population with CNV
    # here rows will be generation, columns (there is only one) is replicate population
    p_res = []
    
    # run simulation to generation 116
    for t in range(int(generation.max()+1)):    
        p = n/N  # counts to frequencies
        p_res.append([p[0]+p[2], p[1]+p[3]])  # frequency of reported CNVs
        p = E @ p.reshape((6, 1))  # natural selection + mutation        
        p /= p.sum()  # rescale proportions
        n = np.random.multinomial(N, np.ndarray.flatten(p)) # random genetic drift
    return np.transpose(p_res)[:,generation.astype(int)]

def CNVsimulator_simpleWF(reps, N, generation, parameters, seed=None):
    s_1, s_2, m_1, m_2 = get_params(parameters)
    evo_reps = []
    for i in range(reps):
        out=simpleWF(N, generation, s_1, s_2, m_1, m_2, seed=seed)
        evo_reps.append(out)
    return np.array(evo_reps)
