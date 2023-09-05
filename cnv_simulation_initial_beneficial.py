import numpy as np

def get_params(parameters):
    if isinstance(parameters, np.ndarray):
        s_cnv, m_cnv, p_0 = np.power(10,parameters)
    if not isinstance(parameters, np.ndarray):
        s_cnv = np.power(10,parameters.sc)
        m_cnv = np.power(10,parameters.mc)
        p_0 = np.power(10, parameters.p0)
        
    return s_cnv, m_cnv, p_0


## Wright Fisher stochastic discrete time model ##
## Modified to include p_0 - proportion of unreported CNVs at gen 0 ##
def simpleWF(N, generation, s_cnv, m_cnv, p_0, seed=None):
    """ CNV evolution simulator
    Simulates CNV and SNV evolution for x generations
    Returns proportion of the population with a CNV for generations observed in Lauer et al. 2018 as 1d np.array same length as generation
    
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
    
    s_cnv : float
        fitness benefit of CNVs  
    m_cnv : float 
        probability mutation to CNV 
    p_0: float
        fraction of population with GAP1 CNV before beginning of experiment
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
    
    # First 20 generations - wt, cnv-, snv
#     g_0 = 20
#     N_0 = np.uint64(10**6)
    
#     w_0 = np.array([1, 1 + s_cnv, 1 + s_snv], dtype = 'float64')
#     S_0 = np.diag(w_0)
#     M_0 = np.array([[1-m_cnv-m_snv,0,0],
#                     [m_cnv,1,0],
#                     [m_snv,0,1]])
#     assert np.allclose(M_0.sum(axis=0), 1)
#     E_0 = M_0 @ S_0
#     n_0 = np.zeros(3)
#     n_0[0] = N_0*(1-p_0)
#     n_0[1] = N_0*p_0
    
#     for t in range(g_0+1):    
#         p = n_0/N_0  # counts to frequencies
#         p = E_0 @ p.reshape((3, 1))  # natural selection + mutation        
#         p /= p.sum()  # rescale proportions
#         n_0 = np.random.multinomial(N_0, np.ndarray.flatten(p)) # random genetic drift
    
    
    # Order is: wt, cnv+, cnv-, snv
    
    w = np.array([1, 1 + s_cnv, 1 + s_cnv, 1 + s_snv], dtype='float64')
    S = np.diag(w)
    
    # make transition rate array
    M = np.array([[1 - m_cnv - m_snv, 0, 0, 0],
                [m_cnv, 1, 0, 0],
                [0, 0, 1, 0],
                [m_snv, 0, 0, 1]], dtype='float64')
    assert np.allclose(M.sum(axis=0), 1)
    
    
    # mutation and selection
    E = M @ S

    # rows are genotypes, p has proportions after initial (unreported) growth
    n = np.zeros(4)
    # n[0] = N*(1-p[1]-p[2]) # wt
    # n[1] = 0 # cnv+
    # n[2] = N*p[1] # cnv-
    # n[3] = N*p[2] # snv
    
    n[2] = N*p_0 # cnv-
    n[0] = N*(1-p_0) # wt
    
    # follow proportion of the population with CNV
    # here rows will be generation, columns (there is only one) is replicate population
    p_cnv = []
    
    # run simulation to generation 116
    for t in range(int(generation.max()+1)):    
        p = n/N  # counts to frequencies
        p_cnv.append(p[1])  # frequency of reported CNVs
        p = E @ p.reshape((4, 1))  # natural selection + mutation        
        p /= p.sum()  # rescale proportions
        n = np.random.multinomial(N, np.ndarray.flatten(p)) # random genetic drift
    
    return np.transpose(p_cnv)[generation.astype(int)]

def CNVsimulator_simpleWF(reps, N, generation, parameters, seed=None):
    s_cnv, m_cnv, p_0 = get_params(parameters)
    evo_reps = []
    for i in range(reps):
        out=simpleWF(N, generation, s_cnv, m_cnv, p_0, seed=seed)
        evo_reps.append(out)
    return np.array(evo_reps)