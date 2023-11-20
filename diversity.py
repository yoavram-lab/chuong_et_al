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
def simpleWF(N, generation, s_cnv, m_cnv, p_0, max_possible, seed=None):
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
    max_possible: int
        maximum number of unique CNVs at GAP1 locus
    """
    
    # SNV parameters as constants
    s_snv = 1e-3
    m_snv = 1e-5
    
    # CNV events counter
    Q,P = np.zeros(generation.max()+2), np.zeros(generation.max()+2)
    Q[0] = min(p_0*N, max_possible) # Number of CNVs, initially cnv without reporter
    P[0] = min(p_0*N, max_possible) # Total number of CNVs generated at gen t, initially cnv without reporter
    
    if seed is not None:
        np.random.seed(seed=seed)
    else:
        np.random.seed()

    
    assert N > 0
    N = np.uint64(N)    
    
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
    n[2] = N*p_0 # cnv-
    n[0] = N*(1-p_0) # wt
    
    # follow proportion of the population with CNV
    # here rows will be generation, columns (there is only one) is replicate population
    p_cnv = []
    
    # run simulation to generation 116
    for t in range(int(generation.max()+1)):
        new_cnvs = min(n[0]*m_cnv, max_possible) # number of new gene duplications at gen. t is m_cnv * N(wt)[t]
        Q[t+1] = new_cnvs
        P[t+1] = new_cnvs
        
        p = n/N  # counts to frequencies
        p_cnv.append(p[1])  # frequency of reported CNVs
        p = E @ p.reshape((4, 1))  # natural selection + mutation
        
        P = (1+s_cnv)*P/p.sum() # selection for previous generations, next gens remain 0
        
        p /= p.sum()  # rescale proportions
        n = np.random.multinomial(N, np.ndarray.flatten(p)) # random genetic drift
    
    return np.array([Q,P]).reshape(2,len(Q))

def CNVsimulator_simpleWF(reps, N, generation, parameters, seed=None):
    s_cnv, m_cnv, p_0 = get_params(parameters)
    evo_reps = []
    for i in range(reps):
        out=simpleWF(N, generation, s_cnv, m_cnv, p_0, seed=seed)
        evo_reps.append(out)
    return np.array(evo_reps)