import numpy as np
import random

def set_hyps(a,a_default):
    if(a is None):
        return a_default
    else:
        return a

def set_seed(seed=1):
    if seed is not None:
        np.random.seed(seed)    
        random.seed(seed)
    
def get_V_diagmain(d=None,p_central=0.9):
    #Generate dXd communication matrix V with diagonals=p_central and other entries being p_off=(1-p_central)/(d-1)
    if d==1:
        return np.array([1])
    p_off=(1-p_central)/(d-1)
    V=np.array([[p_off]*d]*d)
    np.fill_diagonal(V,p_central)
    return V

def get_V_3diags(d=None,p_central=0.8):  
    #Generate dXd communication matrix V with V[i,i]=p_central and V[i,i+1]=V[i,i-1]=p_off=(1-p_central)/2, 
    # and other entries being zero.
    if d==1:
        return np.array([1])
    V=np.zeros(shape=(d,d))
    p_off=(1-p_central)/2
    if d==2:
        V=np.array([p_central,p_off],[p_off,p_central])
    else:
        V[0,0]=p_central
        V[0,1]=p_off
        V[0,d-1]=p_off
        V[d-1,d-1]=p_central
        V[d-1,d-2]=p_off
        V[d-1,0]=p_off
        for i in range(1,d-1):
            V[i,i]=p_central
            V[i,i-1]=p_off
            V[i,i+1]=p_off
    return V

def get_hyp_str(hyp):
    hyp1=hyp.copy()
    hyp1.pop('plot_iters')
    hyp1.pop('color')
    hyp1.pop('marker')
    hyp1.pop('legend')
    hyp1.pop('result_dir')
    return str(hyp1)