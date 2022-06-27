# -*- coding: utf-8 -*-

import numpy as np
from DTD_path import DTD_path
from DTD_utils import *
import matplotlib.pyplot as plt
import os
import time
import pdb

#alpha1_exact=0.04,beta1_exact=0.002 needs more iterations to converge at target state
#alpha1_exact=0.08,beta1_exact=0.002 is pretty close to converge at target state
#alpha1_exact=0.08,beta1_exact=0.002 looks pretty close to converge at target state, 
#  but the numpy array shows that N=100 diverges
#alpha1_exact=0.08,beta1_exact=0.004 is too large
#alpha1_exact=0.09,beta1_exact=0.002 is too large

def main_path(issave=True,num_expr=100,alpha1_exact=0.06,beta1_exact=0.002,alpha_inexact=5,\
         beta_inexact=0.05,totalnum_samples=20000,islog10=False):       #For path finding problem
    start_time = time.time()

    height=10
    width=10
    gridmap=np.zeros((height,width))
    
    #obstacles
    gridmap[0,0:3]=-1
    gridmap[1,9]=-1
    gridmap[2,[1,4,9]]=-1
    gridmap[3,3]=-1
    gridmap[4,7:9]=-1
    gridmap[5,[4,7]]=-1
    gridmap[6,1]=-1
    gridmap[7,8]=-1
    gridmap[8,[5,8]]=-1
    gridmap[9,[3,8]]=-1
    
    #goals of 3 agents
    gridmap[0,7:]=[1,2,3]
        
    #initial locations of 3 agents
    num_agents=3
    init1=[9,0]
    init2=[9,1]
    init3=[9,2]
    init_state=init1+init2+init3
    
    start_iter=0
    num_postavg=20
    outfolder_name="./PathResults/alpha1_exact_"+str(alpha1_exact)+"__"+"beta1_exact"+str(beta1_exact)+"__"+\
    "alpha_inexact"+str(alpha_inexact)+"__"+"beta_inexact"+str(beta_inexact)+"__"+\
        str(totalnum_samples)+"samples/"
    
    V=get_V_diagmain(d=3,p_central=0.6)

    colors=['darkviolet','red','green','blue','black','gold']
    hyps=[]
    k=0
    for N in [1,10,20,50,100]:    
        hyps.append({'alg':'TDC_inexact', 'start_iter':start_iter, 'num_iters':[np.round(totalnum_samples/N).astype('int')], \
                     'num_postavg':num_postavg,'batchsize':N, 'alpha':alpha1_exact*N, 'beta':beta1_exact*N, 'plot_iters':None,\
                     'color':colors[k],'marker':'','legend':'Decentralized TDC(N='+str(N)+')',\
                     'num_rho_localavg':[3],'result_dir':outfolder_name+'ErrorData/TDCinexact_L3_'+str(N)+'batch'})
        k+=1
        
    hyps.append({'alg':'TD0_inexact', 'start_iter':start_iter, 'num_iters':[totalnum_samples], 'num_postavg':num_postavg,\
            'batchsize':1, 'alpha':alpha1_exact, 'beta':0, 'plot_iters':None,\
            'color':colors[5],'marker':'','legend':'Decentralized TD(0)(N=1)',\
            'num_rho_localavg':[3],'result_dir':outfolder_name+"ErrorData/TD0inexact_L3_1batch"})
    
    N=100
    num_iters=np.round(totalnum_samples/N).astype('int')
    if False:
        hyps.append({'alg':'TDC_exact', 'start_iter':start_iter, 'num_iters':[num_iters], 'num_postavg':num_postavg,\
               'batchsize':N, 'alpha':alpha_inexact, 'beta':beta_inexact, 'plot_iters':None,\
               'color':colors[0],'marker':'','legend':'Decentralized TDC(exact '+r'$\rho$'+')',\
               'num_rho_localavg':[0],'result_dir':outfolder_name+"ErrorData/TDC_exactRho"})
        hyps.append({'alg':'TDC_inexact', 'start_iter':start_iter, 'num_iters':[num_iters], 'num_postavg':num_postavg,\
               'batchsize':N, 'alpha':alpha_inexact, 'beta':beta_inexact, 'plot_iters':None,\
               'color':colors[1],'marker':'','legend':'Decentralized TDC(L=1)',\
               'num_rho_localavg':[1],'result_dir':outfolder_name+"ErrorData/TDC_L1"})
        hyps.append({'alg':'TDC_inexact', 'start_iter':start_iter, 'num_iters':[num_iters], 'num_postavg':num_postavg,\
               'batchsize':N, 'alpha':alpha_inexact, 'beta':beta_inexact, 'plot_iters':None,\
               'color':colors[2],'marker':'','legend':'Decentralized TDC(L=3)',\
               'num_rho_localavg':[3],'result_dir':outfolder_name+"ErrorData/TDC_L3"})
        hyps.append({'alg':'TDC_inexact', 'start_iter':start_iter, 'num_iters':[num_iters], 'num_postavg':num_postavg,\
               'batchsize':N, 'alpha':alpha_inexact, 'beta':beta_inexact, 'plot_iters':None,\
               'color':colors[3],'marker':'','legend':'Decentralized TDC(L=5)',\
               'num_rho_localavg':[5],'result_dir':outfolder_name+"ErrorData/TDC_L5"})
        hyps.append({'alg':'TDC_inexact', 'start_iter':start_iter, 'num_iters':[num_iters], 'num_postavg':num_postavg,\
               'batchsize':N, 'alpha':alpha_inexact, 'beta':beta_inexact, 'plot_iters':None,\
               'color':colors[4],'marker':'','legend':'Decentralized TDC(L=7)',\
               'num_rho_localavg':[7],'result_dir':outfolder_name+"ErrorData/TDC_L7"})
        
    print("hyperparamters:")
    for hyp in hyps:
        print(hyp)
    print("\n")
    
    dtd=DTD_path(gridmap,seed_init=1,gamma=0.95,R_finish=3,R_stay_goal=0,R_nocollide=-0.075,R_collide=-0.5,V=V,init_state=init_state)
    dtd.collect(num_expr=num_expr,totalnum_samples=totalnum_samples,seed_collect=1,maxnum_rhoavg=7)   
    if issave:
        dtd.save_collect(data_dir=outfolder_name+"SampleData")
    
    dtd.sim(hyps,init_Theta=None,init_W=np.zeros((num_agents,5*num_agents)),seed_sim=1)
    if issave:
        dtd.save_results(hyps)
    
    plt.figure()
    plt.clf()
    plt.rcParams.update({'font.size': 17})
    plt.close()
    for xlabel in ['number of samples','number of communication rounds']:
        phi=np.array([1.0,0.0,0.0,0.0,0.0]*3)
        dtd.plot_results(hyps[0:6],percentile=95,islog10=False,color_transition=None,xlabel=xlabel,ylabel='value function of target state',\
                         plotresult_dir=outfolder_name+'Figures_varyN',filename='goal_'+xlabel,phi=phi,fontsize=17,lgdsize=15,legend_loc=1)
        num_features=len(phi)
        if False:
            for k in range(num_features):
                phi=np.zeros(num_features)
                phi[k]=1.0
                dtd.plot_results(hyps[0:6],percentile=95,islog10=False,color_transition=None,xlabel=xlabel,ylabel='theta'+str(k),\
                                 plotresult_dir=outfolder_name+'Figures_varyN',filename='theta'+str(k)+'_'+xlabel,phi=phi,fontsize=17,lgdsize=15,legend_loc=1)
    
    # pdb.set_trace()
    if False:
        xlabel='number of communication rounds'
        hyps_inexact=[hyps[jj].copy() for jj in [6,7,8,9,10]]
        phi=np.array([1.0,0.0,0.0,0.0,0.0]*3)
        dtd.plot_results(hyps_inexact,percentile=95,islog10=islog10,color_transition=None,xlabel=xlabel,ylabel='value function of target state',\
                         plotresult_dir=outfolder_name+'Figures_varyL',fontsize=17,lgdsize=13,title="",legend_loc=9)
        dtd.plot_results(hyps_inexact,percentile=95,islog10=islog10,color_transition='black',xlabel=xlabel,\
                         plotresult_dir=outfolder_name+'Figures_varyL',fontsize=17,lgdsize=13,title="",legend_loc=9)
    
    time_info="\n\n: Elapsed time="+str((time.time() - start_time)/60)+" min"
    print(time_info)
    return dtd
















if False:
    for i in range(5):
        for j in range(i+1,7):
            print(str(i)+str(j))
    
    if (1==1) and (3==3):
        print(999)
    
    shape1=(3,4,5)
    s=np.random.normal(size=shape1)
    phi=np.random.normal(size=shape1[2]).reshape((1,1,-1))
    tmp=(s*phi).sum(axis=2)
    
    for i in range(shape1[0]):
        for j in range(shape1[1]):
            err=(s[i,j,:].reshape(-1)*phi.reshape(-1)).sum()-tmp[i,j]
            print(err)
            
    
    issave=True;num_expr=100;alpha1_exact=0.06;beta1_exact=0.002;alpha_inexact=5;\
         beta_inexact=0.05;totalnum_samples=20000;islog10=False
    
    
    dtd.load_results(hyps)
    phi=np.array([1.0,0.0,0.0,0.0,0.0]*3)
    keys=list(dtd.Theta.keys())
    for k in range(6):
        result_preplot=(dtd.Theta[keys[k]].mean(axis=2)*phi).sum(axis=2)
        print("\n\n\n k="+str(k)+", key="+str(keys[k]))
        print(result_preplot)   
        
    
    
    
    