# -*- coding: utf-8 -*-

import numpy as np
from DTD_cliff import DTD_cliff
from DTD_utils import *
import matplotlib.pyplot as plt
import os
import time
import pdb

def main_cliff(issave=True,num_expr=100,alpha1_exact=0.2,beta1_exact=0.002,alpha_inexact=5,\
         beta_inexact=0.05,totalnum_samples=20000,islog10=False):    
    start_time = time.time()

    height=3
    width=4
    num_states=(height*width)**2   #Each agent counts its own state from top row (i=0) to bottom row (i=height-1)
    num_actions=4     #up,down,left,right
    num_agents=2
    r_cliff=-100  #i=height-1, j not equal to 0 or width-1.
    r_goal1=-0.5    #The reward of the agent when it arrives at the destination while the other does not
    r_goal2=0     #The reward of both agents when both agents are at the destination
    r_rest=-1     #The reward of the agent when it is not in cliff nor destination
    
    def ijij2s(i1,j1,i2,j2,height,width): #return the agents' shared state if the two agents are at (i1,j1),(i2,j2) respectively
        s1=i1*width+j1
        s2=i2*width+j2
        return s1*width*height+s2
    
    def next_ij(i,j,a):
        if i==height-1:
            if j==width-1:   
                return (i,j)  #stay at destination
            elif j!=0:
                return (i,0)  #from cliff to starting point
        
        i_next=i
        j_next=j
        if a==0 and i>0:  #up
            i_next-=1
        elif a==1 and i<height-1:   #down
            i_next+=1
        elif a==2 and j>0:  #left
            j_next-=1
        elif a==3 and j<width-1:   #right
            j_next+=1
        return (i_next,j_next)
    
    transP=np.zeros((num_states,num_actions,num_actions,num_states))
    reward=r_rest*np.ones((num_states,num_actions,num_actions,num_states,2))
    s=0
    for i1 in range(height):
        for j1 in range(width):
            for i2 in range(height):
                for j2 in range(width):
                    for a1 in range(num_actions):
                        for a2 in range(num_actions):
                            i1_next,j1_next=next_ij(i1,j1,a1)
                            i2_next,j2_next=next_ij(i2,j2,a2)
                            s_next=ijij2s(i1_next,j1_next,i2_next,j2_next,height,width)
                            transP[s,a1,a2,s_next]=1.0
                            if i1==width and j1>0 and j1<width-1:
                                reward[s,a1,a2,s_next,0]=r_cliff
                            elif i1_next==height-1 and j1_next==width-1:
                                reward[s,a1,a2,s_next,0]=r_goal1
                            
                            if i2==width and j2>0 and j2<width-1:
                                reward[s,a1,a2,s_next,1]=r_cliff
                            elif i2_next==height-1 and j2_next==width-1:
                                reward[s,a1,a2,s_next,1]=r_goal1
                    s+=1
                    
    s_destination=num_states-1
    s_pre=ijij2s(height-2,width-1,height-2,width-1,height,width)  #the state when both agents are right above the destination
    reward[s_destination,:,:,s_destination,:]=r_goal2
    reward[s_pre,:,:,s_destination,:]=r_goal2
    
    init_xi=np.zeros(num_states)
    s_init=ijij2s(height-1,0,height-1,0,height,width)
    init_xi[s_init]=1.0
    
    start_iter=0
    num_postavg=20
    outfolder_name="./CliffResults/alpha1_exact_"+str(alpha1_exact)+"__"+"beta1_exact"+str(beta1_exact)+"__"+\
    "alpha_inexact"+str(alpha_inexact)+"__"+"beta_inexact"+str(beta_inexact)+"__"+\
        str(totalnum_samples)+"samples/"
    
    V=get_V_diagmain(d=2,p_central=0.7)

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
    
    if False:
        N=100
        num_iters=np.round(totalnum_samples/N).astype('int')
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
    
    dtd=DTD_cliff(seed_init=1,state_space=range(num_states),action_spaces=[list(range(num_actions))]*num_agents,\
                 transP=transP,reward=reward,gamma=0.95, behavior_policy=None, \
                 target_policy=None, V=V, features='identity matrix')
        
    dtd.collect(num_expr=num_expr,totalnum_samples=totalnum_samples,init_state=s_init,seed_collect=1,maxnum_rhoavg=7)
    if issave:
        dtd.save_collect(data_dir=outfolder_name+"SampleData")
    
    dtd.sim(hyps,init_Theta=None,init_W=np.zeros((num_agents,num_states)),seed_sim=1)
    if issave:
        dtd.save_results(hyps)
    
    plt.figure()
    plt.clf()
    plt.rcParams.update({'font.size': 17})
    plt.close()
    for xlabel in ['number of samples','number of communication rounds']:
        dtd.plot_results(hyps[0:6],percentile=95,color_transition=None,xlabel=xlabel,\
                         plotresult_dir=outfolder_name+'Figures_varyN',fontsize=17,lgdsize=15,result_type='Theta_avg_err',legend_loc=1)
    if False:
        xlabel='number of communication rounds'
        hyps_inexact=[hyps[jj].copy() for jj in [6,7,8,9,10]]
        dtd.plot_results(hyps_inexact,percentile=95,islog10=islog10,color_transition=None,xlabel=xlabel,\
                         plotresult_dir=outfolder_name+'Figures_varyL',fontsize=17,lgdsize=13,title="",result_type='Theta_avg_err',legend_loc=9)
        dtd.plot_results(hyps_inexact,percentile=95,islog10=islog10,color_transition='black',xlabel=xlabel,\
                         plotresult_dir=outfolder_name+'Figures_varyL',fontsize=17,lgdsize=13,title="",result_type='Theta_consensus_err',legend_loc=9)
    
    time_info="\n\n: Elapsed time="+str((time.time() - start_time)/60)+" min"
    print(time_info)
    return dtd

if False:
    issave=True;num_expr=100;alpha1_exact=0.2;beta1_exact=0.002;alpha_inexact=5;beta_inexact=0.05;totalnum_samples=20000;islog10=False
    keys=list(dtd.results.keys())
    dtd.load_results(hyps)
    for k in range(6):
        result_preplot=dtd.results[keys[k]]['Theta_avg_err']
        print("\n\n\n k="+str(k)+", key="+str(keys[k]))
        print(result_preplot.mean(axis=0))   
