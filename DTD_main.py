# -*- coding: utf-8 -*-

import numpy as np
from DTD import DTD
from DTD_utils import *
import matplotlib.pyplot as plt
import os
import time

def main(isloop=True,issave=True,alpha1_exact=0.2,beta1_exact=0.002,alpha_inexact=5,\
         beta_inexact=0.05,totalnum_samples=20000,islog10=False):
    start_time = time.time()
    
    start_iter=0
    num_postavg=20
    outfolder_name="./ToyResults/alpha1_exact_"+str(alpha1_exact)+"__"+"beta1_exact"+str(beta1_exact)+"__"+\
    "alpha_inexact"+str(alpha_inexact)+"__"+"beta_inexact"+str(beta_inexact)+"__"+\
        str(totalnum_samples)+"samples/"
    if isloop:
        V=get_V_3diags(d=10,p_central=0.8)
        netname='Loop_Link'
    else:
        V=get_V_diagmain(d=10,p_central=0.8)
        netname='Full_Link'

    colors=['darkviolet','red','green','blue','black','gold']
    hyps=[]
    k=0
    for N in [1,10,20,50,100]:    
        # hyps.append({'alg':'TDC_exact', 'start_iter':start_iter, 'num_iters':[np.round(totalnum_samples/N).astype('int')], \
        #              'num_postavg':num_postavg,'batchsize':N, 'alpha':alpha1_exact*N, 'beta':beta1_exact*N, 'plot_iters':None,\
        #                  'color':colors[k],'marker':'','legend':'Decentralized TDC(N='+str(N)+')',\
        #         'num_rho_localavg':[0],'result_dir':outfolder_name+"ErrorData_"+netname+'/TDCexact_'+str(N)+'batch'})
        hyps.append({'alg':'TDC_inexact', 'start_iter':start_iter, 'num_iters':[np.round(totalnum_samples/N).astype('int')], \
                     'num_postavg':num_postavg,'batchsize':N, 'alpha':alpha1_exact*N, 'beta':beta1_exact*N, 'plot_iters':None,\
                     'color':colors[k],'marker':'','legend':'Decentralized TDC(N='+str(N)+')',\
                     'num_rho_localavg':[3],'result_dir':outfolder_name+"ErrorData_"+netname+'/TDCinexact_L3_'+str(N)+'batch'})
        k+=1
        
    hyps.append({'alg':'TD0_inexact', 'start_iter':start_iter, 'num_iters':[totalnum_samples], 'num_postavg':num_postavg,\
            'batchsize':1, 'alpha':alpha1_exact, 'beta':0, 'plot_iters':None,\
            'color':colors[5],'marker':'','legend':'Decentralized TD(0)(N=1)',\
            'num_rho_localavg':[3],'result_dir':outfolder_name+"ErrorData_"+netname+"/TD0inexact_L3_1batch"})
    
    N=100
    num_iters=np.round(totalnum_samples/N).astype('int')
    hyps.append({'alg':'TDC_exact', 'start_iter':start_iter, 'num_iters':[num_iters], 'num_postavg':num_postavg,\
           'batchsize':N, 'alpha':alpha_inexact, 'beta':beta_inexact, 'plot_iters':None,\
           'color':colors[1],'marker':'','legend':'Decentralized TDC(exact '+r'$\rho$'+')',\
           'num_rho_localavg':[0],'result_dir':outfolder_name+"ErrorData_"+netname+"/TDC_exactRho"})
    hyps.append({'alg':'TDC_inexact', 'start_iter':start_iter, 'num_iters':[num_iters], 'num_postavg':num_postavg,\
           'batchsize':N, 'alpha':alpha_inexact, 'beta':beta_inexact, 'plot_iters':None,\
           'color':colors[2],'marker':'','legend':'Decentralized TDC(L=1)',\
           'num_rho_localavg':[1],'result_dir':outfolder_name+"ErrorData_"+netname+"/TDC_L1"})
    hyps.append({'alg':'TDC_inexact', 'start_iter':start_iter, 'num_iters':[num_iters], 'num_postavg':num_postavg,\
           'batchsize':N, 'alpha':alpha_inexact, 'beta':beta_inexact, 'plot_iters':None,\
           'color':colors[3],'marker':'','legend':'Decentralized TDC(L=3)',\
           'num_rho_localavg':[3],'result_dir':outfolder_name+"ErrorData_"+netname+"/TDC_L3"})
    hyps.append({'alg':'TDC_inexact', 'start_iter':start_iter, 'num_iters':[num_iters], 'num_postavg':num_postavg,\
           'batchsize':N, 'alpha':alpha_inexact, 'beta':beta_inexact, 'plot_iters':None,\
           'color':colors[4],'marker':'','legend':'Decentralized TDC(L=5)',\
           'num_rho_localavg':[5],'result_dir':outfolder_name+"ErrorData_"+netname+"/TDC_L5"})
    hyps.append({'alg':'TDC_inexact', 'start_iter':start_iter, 'num_iters':[num_iters], 'num_postavg':num_postavg,\
           'batchsize':N, 'alpha':alpha_inexact, 'beta':beta_inexact, 'plot_iters':None,\
           'color':colors[5],'marker':'','legend':'Decentralized TDC(L=7)',\
           'num_rho_localavg':[7],'result_dir':outfolder_name+"ErrorData_"+netname+"/TDC_L7"})
        
    print("hyperparamters:")
    for hyp in hyps:
        print(hyp)
    print("\n")
    
    dtd=DTD(seed_init=1,V=V)
    dtd.collect(num_expr=100,totalnum_samples=totalnum_samples,init_state=0,seed_collect=1,maxnum_rhoavg=7)
    if issave:
        dtd.save_collect(data_dir=outfolder_name+"SampleData_"+netname)
    
    dtd.sim(hyps,init_Theta=None,init_W=np.zeros((10,5)),seed_sim=1)
    if issave:
        dtd.save_results(hyps)
    
    plt.figure()
    plt.clf()
    plt.rcParams.update({'font.size': 17})
    plt.close()
    for xlabel in ['number of samples','number of communication rounds']:
        dtd.plot_results(hyps[0:6],percentile=95,color_transition=None,xlabel=xlabel,\
                         plotresult_dir=outfolder_name+'Figures_varyN_'+netname,fontsize=17,lgdsize=15,result_type='Theta_avg_err',legend_loc=1)
    
    xlabel='number of communication rounds'
    hyps_inexact=[hyps[jj].copy() for jj in [6,7,8,9,10]]
    if isloop:
        title="Ring network"
    else:
        title="Fully connected network"
    dtd.plot_results(hyps_inexact,percentile=95,islog10=islog10,color_transition=None,xlabel=xlabel,\
                     plotresult_dir=outfolder_name+'Figures_varyL_'+netname,fontsize=17,lgdsize=13,title=title,result_type='Theta_avg_err',legend_loc=9)
    dtd.plot_results(hyps_inexact,percentile=95,islog10=islog10,color_transition='black',xlabel=xlabel,\
                     plotresult_dir=outfolder_name+'Figures_varyL_'+netname,fontsize=17,lgdsize=13,title=title,result_type='Theta_consensus_err',legend_loc=9)
    
    time_info="\n\n: Elapsed time="+str((time.time() - start_time)/60)+" min"
    print(time_info)
    return dtd

if False:
    isloop=False;issave=True;alpha1_exact=0.2;beta1_exact=0.002;alpha_inexact=5;beta_inexact=0.05;totalnum_samples=20000;islog10=False
    
    dtd.load_results(hyps)
    keys=list(dtd.results.keys())
    for k in range(6):
        result_preplot=dtd.results[keys[k]]['Theta_avg_err']
        print("\n\n\n k="+str(k)+", key="+str(keys[k]))
        print(result_preplot.mean(axis=0))   
    
    
