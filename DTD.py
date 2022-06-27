import numpy as np
import random
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from DTD_utils import *
import os
import pdb   #to delete

#default values:
num_states=10
num_actions=2
num_features=5
num_agents=10

class DTD:
    def __init__(self,seed_init=1,state_space=None,action_spaces=None,\
                 transP=None,reward=None,gamma=0.95, behavior_policy=None, \
                 target_policy=None, V=None, features=None):
        set_seed(seed_init)
        self.seed_init=seed_init
        
        self.state_space=set_hyps(a=state_space,a_default=range(num_states))   
        self.action_spaces=set_hyps(a=action_spaces,a_default=[list(range(num_actions))]*num_agents)  
        self.num_states=len(self.state_space)
        self.num_actions=[len(tmp) for tmp in self.action_spaces].copy()    #self.num_actions[m] for agent m
        self.num_agents=len(self.num_actions)
        
        transP_shape=tuple([self.num_states]+self.num_actions+[self.num_states])
        if transP is None:
            self.transP=np.zeros(transP_shape)   #P(s,a1,...,aM,s')
            for s in range(self.num_states):
                num_branches=5
                nonzero_indexes=np.arange(start=s,stop=s+num_branches,step=1)
                tmp=(nonzero_indexes>self.num_states-1)
                nonzero_indexes[tmp]=nonzero_indexes[tmp]-self.num_states
                tmp=np.random.uniform(size=tuple(self.num_actions+[num_branches]))
                exec("self.transP[s"+",:"*(self.num_agents+1)+"]["+":,"*self.num_agents+"nonzero_indexes]=tmp/np.sum(tmp,axis=self.num_agents,keepdims=True)")
        else:
            assert transP.shape == transP_shape, \
                "transP should have shape: (num_states,num_actions1,...,num_actionsM,num_states)"
            transP=np.abs(transP)
            self.transP=transP/np.sum(transP,axis=self.num_agents+1,keepdims=True)
        
        reward_shape=tuple([self.num_states]+self.num_actions+[self.num_states]+[self.num_agents])
        if (reward is not None):
            assert reward.shape==reward_shape,\
                "reward should be either None or an np.array with shape (num_states,num_actions1,...,num_actionsM,num_states,num_agents)"
        self.reward = set_hyps(a=reward,a_default=np.random.uniform(size=reward_shape))
        
        self.gamma=gamma

        self.behavior_policy=[0]*self.num_agents
        if behavior_policy is None:
            for m in range(self.num_agents):
                self.behavior_policy[m]=np.array([[1/self.num_actions[m]]*self.num_actions[m]]*self.num_states)  #pi(s,a)
        else: 
            for m in range(self.num_agents):
                assert behavior_policy[m].shape == (self.num_states,self.num_actions[m]), "behavior_policy["+str(m)+"] should have shape: (num_states,num_actions[m])"    
                behavior_policy[m] = np.abs(behavior_policy[m])
                self.behavior_policy[m]=behavior_policy[m]/np.sum(behavior_policy[m],axis=1,keepdims=True)

        self.target_policy=[0]*self.num_agents
        if target_policy is None:
            for m in range(self.num_agents):
                self.target_policy[m]=np.random.normal(loc=1/self.num_actions[m],scale=0.1/self.num_actions[m],size=(self.num_states,self.num_actions[m]))   #pi(s,a)
                self.target_policy[m]=np.abs(self.target_policy[m])
                self.target_policy[m]=self.target_policy[m]/np.sum(self.target_policy[m],axis=1,keepdims=True)
        else: 
            for m in range(self.num_agents):
                assert target_policy[m].shape == (self.num_states,self.num_actions[m]), "target_policy["+str(m)+"] should have shape: (num_states,num_actions[m])"
                target_policy[m]=np.abs(target_policy[m])
                self.target_policy[m]=target_policy[m]/np.sum(target_policy[m],axis=1,keepdims=True)

        if V is None:
            self.V=get_V_diagmain(d=self.num_agents,p_central=0.9)
        else:
            assert V.shape==(self.num_agents,self.num_agents),"V should have shape (num_agents,num_agents)"
            self.V=np.abs(V)
        if self.num_agents>1:
            assert np.abs(np.sum(self.V,axis=0)-1).max()<1e-12, "V should be doubly stochastic"
            assert np.abs(np.sum(self.V,axis=1)-1).max()<1e-12, "V should be doubly stochastic"
            assert np.abs(self.V-self.V.T).max()<1e-12, "V should be symmetric"
            u,s,vh=np.linalg.svd(self.V)
            s=s[1]
            assert (s>=0)&(s<1),"The second largest singular value of V should be in [0,1)"
        
        if features is None:
            self.features=np.random.uniform(size=(num_features,self.num_states))
            self.features=self.features/np.sqrt(np.sum(self.features**2,axis=0,keepdims=True))
        else: 
            assert features.shape[1]==self.num_states, "features should be a 2d-array with #states columns"
            self.features=features/np.sqrt(np.sum(features**2,axis=0,keepdims=True))
        self.num_features=self.features.shape[0]
        
        self.compute_stationary_dist()
        self.getABCb()
        self.get_theta_star()
        self.results={}

    def collect(self,num_expr=100,totalnum_samples=20000,init_state=0,seed_collect=1,maxnum_rhoavg=7):
        self.num_expr=num_expr
        self.totalnum_samples=totalnum_samples
        self.seed_collect=seed_collect
        self.maxnum_rhoavg=maxnum_rhoavg
        
        set_seed(seed_collect)
        
        error_init_state="init_state must be either an integer in range(num_states) or a probability vector with dim=-#states"
        if type(init_state)==np.ndarray:
            assert np.prod(init_state.shape)==self.num_states, error_init_state
            init_state=np.abs(init_state)
            init_state=init_state/np.sum(init_state)
            self.init_state=np.random.choice(a=self.num_states,size=1,p=init_state)[0]
        else:
            init_state=np.array(init_state)
            assert init_state.shape==(), error_init_state
            self.init_state=init_state.astype(int).reshape((1))[0]
        
        self.states=np.array([[self.init_state]*(self.totalnum_samples+1)]*self.num_expr)  #[k,t]: k-th experiment and t-th sample
        self.actions=np.array([[[0]*self.num_agents]*self.totalnum_samples]*self.num_expr) #[k,t,m]: k-th experiment, t-th sample, m-th agent
        self.R=np.zeros((self.num_expr,self.totalnum_samples,self.num_agents))             #[k,t,m]: k-th experiment, t-th sample, m-th agent
        self.localrho=np.zeros((self.num_expr,self.totalnum_samples,self.num_agents,self.maxnum_rhoavg+1))
        #[k,t,m,ell]: k-th experiment, t-th sample, m-th agent, after ell-th local average of rho
        
        self.globalrho=np.zeros((self.num_expr,self.totalnum_samples))     #[k,t]: k-th experiment, t-th sample
        self.localrho_arithmean=np.zeros((self.num_expr,self.totalnum_samples,self.maxnum_rhoavg+1))     
        #arithmetic mean of local rho among agents
        #[k,t,ell]: k-th experiment, t-th sample, after ell-th local average of rho
        
        ## Begin collecting samples
        a=np.array([0]*self.num_agents)
        for k in range(self.num_expr):
            print("collecting samples for the "+str(k)+"-th experiment...")
            for t in range(self.totalnum_samples):
                s_now=self.states[k,t]
                index="[s_now"
                for m in range(self.num_agents):
                    a[m]=np.random.choice(a=range(self.num_actions[m]),size=1,p=self.behavior_policy[m][s_now])  
                    self.actions[k,t,m]=a[m]
                    index+=","+str(a[m])
                Pnow=eval("self.transP"+index+",:]")
                s_next=np.random.choice(a=range(self.num_states),size=1,p=Pnow)[0]
                self.states[k,t+1]=s_next
    
                #To collect samples of rho and R
                Rm_now=eval("self.reward"+index+",s_next,:]")
                self.R[k,t,:]=Rm_now.copy()
                localrho_now=self.get_local_rho(s_now,a,s_next)   
                self.localrho[k,t,:,0]=localrho_now.copy()
                self.localrho_arithmean[k,t,0]=localrho_now.mean()
                self.globalrho[k,t]=np.prod(localrho_now)
                ln_rho=np.log(localrho_now)
                for ell in range(1,self.maxnum_rhoavg+1):
                    ln_rho=self.V.dot(ln_rho)
                    self.localrho[k,t,:,ell]=np.exp(ln_rho*self.num_agents)
                    self.localrho_arithmean[k,t,ell]=self.localrho[k,t,:,ell].mean()
    
    def save_collect(self,data_dir="./data"):
        if "./" not in data_dir:
            data_dir="./"+data_dir
        self.data_dir=data_dir
        name1='_'+str(self.num_expr)+'experiments_'+str(self.totalnum_samples)+'samples_'+'seed'\
            +str(self.seed_collect)+'_'+str(self.maxnum_rhoavg)+'rhoavgs_atMost.npy'
        if not os.path.isdir(data_dir):
            os.makedirs(data_dir)
        np.save(file=data_dir+'/states'+name1,arr=self.states)
        np.save(file=data_dir+'/actions'+name1,arr=self.actions)
        np.save(file=data_dir+'/rewards'+name1,arr=self.R)
        np.save(file=data_dir+'/localrho'+name1,arr=self.localrho)
        np.save(file=data_dir+'/globalrho'+name1,arr=self.globalrho)
        np.save(file=data_dir+'/localrho_arithmean'+name1,arr=self.localrho_arithmean)
    
    def load_collect(self,num_expr,totalnum_samples,seed_collect,maxnum_rhoavg,data_dir="./data"):
        if "./" not in data_dir:
            data_dir="./"+data_dir
        self.data_dir=data_dir
        self.num_expr=num_expr
        self.totalnum_samples=totalnum_samples
        self.seed_collect=seed_collect
        self.maxnum_rhoavg=maxnum_rhoavg
        name1='_'+str(self.num_expr)+'experiments_'+str(self.totalnum_samples)+'samples_'+'seed'\
            +str(self.seed_collect)+'_'+str(self.maxnum_rhoavg)+'rhoavgs_atMost.npy'

        if not os.path.isdir(data_dir):
            os.makedirs(data_dir)
        self.states=np.load(data_dir+'/states'+name1)
        self.actions=np.load(data_dir+'/actions'+name1)
        self.R=np.load(data_dir+'/rewards'+name1)
        self.localrho=np.load(data_dir+'/localrho'+name1)
        self.globalrho=np.load(data_dir+'/globalrho'+name1)
        self.localrho_arithmean=np.load(data_dir+'/localrho_arithmean'+name1)
        
    def sim1(self,hyp,init_Theta=None,init_W=None,seed_sim=None):
        set_seed(seed_sim)
        
        hyp=hyp.copy()
        hyp_str=get_hyp_str(hyp)
        print(hyp)
        self.results[hyp_str]={}
        if init_Theta is None:
            self.init_Theta=np.random.normal(size=(self.num_agents,self.num_features))
        else:
            assert init_Theta.shape,"init_Theta should be None or np array with shape (self.num_agents,self.num_features)"
            assert init_Theta.shape==(self.num_agents,self.num_features),"init_Theta should be None or np array with shape (self.num_agents,self.num_features)"
            self.init_Theta=init_Theta
            
        if init_W is None:
            self.init_W=np.random.normal(size=(self.num_agents,self.num_features))
        else:
            assert init_W.shape,"init_W should be None or np array with shape (self.num_agents,self.num_features)"
            assert init_W.shape==(self.num_agents,self.num_features),"init_W should be None or np array with shape (self.num_agents,self.num_features)"
            self.init_W=init_W
            
        ##Begin the definition for algorithm
        if not (hyp["alg"] in ["TDC_inexact","TDC_exact","TD0_inexact","TD0_exact"]):
            hyp["alg"]="TDC_inexact"
        
        num_TDiters=np.sum(hyp['num_iters'])
        num_totaliters=num_TDiters+hyp['num_postavg']
        self.Theta=np.zeros(shape=(num_totaliters+1,self.num_agents,self.num_features))
        #Theta[t,m,:] is the theta vector of agent m at t-th iteration, similarly for W
            
        self.Theta_avg=np.zeros(shape=(num_totaliters+1,self.num_features))
        #Theta[t,:] is the average theta vector among agents at t-th iteration, similarly for W

        self.results[hyp_str]['Theta_avg_err']=np.zeros(shape=(self.num_expr,num_totaliters+1))
        self.results[hyp_str]['Theta_consensus_err']=np.zeros(shape=(self.num_expr,num_totaliters+1,self.num_agents))
        self.results[hyp_str]['Theta_err']=np.zeros(shape=(self.num_expr,num_totaliters+1,self.num_agents))
    
        self.results[hyp_str]['local_berr']=np.zeros((self.num_expr,num_TDiters,self.num_agents))
        #[k,t,m]: k-th experiment, t-th sample, m-th agent
        
        self.results[hyp_str]['global_Cerr']=np.zeros((self.num_expr,num_TDiters))
        #[k,t]: k-th experiment and t-th sample
        
        if hyp['alg'] in ["TDC_inexact","TDC_exact"]:
            self.W=np.zeros(shape=(num_totaliters+1,self.num_agents,self.num_features))
            self.W_avg=np.zeros(shape=(num_totaliters+1,self.num_features))
            self.W_star=np.zeros(shape=(num_totaliters+1,self.num_features))       
            
            self.results[hyp_str]['W_avg_err']=np.zeros(shape=(self.num_expr,num_totaliters+1))        
            self.results[hyp_str]['W_consensus_err']=np.zeros(shape=(self.num_expr,num_totaliters+1,self.num_agents))        
            self.results[hyp_str]['W_err']=np.zeros(shape=(self.num_expr,num_totaliters+1,self.num_agents))
            
        if hyp['alg'] in ["TDC_inexact","TD0_inexact"]:
            self.results[hyp_str]['local_Aerr']=np.zeros((self.num_expr,num_TDiters,self.num_agents))
            self.results[hyp_str]['local_Berr']=np.zeros((self.num_expr,num_TDiters,self.num_agents))
        else:
            self.results[hyp_str]['global_Aerr']=np.zeros((self.num_expr,num_TDiters))
            self.results[hyp_str]['global_Berr']=np.zeros((self.num_expr,num_TDiters))

        for k in range(self.num_expr):
            print("conducting experiment "+str(k)+"...")
            self.Theta[0,:,:]=self.init_Theta
            self.W[0,:,:]=self.init_W
            tN=hyp['start_iter']
            
            stage=0
            if type(hyp['num_iters'])==int:
                remain_iters=hyp['num_iters']
            else:
                remain_iters=hyp['num_iters'][0]
            
            if type(hyp['num_rho_localavg'])==int:
                L=hyp['num_rho_localavg']
            else:
                L=hyp['num_rho_localavg'][0]
                
            for t in range(num_TDiters):
                if remain_iters<=0:
                    stage+=1
                    remain_iters=hyp['num_iters'][stage]-1
                    L=hyp['num_rho_localavg'][stage]
                else:
                    remain_iters-=1
                
                i_range=range(tN,tN+hyp['batchsize'])  #i=tN,tN+1,...(t+1)N-1 where N is batchsize
                tN+=hyp['batchsize']
                
                #To update Theta and W
                if hyp['alg']=="TDC_inexact":
                    At,Bt,Ct,bt=self.get_ABCbt_batchavg_inexactrho\
                    (s=self.states[k,list(i_range)+[tN]],\
                     rho_hat=self.localrho[k,i_range,:,L],R=self.R[k,i_range,:],batchsize=hyp['batchsize'])
                    
                    #Communication
                    self.Theta[t+1,:,:]=self.V.dot(self.Theta[t,:,:])
                    self.W[t+1,:,:]=self.V.dot(self.W[t,:,:])
                    
                    #Gradient descent
                    for m in range(self.num_agents):
                        self.Theta[t+1,m,:]+=hyp['alpha']*(At[:,:,m].dot(self.Theta[t,m,:])+\
                                  Bt[:,:,m].dot(self.W[t,m,:])+bt[:,m])
                        self.W[t+1,m,:]+=hyp['beta']*(At[:,:,m].dot(self.Theta[t,m,:])+\
                                  Ct.dot(self.W[t,m,:])+bt[:,m])
                        
                elif hyp['alg']=="TDC_exact":
                    At,Bt,Ct,bt=self.get_ABCbt_batchavg_exactrho\
                    (s=self.states[k,list(i_range)+[tN]],
                     rho=self.globalrho[k,i_range],R=self.R[k,i_range,:],batchsize=hyp['batchsize'])
                    
                    self.Theta[t+1,:,:]=self.V.dot(self.Theta[t,:,:])+hyp['alpha']*(self.Theta[t,:,:].dot(At.T)+\
                              self.W[t,:,:].dot(Bt.T)+bt.T)
                    self.W[t+1,:,:]=self.V.dot(self.W[t,:,:])+hyp['beta']*(self.Theta[t,:,:].dot(At.T)+\
                              self.W[t,:,:].dot(Ct.T)+bt.T)
                    
                elif hyp['alg']=="TD0_inexact":    
                    At,Bt,Ct,bt=self.get_ABCbt_batchavg_inexactrho\
                    (s=self.states[k,list(i_range)+[tN]],\
                     rho_hat=self.localrho[k,i_range,:,L],R=self.R[k,i_range,:],batchsize=hyp['batchsize'])
                    
                    #Communication
                    self.Theta[t+1,:,:]=self.V.dot(self.Theta[t,:,:])
                    
                    #Gradient descent
                    for m in range(self.num_agents):
                        self.Theta[t+1,m,:]+=hyp['alpha']*(At[:,:,m].dot(self.Theta[t,m,:])+bt[:,m])
                        
                else: #if "hyp['alg']=="TD0_exact""
                    At,Bt,Ct,bt=self.get_ABCbt_batchavg_exactrho\
                    (s=self.states[k,list(i_range)+[tN]],\
                     rho=self.globalrho[k,i_range],R=self.R[k,i_range,:],batchsize=hyp['batchsize'])
                    
                    self.Theta[t+1,:,:]=self.V.dot(self.Theta[t,:,:])+hyp['alpha']*(self.Theta[t,:,:].dot(At.T)+bt.T)
                    
                #Compute model average and errors at t-th iteration
                self.compute_avg_and_error(hyp['alg'],hyp_str,k,t,At,Bt,Ct,bt)
                 
            if hyp['num_postavg']>0:
                for t in range(num_TDiters,num_totaliters):  #Post communication
                    self.Theta[t+1,:,:]=self.V.dot(self.Theta[t,:,:])
                    if hyp['alg'] in ["TDC_inexact","TDC_exact"]:  
                        self.W[t+1,:,:]=self.V.dot(self.W[t,:,:])
                    
                    #Compute model average and errors at t-th iteration
                    self.compute_avg_and_error(hyp['alg'],hyp_str,k,t,None,None,None,None)
           
            self.compute_avg_and_error(hyp['alg'],hyp_str,k,num_totaliters,None,None,None,None)
            
    def sim(self,hyps,init_Theta=None,init_W=None,seed_sim=None):
        for hyp in hyps:
            self.sim1(hyp,init_Theta,init_W,seed_sim)
    
    def save_result1(self,hyp):        
        result_dir=hyp['result_dir']
        if "./" not in result_dir:
            result_dir="./"+result_dir
        if not os.path.isdir(result_dir):
            os.makedirs(result_dir)
        hyp_str=get_hyp_str(hyp)
    
        hyp_txt=open(result_dir+'/hyperparameters.txt','w')
        hyp_txt.write(hyp_str)
        hyp_txt.close()
        
        for result_type in self.results[hyp_str].keys():
            np.save(file=result_dir+"/"+result_type+".npy",arr=self.results[hyp_str][result_type])
            
    def save_results(self,hyps):
        for hyp in hyps:
            self.save_result1(hyp)
    
    def load_result1(self,hyp):
        result_dir=hyp['result_dir']
        if "./" not in result_dir:
            result_dir="./"+result_dir
        if not os.path.isdir(result_dir):
            os.makedirs(result_dir)
        hyp_str=get_hyp_str(hyp)
        
        result_types=['Theta_avg_err','Theta_consensus_err','Theta_err','local_berr','global_Cerr',\
                      'W_avg_err','W_consensus_err','W_err','local_Aerr','local_Berr','global_Aerr','global_Berr']
        self.results[hyp_str]={}
        for result_type in result_types:
            try:
                self.results[hyp_str][result_type]=np.load(result_dir+"/"+result_type+".npy")
            except:
                pass

    def load_results(self,hyps):
        for hyp in hyps:
            self.load_result1(hyp)
    
    def plot_results(self,hyps,percentile=95,islog10=False,color_transition='green',xlabel='number of samples',\
                     plotresult_dir='Figures',title='',fontsize=15,lgdsize=10,result_type='Theta_avg_err',legend_loc=1):
        if xlabel!='number of samples':
            xlabel='number of communication rounds'
        
        if "./" not in plotresult_dir:
            plotresult_dir="./"+plotresult_dir
        if plotresult_dir is not None:   #Then begin to save figure
            if not os.path.isdir(plotresult_dir):
                os.makedirs(plotresult_dir)
                
        hyp_txt=open(plotresult_dir+'/hyperparameters.txt','w')
        k=1
        for hyp in hyps:
            hyp_str=get_hyp_str(hyp)
            hyp_txt.write(hyp_str+"\n")
        hyp_txt.close()
                    
        plt.figure()
        plt.clf()
        ymin=float('inf')
        ymax=float('-inf')
        xmin=float('inf')
        
        #To obtain the truncation level xmin in x-axis
        for hyp in hyps:
            hyp=hyp.copy()
            hyp_str=get_hyp_str(hyp)
            if (result_type in self.results[hyp_str].keys()):
                if hyp['plot_iters'] is None:
                    hyp['plot_iters']=np.arange(self.results[hyp_str]['Theta_avg_err'].shape[1])
    
                if xlabel=='number of samples':
                    x=np.array(hyp['plot_iters'])*hyp['batchsize']
                    x_transition=hyp['num_iters']*hyp['batchsize']
                else:
                    x=hyp['plot_iters']
                    x_transition=hyp['num_iters']
                xmin=min([x.max(),xmin])
        
        #Begin to plot theta errors and W errors
        for hyp in hyps:
            hyp=hyp.copy()
            hyp_str=get_hyp_str(hyp)
            if (result_type in self.results[hyp_str].keys()):
                if hyp['plot_iters'] is None:
                    hyp['plot_iters']=np.arange(self.results[hyp_str]['Theta_avg_err'].shape[1])
                        
                if result_type=='Theta_avg_err':
                    result_preplot=np.sqrt(self.results[hyp_str][result_type][:,hyp['plot_iters']]/np.square(self.theta_star).sum())
                else:
                    result_preplot=np.sqrt(self.results[hyp_str][result_type][:,hyp['plot_iters'],:].max(axis=2)/np.square(self.theta_star).sum())
                    
                if islog10:
                    result_preplot=np.log10(result_preplot)
                    
                upper_loss = np.percentile(result_preplot, percentile, axis=0)  
                lower_loss = np.percentile(result_preplot, 100 - percentile, axis=0)
                avg_loss = np.mean(result_preplot, axis=0)
                ymin=min([lower_loss.min(),ymin])
                ymax=max([upper_loss.max(),ymax])
                if xlabel=='number of samples':
                    x=np.array(hyp['plot_iters'])*hyp['batchsize']
                    x_transition=hyp['num_iters']*hyp['batchsize']
                else:
                    x=hyp['plot_iters']
                    x_transition=hyp['num_iters']
                index=(x<=xmin)
                x=x[index]
                plt.plot(x,avg_loss[index],color=hyp['color'],marker=hyp['marker'],label=hyp['legend'])
                plt.fill_between(x,lower_loss[index],upper_loss[index],color=hyp['color'],alpha=0.3,edgecolor="none")
        
        if color_transition is not None:
            plt.plot([x_transition]*2,[ymin,ymax],color=color_transition,linestyle='--',label='Local averaging starts')
        plt.gcf().subplots_adjust(bottom=0.2)
        plt.gcf().subplots_adjust(left=0.17)
        plt.legend(prop={'size':lgdsize},loc=legend_loc)
        plt.title(title)
        plt.xlabel(xlabel)
        if result_type=='Theta_avg_err':
            if islog10:
                plt.ylabel(r'$\log_{10}(||\overline{\theta}_t - \theta^\ast ||/||\theta^\ast||)$')
            else:
                plt.ylabel(r'$||\overline{\theta}_t - \theta^\ast ||/||\theta^\ast||$')
        else:
            if islog10:
                plt.ylabel(r'$\log_{10}(\max_m ||\theta_t^{(m)} - \overline{\theta}_t ||/||\theta^\ast||)$')
            else:
                plt.ylabel(r'$\max_m ||\theta_t^{(m)} - \overline{\theta}_t ||/||\theta^\ast||$')
            
        if fontsize is not None:
            plt.rcParams.update({'font.size': fontsize})

        if plotresult_dir is not None:   #Then begin to save figure
            if xlabel=='number of samples':
                xlabel_name='Samples'
            else:
                xlabel_name='Iters'
            plt.savefig(plotresult_dir+'/'+result_type+'_'+xlabel_name+'.png',dpi=400)
            plt.close()
                
    def get_local_rho(self,s_now,a,s_next):  #a is the list of current actions among agents.
        return np.array([self.target_policy[m][s_now,a[m]]/self.behavior_policy[m][s_now,a[m]] for m in range(self.num_agents)])

    def get_ABCbt_batchavg_exactrho(self,s,rho,R,batchsize):
        #s: np array [s_{tN},...s_{(t+1)N}] with length N+1 where N is batchsize
        #rho: np array [rho_{tN},...,rho_{(t+1)N-1}] with length N
        #R: np array [R_{tN},...,R_{(t+1)N-1}] with N rows, where R_{i} is a row of R_i^{(m)} for agents m=1...M
        phi_now=self.features[:,s[0:batchsize]]
        phi_next=self.features[:,s[1:(1+batchsize)]]
        rho=np.diag(np.array(rho)/batchsize)
        Ct_batchavg=-phi_now.dot(phi_now.T/batchsize)
        Bt_batchavg=((-self.gamma)*phi_next).dot(rho).dot(phi_now.T)
        At_batchavg=-Bt_batchavg.T-phi_now.dot(rho).dot(phi_now.T)
        btm_batchavg=phi_now.dot(rho).dot(R)    #btm_batchavg[:,m] for agent m
        return At_batchavg,Bt_batchavg,Ct_batchavg,btm_batchavg
    
    def get_ABCbt_batchavg_inexactrho(self,s,rho_hat,R,batchsize):
        #s: np array [s_{tN},...s_{(t+1)N}] with length N+1 where N is batchsize
        #rho_hat: np array [rho_hat_{tN},...,rho_hat_{(t+1)N-1}] with N rows,
        #  where rho_hat_{i} is a row of \widehat{rho}_i^{(m)} for all agents m=1,2,...,M.
        #R: np array [R_{tN},...,R_{(t+1)N-1}] with N rows, where R_{i} is a row of R_i^{(m)} for agents m=1...M
        phi_now=self.features[:,s[0:batchsize]]
        phi_next=self.features[:,s[1:(1+batchsize)]]
        Atm_batchavg=np.zeros((self.num_features,self.num_features,self.num_agents))
        Btm_batchavg=np.zeros((self.num_features,self.num_features,self.num_agents))
        Ct_batchavg=-phi_now.dot(phi_now.T)/batchsize
        for m in range(self.num_agents):
            rhom=np.diag(np.array(rho_hat[:,m])/batchsize)
            Btm_batchavg[:,:,m]=((-self.gamma)*phi_next).dot(rhom).dot(phi_now.T)
            Atm_batchavg[:,:,m]=-Btm_batchavg[:,:,m].T-phi_now.dot(rhom).dot(phi_now.T)
        btm_batchavg=phi_now.dot(R*rho_hat)/batchsize     #btm_batchavg[:,m] for agent m.
        return Atm_batchavg,Btm_batchavg,Ct_batchavg,btm_batchavg
    
    def compute_stationary_dist(self):    
    # Compute the stationary distribution
        self.transP_behavior=self.transP.copy()
        self.transP_target=self.transP.copy()
        for m in range(self.num_agents):
            newshape=(self.num_states,)+(1,)*m+(self.num_actions[m],)+(1,)*(self.num_agents-m)
            self.transP_behavior*=self.behavior_policy[m].reshape(newshape)
            self.transP_target*=self.target_policy[m].reshape(newshape)
        newshape=tuple(range(1,self.num_agents+1))
        self.transP_behavior=self.transP_behavior.sum(axis=newshape)
        self.transP_target=self.transP_target.sum(axis=newshape)
        
        evals, evecs = np.linalg.eig(self.transP_behavior.T)  #P.T*evecs=evecs*np.diag(evals)
        evec1 = evecs[:, np.isclose(evals, 1)]
        evec1 = np.abs(evec1[:, 0])
        stationary = evec1 / evec1.sum()
        self.stationary = stationary.real
        
    def getABCb(self):  #Expected values: A, B, C, b  
        D=np.diag(self.stationary)
        self.C=-self.features.dot(D).dot(self.features.T)
        self.B=((-self.gamma)*self.features).dot(self.transP_target.T).dot(D).dot(self.features.T)
        self.A=self.C-self.B.T
        self.b=(self.reward*self.transP.reshape(self.transP.shape+(1,))).sum(axis=self.num_agents+1)
        for m in range(self.num_agents):
            self.b*=self.target_policy[m].reshape\
                ((self.num_states,)+(1,)*m+(self.num_actions[m],)+(1,)*(self.num_agents-m))
        self.b=self.b.sum(axis=tuple(range(1,self.num_agents+1)))
        self.b=self.features.dot(D).dot(self.b)
        
    def get_theta_star(self): 
        self.theta_star=-np.linalg.solve(a=self.A,b=self.b.mean(axis=1))
    
    def get_w_star(self,theta):
        #Theta is np array with shape (self.num_features)
        return -np.linalg.solve(a=self.C,b=self.A.dot(theta)+self.b.mean(axis=1))    
    
    def compute_avg_and_error(self,alg,hyp_str,k,t,At,Bt,Ct,bt):  
        #Compute w_star, agent-average parameters and errors of theta, w, A, B, C, b at the t-th iteration of the k-th experiment
        self.Theta_avg[t,:]=self.Theta[t,:,:].mean(axis=0)            
        self.results[hyp_str]['Theta_avg_err'][k,t]=np.square(self.Theta_avg[t,:]-self.theta_star).sum()
        self.results[hyp_str]['Theta_consensus_err'][k,t,:]=np.square(self.Theta[t,:,:]-self.Theta_avg[t,:].reshape(1,self.num_features)).sum(axis=1)            
        self.results[hyp_str]['Theta_err'][k,t,:]=np.square(self.Theta[t,:,:]-self.theta_star.reshape(1,self.num_features)).sum(axis=1)
        
        if At is not None:
            self.results[hyp_str]['local_berr'][k,t,:]=np.square(bt-self.b).sum(axis=0)
            #[k,t,m]: k-th experiment, t-th sample, m-th agent
            
            self.results[hyp_str]['global_Cerr'][k,t]=np.square(Ct-self.C).sum()
            #[k,t]: k-th experiment and t-th sample

            if alg in ["TD0_inexact","TDC_inexact"]:
                self.results[hyp_str]['local_Aerr'][k,t,:]=np.square(At-self.A.reshape(self.A.shape+(1,))).sum(axis=(0,1))
                self.results[hyp_str]['local_Berr'][k,t,:]=np.square(Bt-self.B.reshape(self.B.shape+(1,))).sum(axis=(0,1))
                #[k,t,m]: k-th experiment, t-th sample, m-th agent
            else:
                self.results[hyp_str]['global_Aerr'][k,t]=np.square(At-self.A).sum()
                self.results[hyp_str]['global_Berr'][k,t]=np.square(Bt-self.B).sum()
                #[k,t]: k-th experiment and t-th sample
                
        if alg in ["TDC_exact","TDC_inexact"]:
            self.W_avg[t,:]=self.W[t,:,:].mean(axis=0)        
            self.W_star[t,:]=self.get_w_star(self.Theta_avg[t,:])
            self.results[hyp_str]['W_avg_err'][k,t]=np.square(self.W_avg[t,:]-self.W_star[t,:]).sum()
            self.results[hyp_str]['W_consensus_err'][k,t,:]=np.square(self.W[t,:,:]-self.W_avg[t,:].reshape(1,self.num_features)).sum(axis=1)
            self.results[hyp_str]['W_err'][k,t,:]=np.square(self.W[t,:,:]-self.W_star[t,:].reshape(1,self.num_features)).sum(axis=1)
            
