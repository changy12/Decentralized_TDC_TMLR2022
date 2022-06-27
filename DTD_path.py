import numpy as np
import random
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from DTD_utils import *
import os
import pdb   #to delete


class DTD_path:   #For path finding problem
    def __init__(self,gridmap,seed_init=1,gamma=0.95,R_finish=3,R_stay_goal=0,R_nocollide=-0.075,R_collide=-0.5,V=None,init_state=[0,0,0,1,1,1]):
        #gridmap[x,y]=-1 if there is obstacle at (x,y), m if it is the m-th agent's goal (unique), 0 otherwise (empty).
        #init_state=[x1,y1,x2,y3,...,xM,yM] where (xm,ym) is the initial coordinate of the agent m.
        #We use numpy coordinate system where left up corner is the orignin. x direction is down and y direction is right.
        set_seed(seed_init)
        self.map=gridmap
        self.xmax=self.map.shape[0]-1
        self.ymax=self.map.shape[1]-1
        self.seed_init=seed_init
        
        self.gamma=gamma
        
        self.init_state=init_state
        self.num_agents=int(len(init_state)/2)
        self.num_features=5*self.num_agents

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
        
        self.R_finish=R_finish
        self.R_stay_goal=R_stay_goal
        self.R_nocollide=R_nocollide
        self.R_collide=R_collide
        
        if False:
            self.compute_stationary_dist()
            self.getABCb()
            self.results={}

    def collect(self,num_expr=100,totalnum_samples=20000,seed_collect=1,maxnum_rhoavg=7):
        self.num_expr=num_expr
        self.totalnum_samples=totalnum_samples
        self.seed_collect=seed_collect
        self.maxnum_rhoavg=maxnum_rhoavg
        
        set_seed(seed_collect)
        
        self.states=np.zeros((self.num_expr,self.totalnum_samples+1,2*self.num_agents)).astype(int)
        #              self.states[k,t,:]: vector [i1,j1,i2,j2,...,iM,jM] of M agents at k-th experiment and t-th sample
        self.states[:,0,:]=self.init_state
        self.actions=np.array([[[0]*self.num_agents]*self.totalnum_samples]*self.num_expr) #[k,t,m]: k-th experiment, t-th sample, m-th agent
        #stay:0, up:1, down:2, left: 3, right: 4
        self.R=np.zeros((self.num_expr,self.totalnum_samples,self.num_agents))             #[k,t,m]: k-th experiment, t-th sample, m-th agent
        self.localrho=np.zeros((self.num_expr,self.totalnum_samples,self.num_agents,self.maxnum_rhoavg+1))
        #[k,t,m,ell]: k-th experiment, t-th sample, m-th agent, after ell-th local average of rho
        
        self.globalrho=np.zeros((self.num_expr,self.totalnum_samples))     #[k,t]: k-th experiment, t-th sample
        self.localrho_arithmean=np.zeros((self.num_expr,self.totalnum_samples,self.maxnum_rhoavg+1))     
        #arithmetic mean of local rho among agents
        #[k,t,ell]: k-th experiment, t-th sample, after ell-th local average of rho
        
        self.phi=np.zeros((self.num_expr,self.totalnum_samples+1,5*self.num_agents))  
        #[k,t,:]: feature vector [phi_0,phi_1,...,phi_{M-1}] corresponding to current state s_t where phi_m is 6-dim binary vector
        #phi_m[0]=1 if m-th agent is at its goal
        #phi_m[1]=1 if m-th agent is adjacent to its goal
        #phi_m[2]=1 if m-th agent is away from goal by 1 step along x direction and 1 step along y direction
        #phi_m[3]=1 if there is no other agents or obstacles in the 3*3 local view of m-th agent (excluding m-th agent's location as the center)
        #phi_m[4]=1 if the m-th agent is in collision with obstacle or another agent now.
        
        ## Begin collecting samples
        for k in range(self.num_expr):
            print("collecting samples for the "+str(k)+"-th experiment...")
            for t in range(self.totalnum_samples):                
                is_finish=True
                for m in range(self.num_agents):
                    x=self.states[k,t,2*m]
                    y=self.states[k,t,2*m+1]
                    self.R[k,t,m]=self.R_nocollide
                    if self.map[x,y]==m:  #if m-th agent reaches goal at current step t
                        self.actions[k,t,m]=0  #stay at goal  
                        self.states[k,t+1,2*m]=x
                        self.states[k,t+1,2*m+1]=y
                        self.R[k,t,m]=self.R_stay_goal
                        self.localrho[k,t,m,0]=1.0
                        self.phi[k,t,5*m]=1.0
                    else:  #if m-th agent has not reached goal at current step t
                        is_finish=False
                        actions_behavior=[0]  #available actions for behavior policy: just do not go outside the map
                        actions_target=[]    #available actions for target policy: do not go outside the map, do not hit obstacle
                        if self.map[x,y]>=0:  #Currently not in obstacle
                            actions_target.append(0)
                        else:   #Currently in obstacle
                            self.R[k,t,m]=self.R_collide
                            self.phi[k,t,5*m+4]=1.0
                            
                        for mm in range(self.num_agents):    #see if agent m collides with agent mm at step t.
                            if mm!=m:
                                dx=np.abs(self.states[k,t,2*mm]-x)
                                dy=np.abs(self.states[k,t,2*mm+1]-y)
                                if (dx<=1) and (dy<=1): #The mm-th agent is in the 3*3 local view centered at the mth agent
                                    if (dx==0) and (dy==0):
                                        self.phi[k,t,5*m+4]=1.0
                                        self.R[k,t,m]=self.R_collide
                                        self.phi[k,t,5*m+4]=1.0
                                    else:
                                        self.phi[k,t,5*m+3]=1.0
                            
                        if x>0:  #consider up direction only if m-th agent is not at the top
                            actions_behavior.append(1)
                            if self.map[x-1,y]==m:    #going up to the goal
                                self.actions[k,t,m]=1
                                self.states[k,t+1,2*m]=x-1
                                self.states[k,t+1,2*m+1]=y
                                self.R[k,t,m]=self.R_nocollide
                                self.localrho[k,t,m,0]=1.0
                                self.phi[k,t,5*m+1]=1.0
                                continue
                            elif self.map[x-1,y]>=0:  #going up will not hit obstacle
                                actions_target.append(1)
                        
                        if x<self.xmax:  #consider down direction only if m-th agent is not at the bottom
                            actions_behavior.append(2)
                            if self.map[x+1,y]==m:   #going down to the goal
                                self.actions[k,t,m]=2
                                self.states[k,t+1,2*m]=x+1
                                self.states[k,t+1,2*m+1]=y
                                self.R[k,t,m]=self.R_nocollide
                                self.localrho[k,t,m,0]=1.0
                                self.phi[k,t,5*m+1]=1.0
                                continue
                            elif self.map[x+1,y]>=0:  #going down will not hit obstacle
                                actions_target.append(2)
                        
                        if y>0:  #consider left direction only if m-th agent is not at the leftmost column 
                            actions_behavior.append(3)
                            if self.map[x,y-1]==m:   #going left to the goal
                                self.actions[k,t,m]=3
                                self.states[k,t+1,2*m]=x
                                self.states[k,t+1,2*m+1]=y-1
                                self.R[k,t,m]=self.R_nocollide
                                self.localrho[k,t,m,0]=1.0
                                self.phi[k,t,5*m+1]=1.0
                                continue
                            elif self.map[x,y-1]>=0:  #going left will not hit obstacle
                                actions_target.append(3)
                                
                        if y<self.ymax:  #consider right direction only if m-th agent is not at the rightmost column 
                            actions_behavior.append(4)
                            if self.map[x,y+1]==m:   #going right to the goal
                                self.actions[k,t,m]=4
                                self.states[k,t+1,2*m]=x
                                self.states[k,t+1,2*m+1]=y+1
                                self.R[k,t,m]=self.R_nocollide
                                self.localrho[k,t,m,0]=1.0
                                self.phi[k,t,5*m+1]=1.0
                                continue
                            elif self.map[x,y+1]>=0:  #going right will not hit obstacle
                                actions_target.append(4)
                        
                        if len(actions_target)==0:  #If all actions lead to obstacle, simply let behavior policy=target policy now
                            actions_target=actions_behavior.copy()
                        
                        self.actions[k,t,m]=np.random.choice(a=actions_behavior,size=1)
                        self.localrho[k,t,m,0]=len(actions_target)/len(actions_behavior)
                        self.states[k,t+1,2*m]=x
                        self.states[k,t+1,2*m+1]=y
                        if self.actions[k,t,m]==1:
                            self.states[k,t+1,2*m]=x-1
                        elif self.actions[k,t,m]==2:
                            self.states[k,t+1,2*m]=x+1
                        elif self.actions[k,t,m]==3:
                            self.states[k,t+1,2*m+1]=y-1
                        elif self.actions[k,t,m]==4:
                            self.states[k,t+1,2*m+1]=y+1
                            
                        #Check for goal and obstacles in (x+-1,y+-1). 
                        tmp=[]
                        try:
                            tmp.append(self.map[x-1,y-1])
                        except:
                            pass
                        try:
                            tmp.append(self.map[x-1,y+1])
                        except:
                            pass
                        try:
                            tmp.append(self.map[x+1,y-1])
                        except:
                            pass
                        try:
                            tmp.append(self.map[x+1,y+1])
                        except:
                            pass
                        if m in tmp:  #m-th agent's goal is in (x+-1,y+-1)
                            self.phi[k,t,5*m+2]=1.0
                        if np.min(tmp)<0:  #There are obstacle(s) in (x+-1,y+-1).
                            self.phi[k,t,5*m+3]=1.0
                
                if is_finish:  #all agents reach goal at current step t
                    for m in range(self.num_agents):
                        self.states[k,(t+1):,m]=self.states[k,t,m]
                        self.phi[k,(t+1):,5*m]=1.0
                    self.actions[k,(t+1):,:]=0
                    self.R[k,(t+1):,:]=self.R_finish
                    self.localrho[k,(t+1):,:,:]=1.0
                    self.localrho_arithmean[k,(t+1):,:]=1.0
                    break

                self.localrho_arithmean[k,t,0]=self.localrho[k,t,:,0].mean()
                self.globalrho[k,t]=np.prod(self.localrho[k,t,:,0])
                ln_rho=np.log(self.localrho[k,t,:,0])
                for ell in range(1,self.maxnum_rhoavg+1):
                    ln_rho=self.V.dot(ln_rho)
                    self.localrho[k,t,:,ell]=np.exp(ln_rho*self.num_agents)
                    self.localrho_arithmean[k,t,ell]=self.localrho[k,t,:,ell].mean()
                    
            if not is_finish:   #set phi for the final state sample at t=self.totalnum_samples+1
                t=self.totalnum_samples+1
                for m in range(self.num_agents):
                    x=self.states[k,t,2*m]
                    y=self.states[k,t,2*m+1]
                    if self.map[x,y]==m:
                        self.phi[k,t,5*m]=1.0
                    else:
                        x_available=[]
                        if x>0:
                            x_available.append(x-1)
                        if x<self.xmax:
                            x_available.append(x+1)
                        y_available=[]
                        if y>0:
                            y_available.append(y-1)
                        if y<self.ymax:
                            y_available.append(y+1)
                        if self.map[x,y]<0:
                            self.phi[k,t,5*m+4]=1.0
                            
                        adjacents=[self.map[xx,y] for xx in x_available]+[self.map[x,yy] for yy in y_available]
                        if m in adjacents:
                            self.phi[k,t,5*m+1]=1.0
                        if np.min(adjacents)<0:
                            self.phi[k,t,5*m+4]=1.0
                        
                        corner_adjacents=[self.map[xx,yy] for xx in x_available for yy in y_available]
                        if m in corner_adjacents:
                            self.phi[k,t,5*m+2]=1.0
                        if np.min(corner_adjacents)<0:
                            self.phi[k,t,5*m+3]=1.0
                        
                        for mm in range(self.num_agents):    #see if agent m collides with agent mm at step t.
                            if mm!=m:
                                dx=np.abs(self.states[k,t,2*mm]-x)
                                dy=np.abs(self.states[k,t,2*mm+1]-y)
                                if (dx<=1) and (dy<=1): #The mm-th agent is in the 3*3 local view centered at the mth agent
                                    if (dx==0) and (dy==0):
                                        self.phi[k,t,5*m+4]=1.0
                                        self.phi[k,t,5*m+4]=1.0
                                    else:
                                        self.phi[k,t,5*m+3]=1.0
                    
    
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
        np.save(file=data_dir+'/phi'+name1,arr=self.phi)
    
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
        self.phi=np.load(data_dir+'/phi'+name1)
        
    def sim1(self,hyp,init_Theta=None,init_W=None,seed_sim=None):
        set_seed(seed_sim)
        
        hyp=hyp.copy()
        hyp_str=get_hyp_str(hyp)
        print(hyp)
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
        self.Theta[hyp_str]=np.zeros(shape=(self.num_expr,num_totaliters+1,self.num_agents,self.num_features))
        #Theta[t,m,:] is the theta vector of agent m at t-th iteration, similarly for W

        if False:   #Not needed for DTD_path.py
            self.Theta_avg=np.zeros(shape=(num_totaliters+1,self.num_features))
            #Theta_avg[t,:] is the average theta vector among agents at t-th iteration, similarly for W
        
            self.results[hyp_str]['Theta_avg_err']=np.zeros(shape=(self.num_expr,num_totaliters+1))
            self.results[hyp_str]['Theta_consensus_err']=np.zeros(shape=(self.num_expr,num_totaliters+1,self.num_agents))
            self.results[hyp_str]['Theta_err']=np.zeros(shape=(self.num_expr,num_totaliters+1,self.num_agents))
            self.results[hyp_str]['local_berr']=np.zeros((self.num_expr,num_TDiters,self.num_agents))        #[k,t,m]: k-th experiment, t-th sample, m-th agent        
            self.results[hyp_str]['global_Cerr']=np.zeros((self.num_expr,num_TDiters))  #[k,t]: k-th experiment and t-th sample
        
        if hyp['alg'] in ["TDC_inexact","TDC_exact"]:
            self.W=np.zeros(shape=(num_totaliters+1,self.num_agents,self.num_features))
            self.W_avg=np.zeros(shape=(num_totaliters+1,self.num_features))
            self.W_star=np.zeros(shape=(num_totaliters+1,self.num_features))       
            
            # self.results[hyp_str]['W_avg_err']=np.zeros(shape=(self.num_expr,num_totaliters+1))        
            # self.results[hyp_str]['W_consensus_err']=np.zeros(shape=(self.num_expr,num_totaliters+1,self.num_agents))        
            # self.results[hyp_str]['W_err']=np.zeros(shape=(self.num_expr,num_totaliters+1,self.num_agents))
            
        if False:   #Not needed for DTD_path.py
            if hyp['alg'] in ["TDC_inexact","TD0_inexact"]:
                self.results[hyp_str]['local_Aerr']=np.zeros((self.num_expr,num_TDiters,self.num_agents))
                self.results[hyp_str]['local_Berr']=np.zeros((self.num_expr,num_TDiters,self.num_agents))
            else:
                self.results[hyp_str]['global_Aerr']=np.zeros((self.num_expr,num_TDiters))
                self.results[hyp_str]['global_Berr']=np.zeros((self.num_expr,num_TDiters))

        for k in range(self.num_expr):
            print("conducting experiment "+str(k)+"...")
            self.Theta[hyp_str][k,0,:,:]=self.init_Theta
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
                    (s=self.states[k,list(i_range)+[tN]],rho_hat=self.localrho[k,i_range,:,L],R=self.R[k,i_range,:],\
                     phi_now=self.phi[k,i_range,:].T,phi_next=self.phi[k,range(tN-hyp['batchsize']+1,tN+1),:].T,batchsize=hyp['batchsize'])
                    
                    #Communication
                    self.Theta[hyp_str][k,t+1,:,:]=self.V.dot(self.Theta[hyp_str][k,t,:,:])
                    self.W[t+1,:,:]=self.V.dot(self.W[t,:,:])
                    
                    #Gradient descent
                    for m in range(self.num_agents):
                        self.Theta[hyp_str][k,t+1,m,:]+=hyp['alpha']*(At[:,:,m].dot(self.Theta[hyp_str][k,t,m,:])+\
                                  Bt[:,:,m].dot(self.W[t,m,:])+bt[:,m])
                        self.W[t+1,m,:]+=hyp['beta']*(At[:,:,m].dot(self.Theta[hyp_str][k,t,m,:])+\
                                  Ct.dot(self.W[t,m,:])+bt[:,m])
                        
                elif hyp['alg']=="TDC_exact":
                    At,Bt,Ct,bt=self.get_ABCbt_batchavg_exactrho\
                    (s=self.states[k,list(i_range)+[tN]],rho=self.globalrho[k,i_range],R=self.R[k,i_range,:],\
                     phi_now=self.phi[k,i_range,:].T,phi_next=self.phi[k,range(tN-hyp['batchsize']+1,tN+1),:].T,batchsize=hyp['batchsize'])
                    
                    self.Theta[hyp_str][k,t+1,:,:]=self.V.dot(self.Theta[hyp_str][k,t,:,:])+hyp['alpha']*(self.Theta[hyp_str][k,t,:,:].dot(At.T)+\
                              self.W[t,:,:].dot(Bt.T)+bt.T)
                    self.W[t+1,:,:]=self.V.dot(self.W[t,:,:])+hyp['beta']*(self.Theta[hyp_str][k,t,:,:].dot(At.T)+\
                              self.W[t,:,:].dot(Ct.T)+bt.T)
                    
                elif hyp['alg']=="TD0_inexact":    
                    At,Bt,Ct,bt=self.get_ABCbt_batchavg_inexactrho\
                    (s=self.states[k,list(i_range)+[tN]],rho_hat=self.localrho[k,i_range,:,L],R=self.R[k,i_range,:],\
                     phi_now=self.phi[k,i_range,:].T,phi_next=self.phi[k,range(tN-hyp['batchsize']+1,tN+1),:].T,batchsize=hyp['batchsize'])
                    
                    #Communication
                    self.Theta[hyp_str][k,t+1,:,:]=self.V.dot(self.Theta[hyp_str][k,t,:,:])
                    
                    #Gradient descent
                    for m in range(self.num_agents):
                        self.Theta[hyp_str][k,t+1,m,:]+=hyp['alpha']*(At[:,:,m].dot(self.Theta[hyp_str][k,t,m,:])+bt[:,m])
                        
                else: #if "hyp['alg']=="TD0_exact""
                    At,Bt,Ct,bt=self.get_ABCbt_batchavg_exactrho\
                    (s=self.states[k,list(i_range)+[tN]],rho=self.globalrho[k,i_range],R=self.R[k,i_range,:],\
                     phi_now=self.phi[k,i_range,:].T,phi_next=self.phi[k,range(tN-hyp['batchsize']+1,tN+1),:].T,batchsize=hyp['batchsize'])
                    
                    self.Theta[hyp_str][k,t+1,:,:]=self.V.dot(self.Theta[hyp_str][k,t,:,:])+hyp['alpha']*(self.Theta[hyp_str][k,t,:,:].dot(At.T)+bt.T)
                    
                if False:
                    #Compute model average and errors at t-th iteration
                    self.compute_avg_and_error(hyp['alg'],hyp_str,k,t,At,Bt,Ct,bt)
                 
            if hyp['num_postavg']>0:
                for t in range(num_TDiters,num_totaliters):  #Post communication
                    self.Theta[hyp_str][k,t+1,:,:]=self.V.dot(self.Theta[hyp_str][k,t,:,:])
                    if hyp['alg'] in ["TDC_inexact","TDC_exact"]:  
                        self.W[t+1,:,:]=self.V.dot(self.W[t,:,:])
                    
                    if False:
                        #Compute model average and errors at t-th iteration
                        self.compute_avg_and_error(hyp['alg'],hyp_str,k,t,None,None,None,None)
           
            if False:
                self.compute_avg_and_error(hyp['alg'],hyp_str,k,num_totaliters,None,None,None,None)
            
    def sim(self,hyps,init_Theta=None,init_W=None,seed_sim=None):
        self.Theta={}
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
        
        np.save(file=result_dir+"/Theta.npy",arr=self.Theta[hyp_str])
        np.save(file=result_dir+"/W.npy",arr=self.W)
        if False:
            for result_type in self.results[hyp_str].keys():
                np.save(file=result_dir+"/"+result_type+".npy",arr=self.results[hyp_str][result_type])
        
            
    def save_results(self,hyps):
        for hyp in hyps:
            self.save_result1(hyp)
    
    def load_result1(self,hyp):
        hyp_str=get_hyp_str(hyp)
        result_dir=hyp['result_dir']
        if "./" not in result_dir:
            result_dir="./"+result_dir
        if not os.path.isdir(result_dir):
            os.makedirs(result_dir)
            
        self.Theta[hyp_str]=np.load(file=result_dir+"/Theta.npy")
        self.W=np.load(file=result_dir+"/W.npy")
        
        if False:
            result_types=['Theta_avg_err','Theta_consensus_err','Theta_err','local_berr','global_Cerr',\
                          'local_Aerr','local_Berr','global_Aerr','global_Berr']
            self.results[hyp_str]={}
            for result_type in result_types:
                try:
                    self.results[hyp_str][result_type]=np.load(result_dir+"/"+result_type+".npy")
                except:
                    pass

    def load_results(self,hyps):
        self.Theta={}
        for hyp in hyps:
            self.load_result1(hyp)
    
    def plot_results(self,hyps,percentile=95,islog10=False,color_transition='green',xlabel='number of samples',ylabel='value function of target state',\
                     plotresult_dir='Figures',title='',filename=None,phi=np.array([1.0,0.0,0.0,0.0,0.0]),fontsize=15,lgdsize=10,legend_loc=1):
        #y-axis in plot: <phi, Theta[t]> along iterations t.
        if xlabel!='number of samples':
            xlabel='number of communication rounds'
        
        phi=np.array(phi).reshape((1,1,-1))
        
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
            # if (result_type in self.results[hyp_str].keys()):
            # pdb.set_trace()
            if hyp['plot_iters'] is None:
                hyp['plot_iters']=np.arange(self.Theta[hyp_str].shape[1])

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
            if hyp['plot_iters'] is None:
                hyp['plot_iters']=np.arange(self.Theta[hyp_str].shape[0])
                    
            result_preplot=(self.Theta[hyp_str][:,hyp['plot_iters'],:,:].mean(axis=2)*phi).sum(axis=2)
                
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
        plt.ylabel(ylabel)
            
        if fontsize is not None:
            plt.rcParams.update({'font.size': fontsize})

        if plotresult_dir is not None:   #Then begin to save figure
            if xlabel=='number of samples':
                xlabel_name='Samples'
            else:
                xlabel_name='Iters'
            if filename is None:
                filename=xlabel
            plt.savefig(plotresult_dir+'/'+filename+'.png',dpi=400)
            plt.close()
                
    def get_ABCbt_batchavg_exactrho(self,s,rho,R,phi_now,phi_next,batchsize):
        #s: np array [s_{tN},...s_{(t+1)N}] with length N+1 where N is batchsize
        #rho: np array [rho_{tN},...,rho_{(t+1)N-1}] with length N
        #R: np array [R_{tN},...,R_{(t+1)N-1}] with N rows, where R_{i} is a row of R_i^{(m)} for agents m=1...M
        #phi_now: np array with N feature column vectors for batches i=tN,...,(t+1)N-1
        #phi_next: np array with N feature column vectors for batches i=tN+1,...,(t+1)N
        
        if False:
            if type(self.features) is str:  #When feature is identity matrix
                phi_now=np.zeros((self.num_features,batchsize))
                phi_now[s[0:batchsize],range(batchsize)]=1.0
                phi_next=np.zeros((self.num_features,batchsize))
                phi_next[s[1:(1+batchsize)],range(batchsize)]=1.0
            else:
                phi_now=self.features[:,s[0:batchsize]]
                phi_next=self.features[:,s[1:(1+batchsize)]]
            
        rho=np.diag(np.array(rho)/batchsize)
        Ct_batchavg=-phi_now.dot(phi_now.T/batchsize)
        Bt_batchavg=((-self.gamma)*phi_next).dot(rho).dot(phi_now.T)
        At_batchavg=-Bt_batchavg.T-phi_now.dot(rho).dot(phi_now.T)
        btm_batchavg=phi_now.dot(rho).dot(R)    #btm_batchavg[:,m] for agent m
        return At_batchavg,Bt_batchavg,Ct_batchavg,btm_batchavg
    
    def get_ABCbt_batchavg_inexactrho(self,s,rho_hat,R,phi_now,phi_next,batchsize):
        #s: np array [s_{tN},...s_{(t+1)N}] with length N+1 where N is batchsize
        #rho_hat: np array [rho_hat_{tN},...,rho_hat_{(t+1)N-1}] with N rows,
        #  where rho_hat_{i} is a row of \widehat{rho}_i^{(m)} for all agents m=1,2,...,M.
        #R: np array [R_{tN},...,R_{(t+1)N-1}] with N rows, where R_{i} is a row of R_i^{(m)} for agents m=1...M
        #phi_now: np array with N feature column vectors for batches i=tN,...,(t+1)N-1
        #phi_next: np array with N feature column vectors for batches i=tN+1,...,(t+1)N
        
        if False:
            if type(self.features) is str:  #When feature is identity matrix
                phi_now=np.zeros((self.num_features,batchsize))
                phi_now[s[0:batchsize],range(batchsize)]=1.0
                phi_next=np.zeros((self.num_features,batchsize))
                phi_next[s[1:(1+batchsize)],range(batchsize)]=1.0
            else:
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









    if False:
        def get_local_rho(self,s_now,a,s_next):  #a is the list of current actions among agents.
            return np.array([self.target_policy[m][s_now,a[m]]/self.behavior_policy[m][s_now,a[m]] for m in range(self.num_agents)])
        
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
            if type(self.features) is str:  #When feature is identity matrix
                self.C=-D.copy()
                self.B=(-self.gamma)*(self.transP_target.T).dot(D)
            else:
                self.C=-self.features.dot(D).dot(self.features.T)
                self.B=((-self.gamma)*self.features).dot(self.transP_target.T).dot(D).dot(self.features.T)
            self.A=self.C-self.B.T
            self.b=(self.reward*self.transP.reshape(self.transP.shape+(1,))).sum(axis=self.num_agents+1)
            for m in range(self.num_agents):
                self.b*=self.target_policy[m].reshape\
                    ((self.num_states,)+(1,)*m+(self.num_actions[m],)+(1,)*(self.num_agents-m))
            self.b=self.b.sum(axis=tuple(range(1,self.num_agents+1)))
            if type(self.features) is str:  #When feature is identity matrix
                self.b=D.dot(self.b)
            else:
                self.b=self.features.dot(D).dot(self.b)
            
        def get_theta_star(self,theta_hat): 
            try:
                return -np.linalg.solve(a=self.A,b=self.b.mean(axis=1))
            except np.linalg.LinAlgError:
                lambda1=np.linalg.pinv(self.B.dot(self.B.T)).dot(self.B.dot(theta_hat)+self.b.mean(axis=1))
                return theta_hat-self.B.T.dot(lambda1)    
    
        def compute_avg_and_error(self,alg,hyp_str,k,t,At,Bt,Ct,bt):  
            #Compute w_star, agent-average parameters and errors of theta, w, A, B, C, b at the t-th iteration of the k-th experiment
            self.Theta_avg[t,:]=self.Theta[t,:,:].mean(axis=0)            
            theta_star=self.get_theta_star(self.Theta_avg[t,:])
            theta_star_normsq=np.square(theta_star).sum()
            self.results[hyp_str]['Theta_avg_err'][k,t]=np.square(self.Theta_avg[t,:]-theta_star).sum()/theta_star_normsq
            self.results[hyp_str]['Theta_consensus_err'][k,t,:]=np.square(self.Theta[t,:,:]-self.Theta_avg[t,:].reshape(1,self.num_features)).sum(axis=1)/theta_star_normsq
            self.results[hyp_str]['Theta_err'][k,t,:]=np.square(self.Theta[t,:,:]-theta_star.reshape(1,self.num_features)).sum(axis=1)/theta_star_normsq
            
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
            
