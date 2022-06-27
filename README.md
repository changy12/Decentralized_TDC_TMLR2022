# Decentralized_TDC_TMLR2022
This Python code is for the TMLR 2022 paper:
Ziyi Chen, Yi Zhou, Rong-Rong Chen. Multi-Agent Off-Policy TDC with Near-Optimal Sample and Communication Complexities. TMLR 2022.


(1) For the simulation experiment in Section 5.1:

Run DTD_main.py first, which defines the main function that will use the class DTD in DTD.py and the functions in DTD_utils.py. 

Then run the command "dtd=main(isloop=False)" which saves the experimental results (figures and numerical arrays) for the fully connected network in the folder--
"ToyResults/alpha1_exact_0.2__beta1_exact0.002__alpha_inexact5__beta_inexact0.05__20000samples"

Inside this folder, the result figures for comparing different batchsizes N and communication rounds L are plotted into the folders "Figures_varyN_Full_Link" and 
"Figures_varyL_Full_Link" respectively. The data files of Markovian samples in .npy format is saved in the folder "SampleData_Full_Link". The error arrays are 
saved into "ErrorData_Full_Link" in .npy format. If you do not want to save these data files, set "issave=False" (default: True) when running "dtd=main(isloop=False)". 

Similarly, run "dtd=main(isloop=True)" and you will obtain the results for the ring network in the same folder--
"ToyResults/alpha1_exact_0.2__beta1_exact0.002__alpha_inexact5__beta_inexact0.05__20000samples"

Inside this folder, the folders' names are similar to those above for the fully connected network except that "Full_Link" will be replaced with "Loop_Link". 


(2) For the experiment in Section 5.2 for Two-Agent Cliff Navigation Problem:

Run DTD_main_diff.py first, then run the command "dtd=main_cliff()", which saves the results in the folder-- 
"CliffResults/alpha1_exact_0.2__beta1_exact0.002__alpha_inexact5__beta_inexact0.05__20000samples".


(3) For the experiment in Section 5.3 for Path Finding Problem:

Run DTD_main_diff.py first, then run the command "dtd=main_cliff()", which saves the results in the folder-- 
"PathResults/alpha1_exact_0.06__beta1_exact0.002__alpha_inexact5__beta_inexact0.05__20000samples".

In the folder "Figure_varyN" inside the above folder, "goal_number of samples.png" and "goal_number of communication rounds.png" are used by Figure 7 of this paper. 

