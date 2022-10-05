# MDP: Privacy-Preserving GNN Based on Matrix Decomposition and Differential Privacy
## GCN 
Step.1 Run GCN\datasets\pubmed\EigenvalueSelection_GCN.m with Matlab to complete the process of eigenvalue selection mentioned in the paper<br>

Step.2 Save the variables out_1 and out_2 generated after Step.1 as L1.mat and L2.mat respectively (in the same directory)<br>

Step.3 run GCN\train.py<br>
(**NC** in train.py corresponds to the number of calculators)<br>
(**epsilon** in utils.py corresponds to privacy budget)<br>

## GAT
Step.1 Run GAT\datasets\pubmed\EigenvalueSelection_GAT.m with Matlab to complete the process of eigenvalue selection mentioned in the paper<br>

Step.2 Save the variables m1 and m2 generated after Step.1 as m1.mat and m2.mat respectively (in the same directory)<br>

Step.3 run GAT\train.py<br>
(**NC** in train.py corresponds to the number of calculators)<br>
(**epsilon** in utils.py corresponds to privacy budget)<br>