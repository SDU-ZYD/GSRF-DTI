# GSRF-DTI——Drug-target interaction prediction framework based on drug-target pairs network and representation learning on large graph
GSRF-DTI transforms the link prediction problem into a binary classification problem of nodes by building a DTPs-NET to consider the influence of the association between DTPs on DTI prediction. In the process of learning node features, GSRF-DTI integrates the association information between a variety of biological entities by using the graph embedding algorithms, and effectively learns the potential features of the networks, which has achieved good results.
## The environment of GSRF-DTI
```
Linux OS
python  3.9.13
pytorch  1.6.0
pandas  1.4.1
numpy  1.21.5
sklearn  1.0.2
matplotlib  3.5.2
DGL  0.9.1
tqdm  4.64.1
json  0.9.6
```
## Quick start
We provide the data files `output.rar` and `dt_feature.rar` needed to finally implement the DTI prediction.To verify a successful result of GSRF-STI ,  you can run the file (`src/GraphSAGE_.py`) directly.
## Run the GSRF-DTI model for DTI prediction
1.-Download the data though this [link](https://github.com/SDU-ZYD/GSRF-DTI), and the drug/target similarity networks are obtained after data preprocessing. Then the homogenous network of drug/target is obtained by integrating multiple similarity networks.          
2.-Run `DeepWalk.py`        
3.-Run `DTPs_NET.py`       
4.-Run `GraphSAGE_.py`             
5.-Run `Random_Forest_.py`                    
## Code and data
### `src/` directory
- `DeepWalk.py`: learn the features of drugs and targets
- `DTPs_NET.py`: construct the drug-target pairs network (DTPs-NET) and obtain the initial feature of drug-target pairs(DTPs)
- `GraphSAGE_.py`: learn the potential features of the DTPs-NET
- `logistic_.py`: predict drug-target interactions(DTIs) by logistic regession (LR)
- `svm_.py`: predict drug-target interactions(DTIs) by support vector machine (SVM)
- `Random_Forest_.py`: predict drug-target interactions(DTIs) by random forest (RF)
### `data/` directory
- `sevenNets.rar` and `sim_network.rar`: raw data, see for details at https://github.com/luoyunan/DTINet.git
- `ddi.txt`: drug homogenous network
- `ppi.txt`: target homogenous network
- `output.rar`: DTPs-NET
- `dt_feature.rar`: the initial feature of DTPs
### `new_datasets/` directory
- The new dataset constructed in this paper is used to test GSRF-DTI, and details of the files in the folder are provided in `README.txt`.
### `dti/` directory
- `node_map.json`: a complete ID mapping of DTPs
```
(0,0)-->0    
(0,1)-->1           
    ...           
(707,1511)-->1070495
```
 where `(0,0)`  represents the node consisting of `drug0` and `target0`.
- `label_map.json`: a complete ID mapping of DTPs' label
- `labels_node.pkl`: record the node labels after the mapping
- `positive_labels_node_1923.pkl`: record the IDs and labels of all positive samples
- `negetive_labels_node.pkl`: record the IDs and labels of all negetive samples
- `labels_select_node.pkl`: record the IDs and labels of 1,923 positive samples and 1,923 negative samples randomly selected from all negative samples
- `train_val_data_3846_5val0.pkl`: record the first fold sample information when performing five fold cross validation
- `train_val_data_3846_5val1.pkl`: record the second fold sample information when performing five fold cross validation
- `train_val_data_3846_5val2.pkl`: record the third fold sample information when performing five fold cross validation
- `train_val_data_3846_5val3.pkl`: record the fourth fold sample information when performing five fold cross validation
- `train_val_data_3846_5val4.pkl`: record the fifth fold sample information when performing five fold cross validation
- `features_val4_e50.pkl`: record the feature embedding obtained in the fourth fold verification using GraphSAGE algorithm and `epoch=50`
<hr>
<h3>
Note that the original data contains 708 drugs and 1512 target proteins, so the DTPs-NET constructed by GSRF-DTI method contains 1070496 nodes and 2374360128 edges. Therefore, the construction of this large-scale network and the learning of node feature representation require relatively high computer performance!!!                 
</h3>
