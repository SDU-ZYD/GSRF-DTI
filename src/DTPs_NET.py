
# ************************
# If you just want to reproduce GSRF-DTI, you don't need to run this code. Because the 'dt_feature.txt' and 'output.txt' data files required by GraphSAGE have been uploaded to the data file.
# This script is used to build the initial features of the nodes(DTPs) in DTPS-NET.
# You need to run DeepWalk.py before you can run dtp_feat_content.py, to obtain the id of the drugs and the targets as well as the feature embedding, respectively.
# Then you can run 'dtp_feat_content.py' directly, and the result will be the drug feature file 'd_feat.csv' and the target feature file 'p_feat.csv', as well as the id file 'dt_feature_inds.txt' and initial feature file 'dt_feature.txt' of the drug target pairs.
# In addition, running the last line of code and you will get DTPs-NET's edge set file 'output.txt'.
# ************************

import numpy as np
import pandas as pd

#Ordered characteristics of drugs or targets
def load_feat(file_embedding, file_node, file_save):
    embedding = np.loadtxt(file_embedding)
    index=np.loadtxt(file_node)
    indexs=index.astype(int)
    data = pd.DataFrame(embedding,index=indexs)
    feat=data.sort_index()
    d_or_p_feat=feat.values
    feat.to_csv(file_save,sep='\t')   
    return d_or_p_feat

def get_dtp_features():
    dd_feat = load_feat(d_file_embedding, d_file_node, d_file_save)
    pp_feat = load_feat(p_file_embedding, p_file_node, p_file_save)
    dt_feature=[]
    dt_feature_inds=open('./data/dt_feature_inds.txt','w')
    for i,line1 in enumerate(dd_feat.tolist()):
        for j,line2 in enumerate(pp_feat.tolist()):
            dt_feat = np.concatenate((line1,line2),axis=0)
            dt_feature.append(dt_feat.tolist())
            # dt_feature.append(str(dt_feat).replace('\n','').replace("[","").replace("]","").replace("'","").replace("]","'"))
            # dt_feature.writelines(str(dt_feature).replace('\n','').replace(',','').replace('\t','')+'\n')
            dt_feature_inds.write('({}, {})\n'.format(i, j))

    dtp_feat=pd.DataFrame(dt_feature)
    dt_feats = dtp_feat.drop(['Unnamed: 0'],axis=1)
    dt_feature_inds = pd.read_table('./data/dt_feature_inds.txt')
    dtp_feats = pd.concat([dt_feature_inds,dt_feats],axis=1)
    f = open('./data/dt_feature.txt', "w")
    for indexs in dtp_feats.index:
        line = dtp_feats.loc[indexs].values[1:]
        dtp_feats_str = str(line).replace('[', '').replace(']', '').replace("'", '').replace('\n', '')
        f.write('{}\n'.format(dtp_feats_str))    
#Construct the edge set of DTPs-NET
def get_output():
    output = open('output.txt', 'w')
    rows = 708
    cols = 1512
    for i in range(rows*cols):
        for j in range(rows*cols):
            row1 = i//cols
            col1 = i%cols
            row2 = j//cols
            col2 = j%cols
            if row1 == row2 or col1 == col2:
                if row1 == row2 and col1 == col2:
                    continue
                output.write('({}, {}), ({}, {})\n'.format(row1, col1, row2, col2))
    return
d_file_embedding = './data/ddii_embedding.txt'
d_file_node = './data/dnode.txt'
d_file_save = './data/d_feat.csv'
p_file_embedding = './data/ddii_embedding.txt'
p_file_node = './data/dnode.txt'
p_file_save = './data/p_feat.csv'
get_dtp_features()
get_output()



