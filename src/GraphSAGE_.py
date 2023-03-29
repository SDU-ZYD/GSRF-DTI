############
import torch as torch
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from dgl.nn.pytorch import SAGEConv
from sklearn.metrics import roc_curve,auc,precision_recall_curve

import random
import pickle
import json
import gc
import dgl

from glob import glob
from tqdm import tqdm

class GraphSAGE(nn.Module):
    def __init__(self, 
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout,
                 aggregator):
        
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.layer = nn.ModuleList()
        self.layer.append(SAGEConv(in_feats, n_hidden[0], aggregator))
        for i in range(1, n_layers - 1):
            self.layer.append(SAGEConv(n_hidden[0], n_hidden[1], aggregator))
        self.layer.append(SAGEConv(n_hidden[0], n_hidden[1], aggregator))
        self.dropout = nn.Dropout(dropout)
        self.activation = activation
        self.linear = nn.Linear(hidden_size[1],n_classes)
    def forward(self, blocks, feas):
        h = feas
        for i, (layer, block) in enumerate(zip(self.layer, blocks)):
            h = layer(block, h)
            if i != self.n_layers - 1:
                h = self.activation(h)
                h = self.dropout(h)
        h_feature = h
        h = self.linear(h)          
        return h,h_feature

def evaluate(data, epoches,val_nid, val_mask,val_nid_n_val,val_mask_n_val, args,sample_size,model):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device("cpu")
    in_feats, n_classes, my_net = data
    hidden_size, n_layers, activation, dropout, aggregator, batch_s, num_worker  = args
    # The process of training the model 
    if model is None:
        model = GraphSAGE(in_feats, hidden_size, n_classes, n_layers, activation, dropout, aggregator)
        model_path = glob("./dti/graphsage_*.pt")[-1]
        # model_path = "./dti/graphsage_before5_val4_50+100+100.pt"
        model_static = torch.load(model_path)
        model.load_state_dict(model_static.state_dict())
    labels = my_net.ndata['label']
    model.eval()
    model.to(device)
    sampler = dgl.dataloading.MultiLayerNeighborSampler(sample_size)
    dataloader = dgl.dataloading.NodeDataLoader(
                my_net,
                val_nid,
                # val_nid_n_val,
                sampler,
                batch_size = batch_s,
                shuffle=True,
                drop_last=False,
                num_workers=0
            )

    ret = torch.zeros(my_net.num_nodes(), n_classes)
    
    # features = []
    for input_nodes, output_nodes, blocks in dataloader:
        h = blocks[0].srcdata['features'].to(device)
        block = [block_.int().to(device) for block_ in blocks]
        with torch.no_grad():
            h,h_feature = model(block, h)
        # h_n = h_feature.numpy()
        # for i in range(len(h_n)):
        #     features.append(h_n[i])
        ret[output_nodes] = h.cpu()
    # with open("./dti/features_val4_e50.pkl",'wb') as f:
    #     pickle.dump(features,f)
    label_pred = ret
    new_label_pred= label_pred[val_mask]
    # new_label_pred= label_pred[val_mask_n_val]
    pred_prob = torch.softmax(new_label_pred, dim=1)
    pred = torch.argmax(new_label_pred, dim=1).cpu().numpy()
    target = labels[val_mask].cpu().numpy()
    # target =labels[val_mask_n_val].cpu().numpy()
    
    num = len(new_label_pred)
    acc = (pred == target).sum() / num
    precision = np.sum((pred+target)==2)/pred.sum()
    recall = np.sum((pred+target)==2)/target.sum()
    f1 = 2*(precision*recall)/(precision+recall)
    fpr,tpr,threshold = roc_curve(target,pred_prob[:,1])
    roc_auc = auc(fpr,tpr)
    P,R,thres = precision_recall_curve(target,pred_prob[:,1])
    pr_auc = auc(R,P)
    print('Test ACC: %.4f |precision: %0.4f | recall: %0.4f | f1: %0.4f| auc: %.4f | aupr: %.4f'% (acc, precision,recall,f1, roc_auc,pr_auc))
    F = open("./result/G_parameter_adjust/split5_val4_lstm.txt",'a')
    s = str('Test ACC: %.4f |precision: %0.4f | recall: %0.4f | f1: %0.4f| auc: %.4f | aupr: %.4f'% (acc, precision,recall,f1, roc_auc,pr_auc))
    TPTN=(pred == target).sum()
    TP=np.sum((pred+target)==2)
    pred1number =pred.sum()
    number=str('TP+TN= %.0f |TP= %.0f |pred.sum= %.0f'%(TPTN,TP,pred1number))
    F.write("epoch="+str(epoches)+'|'+"lr="+str(learning_rate)+'|'+number+'|'+s+'\n')
    F.close()

    return acc,precision,recall,f1,fpr,tpr,roc_auc,P,R,pr_auc

def run(data, train_val_data, args, sample_size, learning_rate,epoches, device_num):
    print("Start training")
    train_mask, val_mask, val_mask_n_val,train_nid, val_nid,val_nid_n_val = train_val_data
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device("cpu")
    in_feats, n_classes, my_net = data
    hidden_size, n_layers, activation, dropout, aggregator, batch_s, num_worker  = args
    # The process of training the model  
    model = GraphSAGE(in_feats, hidden_size, n_classes, n_layers, activation, dropout, aggregator)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fun = nn.CrossEntropyLoss()
    loss_fun.to(device)
    sampler = dgl.dataloading.MultiLayerNeighborSampler(sample_size)
    dataloader = dgl.dataloading.NodeDataLoader(
        my_net,
        train_nid,
        sampler,
        batch_size = batch_s,
        shuffle=True,
        drop_last=False,
        num_workers=num_worker
    )
    for epoch in range(epoches):
        model.train()
        print(f"***************************{epoch}*********************************")
        for batch, (input_nodes, output_nodes, block) in tqdm(enumerate(dataloader)):
            batch_feature = block[0].srcdata['features']
            batch_label = block[-1].dstdata['label']
            batch_feature, batch_label = batch_feature.to(device),batch_label.to(device)
            block = [block_.int().to(device) for block_ in block]
            model_pred,mode_feature = model(block, batch_feature)
            loss = loss_fun(model_pred, batch_label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if batch % 2 == 0:
                print('Batch %d | Loss: %.7f' % (batch, loss.item()))

        torch.save(model,"./dti/graphsage_before5_val1_e{}_lr0.01.pt".format(epoch))
        torch.save(my_net.ndata['features'],"./dti/graphsage_embedding.pt")
    del dataloader
    gc.collect()       

    # Model training completed
    acc,precision,recall,f1,fpr,tpr,roc_auc,P,R,pr_auc = evaluate(data,epoch, val_nid, val_mask, val_nid_n_val, val_mask_n_val,args,sample_size,model=model)
    print('Test ACC: %.4f |precision: %0.4f | recall: %0.4f | f1:%0.4f| auc:%.4f | aupr:%.4f'% (acc, precision,recall,f1, roc_auc,pr_auc))
    with open("./dti/metrci.pkl",'wb') as f:
        pickle.dump([fpr,tpr,P,R],f)
    plt.subplot(121)
    plt.plot(fpr, tpr, 'k--', label='ROC (area = {0:.2f})', lw=2)
    
    plt.title('ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

    plt.subplot(122)
    plt.plot(R, P, 'k--', label='P-R (area = {0:.2f})', lw=2)
    plt.title('Precision/Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.xlim([-0.05, 1.05])
    plt.show()

    return model

def train_val_split(node_fea,labels_select_node,node_fea_all):
    print("split data") 
    with open("./dti/negetive_labels_node_1923.pkl",'rb') as f:
        negetive_labels_node_1923 = pickle.load(f)
    train_node_ids = []
    val_node_ids = []
    val_node_ids_n_val =[]

    for label,group in node_fea.groupby('label_number'):
        node_id_numbers = group.sort_values('node_id_number')['node_id_number']
        numbers = len(node_id_numbers) 
        ratio = int(numbers*1/5)
        data_split  = []
        for i in list(range(0,numbers,ratio)):
            data_split.append([i,i+ratio])
        val_ratio_number = data_split[1]
        data_split.remove(val_ratio_number)
        for s,e in data_split:
            train_node_ids.extend(node_id_numbers[s:e])
        s,e = val_ratio_number
        val_node_ids.extend(node_id_numbers[s:e])
    for i in range(len(labels_select_node)):
        val_node_ids_n_val.append(labels_select_node[i][1])

    val_nid_dict_n_val = {nid:i for i,nid in enumerate(val_node_ids_n_val)} 
    train_nid_dict = {nid:i for i,nid in enumerate(train_node_ids)}
    val_nid_dict = {nid:i for i,nid in enumerate(val_node_ids)}
    
    train_mask = node_fea_all['node_id_number'].apply(lambda x : train_nid_dict.get(x) is not None)
    val_mask = node_fea_all['node_id_number'].apply(lambda x : val_nid_dict.get(x) is not None)
    val_mask_n_val = node_fea_all['node_id_number'].apply(lambda x : val_nid_dict_n_val.get(x) is not None) 

    return train_mask, val_mask, val_mask_n_val,train_node_ids, val_node_ids,val_node_ids_n_val

def read_dti_data(dti_feat_file,dti_adj_file):
    feat_data = []
    labels = []  # label sequence of node
    node_map = {}  # map node to Node_ID
    label_map = {}  # map label to Label_ID
    ind = 0
    negetive_labels =[]
    positive_labels =[]
    
    with open(dti_feat_file) as fp:
        for i, line in tqdm(enumerate(fp)):
            info = line.strip().split()
            feat_data.append([float(x) for x in info[2:-1]])  
            info1=info[:2]
            info2=str(info1).replace("'",'').replace(',','').strip()
            if info2 not in node_map:
                node_map[info2] = ind
                ind += 1
            if not info[-1] in label_map:
                label_map[info[-1]] = len(label_map) 
            labels.append([label_map[info[-1]],node_map[info2]])
            if info[-1] == '1':
                positive_labels.append([label_map[info[-1]],node_map[info2]])
            else:
                negetive_labels.append([label_map[info[-1]],node_map[info2]])

    feat_data = np.asarray(feat_data)
    labels_node = np.asarray(labels, dtype=np.int64)
    negetive_labels_node = np.asarray(negetive_labels, dtype=np.int64)
    a = random.sample(negetive_labels,1923)
    positive_labels_node_1923 = np.asarray(positive_labels, dtype=np.int64)

    labels_select = []
    for p_l in positive_labels:
        labels_select.append(p_l)
    for n_l in a:
        labels_select.append(n_l)
    labels_select_node = np.asarray(labels_select, dtype=np.int64)  
    
    edge3 = []
    edge4 = []
   
    with open(dti_adj_file) as fp:
        for i, line in tqdm(enumerate(fp)):
            info = line.strip().split()
            assert len(info) == 4
            info1 = info[:2]
            info11 = info[2:4]
            info2 = str(info1).replace("'", '').replace(',', '').strip()
            info3 = str(info11).replace("'", '').replace(',', '').strip()
            paper1 = node_map[info2]
            paper2 = node_map[info3]
            edge3.append(paper1)
            edge4.append(paper2)

    edge3 = np.array(edge3)
    edge4 = np.array(edge4)
    pickle.dump(edge3, open("./dti/edge3.pkl",'wb'), protocol=4)
    pickle.dump(edge4, open("./dti/edge4.pkl",'wb'), protocol=4)

    with open("./dti/label_map.json",'w') as f:
        json.dump(label_map,f)
    with open("./dti/node_map.json",'w') as f:
        json.dump(node_map,f)
    pickle.dump(labels_node, open("./dti/labels_node.pkl",'wb'))
    pickle.dump(labels_select_node, open("./dti/labels_select_node.pkl",'wb'))
    pickle.dump(negetive_labels_node, open("./dti/negetive_labels_node.pkl",'wb'))
    pickle.dump(positive_labels_node_1923, open("./dti/positive_labels_node_1923.pkl",'wb'))
    pickle.dump(feat_data, open("./dti/feat_data.pkl",'wb'))

    return feat_data,labels_node,label_map,node_map,edge3,edge4

def loaddata(is_training):
    src= pickle.load(open("./dti/edge3.pkl",'rb'))
    print("downloaded edge3 data")
    dst= pickle.load(open("./dti/edge4.pkl",'rb'))
    print("downloaded edge4 data")
    with open("./dti/node_map.json",'r') as f:
        node_map = json.load(f)
    with open("./dti/label_map.json",'r') as f:
        label_map = json.load(f)
    with open("./dti/labels_node.pkl",'rb') as f:
        labels_node = pickle.load(f)
    print("downloaded labels_node data")
    with open("./dti/labels_select_node.pkl",'rb') as f:
        labels_select_node = pickle.load(f)
    print("downloaded labels_select_node data")
    with open("./dti/feat_data.pkl",'rb') as f:
        feat_data = pickle.load(f)
    print("downloaded feat_data data")
    
    u = np.concatenate([src, dst])
    # v = np.concatenate([dst, src])
    del src
    del dst
    gc.collect()
    v = np.concatenate([u[2374360128:len(u)], u[0:2374360128]])
    my_net = dgl.graph((u, v))
    del u
    del v
    gc.collect()

    node_number = len(node_map)
    in_feats = feat_data.shape[1]
    if is_training:
        tensor_fea = torch.tensor(feat_data, dtype=torch.float32)
        fea_np = nn.Embedding(node_number, in_feats)
        fea_np.weight = nn.Parameter(tensor_fea)
        del tensor_fea
        gc.collect()
        my_net.ndata['features'] = fea_np.weight
    else:
        graphsage_embedding = torch.load("./dti/graphsage_embedding.pt")
        my_net.ndata['features'] = graphsage_embedding
        del graphsage_embedding
        gc.collect()
    my_net.ndata['label'] = torch.tensor(labels_node[:,0])
    n_classes = len(label_map)
    data = in_feats, n_classes, my_net

    node_fea_all = pd.DataFrame(labels_node,columns=['label_number','node_id_number'])
    node_fea = pd.DataFrame(labels_select_node,columns=['label_number','node_id_number'])
    train_val_data = train_val_split(node_fea,labels_select_node,node_fea_all)
    return data, train_val_data
#Read the feature embedding after using Deepwalk
dti_feat_file="./data/dt_feature.txt"
dti_adj_file="./data/output.txt"
feat_data,labels_node,label_map,node_map,edge3,edge4 = read_dti_data(dti_feat_file,dti_adj_file)
pickle.dump(feat_data, open("./dti/feat_data.pkl",'wb'))

# If you only want to use the good model of training for prediction, is_training is changed to 'False'
data,train_val_data = loaddata(is_training=True)
with open("./dti/train_val_data_3846_5val1.pkl",'wb') as f:
    pickle.dump(train_val_data,f)
print("Data loading completed")
hidden_size = [64,32]
n_layers = 2
sample_size = [10, 50]
activation = F.relu
dropout = 0.2
aggregator = 'mean'
batch_s = 300
num_worker = 0
learning_rate = 0.001
epoches = 50
device = 1

args =  hidden_size, n_layers, activation, dropout, aggregator, batch_s, num_worker  
# Train the model
trained_model = run(data, train_val_data, args, sample_size, learning_rate,epoches,device)
# Use the trained model alone to predict
train_mask, val_mask, val_mask_n_val,train_nid, val_nid,val_nid_n_val = train_val_data
acc,precision,recall,f1,fpr,tpr,roc_auc,P,R,pr_auc = evaluate(data, epoches,val_nid, val_mask,val_nid_n_val,val_mask_n_val, args,sample_size,trained_model)
print('Test ACC: %.4f |precision: %0.4f | recall: %0.4f | f1:%0.4f| auc:%.4f' % (acc, precision,recall,f1, roc_auc))


