#*************************

import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve,auc,precision_recall_curve

#Loading the initial features before GraphSAGE were not used.
with open("./dti/feat_label_select.pkl",'rb') as f:
        feat_label_select = pickle.load(f)
print("downloaded feat_label_select data")
#Loading the feature embedding after using GraphSAGE.
with open("./dti/features_val4_e50.pkl",'rb') as f:
        features = pickle.load(f)
print("downloaded features data")
with open("./dti/labels_select_node.pkl",'rb') as f:
        labels_select_node = pickle.load(f)
print("downloaded labels_select_node data")
with open("./dti/svm_node.pkl",'rb') as f:
        node_ids = pickle.load(f)
x=[]
y=[]
# for i in range(len(feat_label_select)):
#     x.append(feat_label_select[i][0:200])
#     y.append(feat_label_select[i][-1])
for i in range(len(features)):
    x.append(features[i])
    y.append(labels_select_node[i][0])
x = np.array(x)
y = np.array(y)

sum_acc = 0
sum_roc_auc = 0
sum_pr_auc = 0
sum_precision = 0
sum_recalll = 0
sum_f1 = 0

with open("./result/other_algorithm/RandomForest_result.txt",'w') as F:
    for i in range(50):
        x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=i,test_size=0.2)

        rfc= RandomForestClassifier()
        rfc.fit(x_train,y_train)
        y_p=rfc.predict_proba(x_test)[:,1]
        y_pred = rfc.predict(x_test)
        rfc.score(x_test,y_test)

        num = len(y_pred)
        acc = (y_pred == y_test).sum() / num
        precision = np.sum((y_pred+y_test)==2)/y_pred.sum()
        recall = np.sum((y_pred+y_test)==2)/y_test.sum()
        f1 = 2*(precision*recall)/(precision+recall)
        fpr,tpr,threshold = roc_curve(y_test,y_p)
        roc_auc = auc(fpr,tpr)
        P,R,thres = precision_recall_curve(y_test,y_p)
        pr_auc = auc(R,P)
        
        sum_acc = sum_acc+acc
        sum_precision = sum_precision + precision
        sum_recalll = sum_recalll + recall
        sum_f1 = sum_f1 + f1
        sum_roc_auc = sum_roc_auc + roc_auc
        sum_pr_auc = sum_pr_auc + pr_auc
        print('Test ACC: %.4f |precision: %0.4f | recall: %0.4f | f1: %0.4f| auc: %.4f | aupr: %.4f'% (acc, precision,recall,f1, roc_auc,pr_auc))
        s = str('Test ACC: %.4f |precision: %0.4f | recall: %0.4f | f1: %0.4f| auc: %.4f | aupr: %.4f'% (acc, precision,recall,f1, roc_auc,pr_auc))
        TPTN=(y_pred == y_test).sum()
        TP=np.sum((y_pred+y_test)==2)
        pred1number =y_pred.sum()
        number=str('TP+TN=%.0f |TP=%.0f |pred.sum=%.0f'%(TPTN,TP,pred1number))
        F.write("random_state="+str(i)+'|'+number+'|'+s+'\n')

    mean_acc = sum_acc/(i+1)
    mean_precision = sum_precision/(i+1)
    mean_recall = sum_recalll/(i+1)
    mean_f1 = sum_f1/(i+1)
    mean_roc_auc = sum_roc_auc/(i+1)
    mean_pr_auc = sum_pr_auc/(i+1)
    ss = str('Test mean_ACC: %.4f |mean_precision: %0.4f | mean_recall: %0.4f | mean_f1: %0.4f| mean_auc: %.4f | mean_aupr: %.4f'% (mean_acc, mean_precision,mean_recall,mean_f1, mean_roc_auc,mean_pr_auc))
    F.write(ss+'\n')

