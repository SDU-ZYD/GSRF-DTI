
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import numpy as np
import pickle
import json
from tqdm import tqdm
from sklearn.metrics import roc_curve,auc,precision_recall_curve
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
 
dti_feat_file="./dti/dt_feature.txt"
with open("./dti/labels_select_node.pkl",'rb') as f:
        labels_select_node = pickle.load(f)
with open("./dti/node_map.json",'r') as f:
        node_map = json.load(f)
feat_label_select=[]
svm_node =[]
m=0
with open(dti_feat_file) as fp:
    for i, line in tqdm(enumerate(fp)):
        info = line.strip().split()
        info1=info[:2]      
        info2=str(info1).replace("'",'').replace(',','').strip()
        if node_map[info2] in labels_select_node[:,1]:
            feat_label_select.append([float(x) for x in info[2:len(info)]])
            svm_node.append([info2,node_map[info2]])
            m = m+1
        else:
            continue
with open("./dti/feat_label_select.pkl",'wb') as f:
    pickle.dump(feat_label_select,f)
with open("./dti/svm_node.pkl",'wb') as f:
    pickle.dump(svm_node,f)

with open("./dti/features_val4_e50.pkl",'rb') as f:
        features = pickle.load(f)
print("downloaded features data")
with open("./dti/labels_select_node.pkl",'rb') as f:
        labels_select_node = pickle.load(f)
print("downloaded labels_select_node data")

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
with open("/media/rasho/Ningning/graphSAGE-pytorch-master/other_algorithm/SVM_result.txt",'w') as F:
    for i in range(50):
        x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=1,test_size=0.2)
        svm_classification =SVC()
        svm_classification.fit(x_train,y_train)
        y_p=svm_classification.decision_function(x_test)
        y_pred = svm_classification.predict(x_test)
        svm_classification.score(x_test,y_test)

        num = len(y_pred)
        acc = (y_pred == y_test).sum() / num
        precision = np.sum((y_pred+y_test)==2)/y_pred.sum()
        recall = np.sum((y_pred+y_test)==2)/y_test.sum()
        f1 = 2*(precision*recall)/(precision+recall)
        fpr,tpr,threshold = roc_curve(y_test,y_p)
        roc_auc = auc(fpr,tpr)
        P,R,thres = precision_recall_curve(y_test,y_p)
        pr_auc = auc(R,P)
        # G_SVM_fpr = fpr
        # G_SVM_tpr = tpr
        # G_SVM_P = P
        # G_SVM_R = R
        # SVM_fpr = fpr
        # SVM_tpr = tpr
        # SVM_P = P
        # SVM_R = R
        cm = confusion_matrix(y_test,y_pred)
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
