import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import torch
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from torch_geometric.nn.models import Node2Vec
from sklearn.metrics import roc_auc_score
from sklearn import metrics
import os
import sys
from sklearn.metrics import accuracy_score


PARAMS = {
    'embedding_dim': 128,
    'walk_length': 64,
    'context_size': 64,
    'walks_per_node': 64,
    'num_negative_samples': 1,
}
LR = 1e-2
WEIGHT_DECAY = 5e-4
EPOCHS = 50
DEV = torch.device('cuda')


def train_epoch(n2v, n2v_loader, n2v_optimizer, X, train_y, train_mask, val_y, val_mask, test_mask,test_y):
    print('Two-Step model train epoch')

    X = X.to(DEV)
    train_y = train_y.to(DEV)
    val_y=val_y.to(DEV)
    val_y = val_y.to(DEV)
    Z = None

    n2v.train()
    for i in range(EPOCHS):
        n2v_train_loss = 0

        for (pos_rw, neg_rw) in n2v_loader:
            n2v_optimizer.zero_grad()
            loss = n2v.loss(pos_rw.to(DEV), neg_rw.to(DEV))
            loss.backward()
            n2v_optimizer.step()
            n2v_train_loss += loss.data.item()
        print(f'Epoch {i}. N2V Train_Loss:', n2v_train_loss)
    print('')
    n2v.eval()
    Z = n2v().detach()

    if X is None:
        train_x = Z[train_mask]
        val_x = Z[val_mask]
        test_x = Z[test_mask]
    elif Z is not None:
        train_x = torch.cat([Z[train_mask], X[train_mask]], dim=1)
        val_x = torch.cat([Z[val_mask], X[val_mask]], dim=1)
        test_x = torch.cat([Z[test_mask], X[test_mask]], dim=1)
    else:
        train_x = X[train_mask]
        val_x = X[val_mask]
        test_x = X[test_mask]


    probs, val_probs = mlp_fit_predict(
        train_x.detach().cpu() , train_y.detach().cpu() , test_x.detach().cpu() , val=(val_x.detach().cpu() , val_y.detach().cpu() ), return_val_probs=True)
    val_roc_auc = roc_auc_score(val_y.cpu().numpy(), val_probs)
##############
   
    print(probs.shape)
    print( val_probs.shape)
    
    
    
    auc_total = metrics.roc_auc_score(test_y,probs)
    print(f'\n Test auc: {auc_total*100:.2f}%', )
  
    precision, recall, thresholds= metrics.precision_recall_curve(test_y, probs)
    
    # Use AUC function to calculate the area under the curve of precision recall curve
    auc_precision_recall = metrics.auc(recall, precision)
    print(f'\n Test precision recall curve: {auc_precision_recall*100:.2f}%' )

    
    # Average precision score 
    average_precision = metrics.average_precision_score(test_y, probs)
    print(f'\nTest average_precision: {average_precision*100:.2f}%')
    

    mcc=metrics.matthews_corrcoef(test_y, probs.round())
    print(f'\nTest matthews corrcoef: {mcc*100:.2f}%')

    test_acc=accuracy_score(test_y,probs.round())
    # test_acc = accuracy(test_y, torch.max(probs,1))
    print(f'\nTest accracy: {test_acc*100:.2f}%')
 
##############
    # print('Validation ROC_AUC:', val_roc_auc)
    return auc_total,auc_precision_recall,average_precision,mcc,test_acc


def fit_predict(edge_index, X, train_y, train_mask, val_y, val_mask, test_mask,test_y):
    n2v = Node2Vec(edge_index, **PARAMS).to(DEV)
    n2v_loader = n2v.loader(batch_size=128, shuffle=True, num_workers=0)
    n2v_optimizer = optim.Adam(n2v.parameters(), lr=LR)

    auc_total,auc_precision_recall,precision,mcc,test_acc = train_epoch(
        n2v, n2v_loader, n2v_optimizer, X, train_y, train_mask, val_y, val_mask, test_mask,test_y)

    return auc_total,auc_precision_recall,precision,mcc,test_acc

def variance(data):
    # Number of observations
     print(data)
     n = len(data)
     
    # Mean of the data
     mean = sum(data) / n
     
    # Square deviations
     deviations = [(x - mean) ** 2 for x in data]
     
   # Variance
     variance = sum(deviations) / n
    
     return variance


def main(args):
    roc_aucs = []

    auc_average=0
    auc_precision_recall_average=0
    average_precision_average=0
    mcc_average=0
    test_acc_average=0

    auc_roc_varience=[]
    auc_pr_varience=[]
    precision_varience=[]
    mcc_varience=[]
    acc_varience=[]


    for i in range(max(args.n_runs, 1)):
        print('random run number:',i)
        seed = i
        set_seed(seed)

        (edge_index, _), X,labels, (train_idx, train_y), \
            (val_idx, val_y), (test_idx, test_y), names = tools.get_data(
                args.__dict__, seed=seed)

        if X is None or not X.shape[1]:
            raise ValueError('No features')

        auc_total,auc_precision_recall,precision,mcc,test_acc = fit_predict(edge_index, X, train_y,
                            train_idx, val_y, val_idx, test_idx,test_y)

        auc_average=auc_total+auc_average

        auc_roc_varience.append(auc_total)


        auc_precision_recall_average=auc_precision_recall+auc_precision_recall_average

        auc_pr_varience.append(auc_precision_recall)


        average_precision_average=precision+average_precision_average

        precision_varience.append(precision)


        mcc_average=mcc+mcc_average

        mcc_varience.append(mcc)


        test_acc_average=test_acc+test_acc_average

        acc_varience.append(test_acc)

    auc_average=auc_average/10

    auc_precision_recall_average=auc_precision_recall_average/10

    average_precision_average=average_precision_average/10

    mcc_average=mcc_average/10

    test_acc_average=test_acc_average/10

    print('****************************')
        
    print(f'\n Test auc: {auc_average*100:.2f}%')

    print(f'\n Test precision recall curve: {auc_precision_recall_average*100:.2f}%' )

    print(f'\n precision average: {average_precision_average*100:.2f}%')
    
    print(f'\n Test matthews corrcoef average: {mcc_average*100:.2f}%')

    print(f'\n Test accracy: {test_acc_average*100:.2f}%')

    import math 

    v1=math.sqrt(variance(auc_roc_varience))

    v2=math.sqrt(variance(auc_pr_varience))
        
    v3=math.sqrt(variance(precision_varience))

    v4=math.sqrt(variance(mcc_varience))

    v5=math.sqrt(variance(acc_varience))

    
    print(f'\n auc-roc standard deviation: {v1:.2f}%'  )

    print(f'\n auc-pr standard deviation: {v2:.2f}%'  )

    print(f'\n precision standard deviation: {v3:.2f}%'  )

    print(f'\n matthews corrcoef standard deviation: {v4:.2f}%'  )

    print(f'\n accracy standard deviation: {v5:.2f}%'  )

    # print(f'\n\lr {{N2V-MLP}} & ${auc_average*100:.2f}(\pm{v1:.2f})$& ${auc_precision_recall_average*100:.2f}(\pm{v2:.2f})$ & ${average_precision_average*100:.2f}(\pm{v3:.2f})$ & ${mcc_average*100:.2f}(\pm{v4:.2f})$  & ${test_acc_average*100:.2f}(\pm{v5:.2f})$\\\\hline'  )

    print(f'\n {{N2V-MLP}} & ${auc_average*100:.2f}(\pm{v1:.2f})$ & ${auc_precision_recall_average*100:.2f}(\pm{v2:.2f})$ & ${average_precision_average*100:.2f}(\pm{v3:.2f})$ & ${mcc_average*100:.2f}(\pm{v4:.2f})$ & ${test_acc_average*100:.2f}(\pm{v5:.2f})')


def get_name(args):
    if args.name:
        return args.name

    name = 'N2V'
    if args.no_ppi:
        name += '_NO-PPI'
    if args.expression:
        name += '_EXP'
    if args.sublocs:
        name += '_SUB'
    if args.orthologs:
        name += '_ORT'

    return name


def save_preds(preds, args, seed):
    os.makedirs("./preds", exist_ok=True)
    name = get_name(args) + f'_{args.organism}_{args.ppi}_s{seed}.csv'
    name = name.lower()
    path = os.path.join('preds', name)
    df = pd.DataFrame(preds, columns=['Pred', 'Gene', 'Label'])
    df.to_csv(path)
    print('Saved the predictions to:', path)


if __name__ == '__main__':
    sys.path.append('.')
    from runners import tools
    from runners.run_mlp import mlp_fit_predict
    from utils.utils import *

    args = tools.get_args(parse=True)
    print("ARGS:", args)

    main(args)

    name = get_name(args)

    df_path = 'results/results.csv'
    if os.path.isfile(df_path):
        df = pd.read_csv(df_path)
