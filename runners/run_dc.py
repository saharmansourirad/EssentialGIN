
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import Linear, Dropout
from torch_geometric.nn import GraphConv,SAGEConv,GCNConv, GATv2Conv,GINConv,GCN2Conv,GINEConv,SAGPooling
from torch_geometric.nn import global_mean_pool, global_add_pool,SAGPooling
from torch_geometric.nn.models import Node2Vec
from torch.nn import LayerNorm, Linear, Sequential, BatchNorm1d, ReLU, Dropout
from torch_geometric.nn.inits import glorot
from torch_geometric.data import Data
from torch_geometric.nn.conv import gcn2_conv
from torch_geometric.utils import to_networkx
from torch_geometric.loader import DataLoader,NeighborLoader
from pprint import pprint
import networkx as nx
sys.path.append('.')
from sklearn.metrics import accuracy_score
#from torchviz import make_dot

import pandas as pd
import numpy as np
from tqdm import tqdm
import optuna
from sklearn import metrics

from run_mlp import mlp_fit_predict
from utils.utils import *
from runners import tools

import networkx as nx
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import torch_geometric.transforms as T

# from layer import GCNIIdenseConv
import math



def accuracy(pred_y, y):
    """Calculate accuracy."""
    return ((pred_y == y).sum() / len(y)).item()

def evalAUC(model, X, A, y, mask, logits=None):
    assert(model is not None or logits is not None)
    if model is not None:
        model.eval()
        with torch.no_grad():
            logits = model(X, A)
            logits = logits[mask]
    probs = torch.sigmoid(logits)
    probs = probs.cpu().numpy()
    y = y.cpu().numpy()
    auc = metrics.roc_auc_score(y, probs)
    
    return auc


def get_name(args):
    if args.name:
        return args.name
    name = 'GAT'
    if args.no_ppi:
        name += '_NO-PPI'
    if args.expression:
        name += '_EXP'
    if args.sublocs:
        name += '_SUB'
    if args.orthologs:
        name += '_ORT'
    return name

def save_preds(preds, name, args, seed):
    name = name.lower() + f'_{args.organism}_{args.ppi}_s{seed}.csv'
    path = os.path.join('outputs/preds', name)

    df = pd.DataFrame(preds, columns=['Gene', 'Pred'])
    df.to_csv(path)
    print('Saved the predictions to:', path)

import random  

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def degree_cebtrality(X,edge_index,labels,test_y,test_idx,genes):

    total = labels.groupby(['Label'])['Label'].count()

    essential_genes=labels.loc[labels['Label']==1]

    count_essential_genes=len(essential_genes) 

    none_essential_genes=labels.loc[labels['Label']==0]

    count_none_essential_geness=len(none_essential_genes) 

    N = len(X)
    mapping = dict(zip(genes, range(N)))

    # print('mapping',mapping)

    essential_genes_indexes = [mapping[t] for t in essential_genes.index]

    genes_index= labels.rename(index=mapping)

  
    data = Data(x=X, edge_index=edge_index,y=labels)
   
    graph = to_networkx(data,to_undirected=False,remove_self_loops=False )

    results=nx.degree_centrality(graph)

    degCent_sorted=dict(sorted(results.items(), key=lambda item: item[1],reverse=True))

    results_df=pd.DataFrame.from_dict(degCent_sorted, orient='index')

    res_df = results_df.reindex(genes_index.index) 

    y=genes_index.iloc[:,0]
    DC_y=res_df.iloc[:,0]

    auc_total = metrics.roc_auc_score(test_y,DC_y[test_idx])

    precision, recall, thresholds= metrics.precision_recall_curve(test_y,DC_y[test_idx])
    
    # Use AUC function to calculate the area under the curve of precision recall curve
    auc_precision_recall = metrics.auc(recall, precision)
    # print(f'\n Test precision recall curve: {auc_precision_recall*100:.2f}%' )

    # Average precision score 
    average_precision = metrics.average_precision_score(test_y,DC_y[test_idx])
    # print(f'\nTest average_precision: {average_precision*100:.2f}%')

    results_df.iloc[0:count_essential_genes, :]=1
    ess=results_df.iloc[0:count_essential_genes, :]

    # print(ess)
    # print(ess.shape)
    
    results_df.iloc[count_essential_genes:results_df.shape[0], :]=0
    ness=results_df.iloc[count_essential_genes:results_df.shape[0], :]

    results_df = results_df.reindex(genes_index.index) 
    
    y=genes_index.iloc[:,0]
    DC_y=results_df.iloc[:,0]

    mcc=metrics.matthews_corrcoef(DC_y[test_idx].astype(int),test_y)
    # print(f'\nTest matthews corrcoef: {mcc*100:.2f}%')

    test_acc = accuracy_score(DC_y[test_idx].astype(int),test_y)
    # print(f'\n acc: {test_acc*100:.2f}%')


    return auc_total,auc_precision_recall,average_precision,mcc,test_acc

def main(args, name='', seed=200, save=True):

    set_seed(seed)

    snapshot_name = f'{args.organism}_{args.ppi}'
    for p in ['expression', 'orthologs', 'sublocs']:
        if args.__dict__[p]:
            snapshot_name += f'_{p}'

    weightsdir = './outputs/weights/gat'
    outdir = './output/results/{args.organism}/gat'
    savepath = os.path.join(weightsdir, snapshot_name)

    # Getting the data ----------------------------------
    
    (edge_index, edge_weights), X,labels, (train_idx, train_y), \
        (val_idx, val_y), (test_idx, test_y), genes = tools.get_data(
            args.__dict__, seed=seed, weights=True)

    df = pd.DataFrame(columns=['auc_roc','auc_pr','auc_precision_recall','average_precision','mcc','test_acc'],
                  index=['human','melanogaster','yeast','coli'])
    columns=['auc_roc','auc_pr','auc_precision_recall','average_precision','mcc','test_acc']

    df2 = pd.DataFrame([ [1,2,3,4,5,6], [7, 8,9,10,11,12]],columns=columns,
                  index=['human','melanogaster'] )


    auc_total,auc_precision_recall,average_precision,mcc,test_acc=degree_cebtrality(X,edge_index,labels,test_y,test_idx,genes)
  # ---------------------------------------------------
   
    return  auc_total,auc_precision_recall,average_precision,mcc,test_acc

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

if __name__ == '__main__':
    args = tools.get_args()

    name = get_name(args)
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

    if args.n_runs:
        args.train = True
        args.test = True

        scores = []
        for i in range(args.n_runs):
           
            # print('*****************')
            # print('random run number:',i)

            auc_total,auc_precision_recall,precision,mcc,test_acc= main(args, name=name, seed=i)

            
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

        auc_average=auc_average/args.n_runs

        auc_precision_recall_average=auc_precision_recall_average/args.n_runs

        average_precision_average=average_precision_average/args.n_runs

        mcc_average=mcc_average/args.n_runs

        test_acc_average=test_acc_average/args.n_runs

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

    
        print(f'\n auc-roc varience: {v1:.2f}%'  )

        print(f'\n auc-pr varience: {v2:.2f}%'  )

        print(f'\n precision varience: {v3:.2f}%'  )

        print(f'\n matthews corrcoef varience: {v4:.2f}%'  )

        print(f'\n accracy varience: {v5:.2f}%'  )

        print(f'\n 	\lr{{GIN-MLP}} & ${auc_average*100:.2f}(\pm{v1:.2f})$& ${auc_precision_recall_average*100:.2f}(\pm{v2:.2f})$ & ${average_precision_average*100:.2f}(\pm{v3:.2f})$ & ${mcc_average*100:.2f}(\pm{v4:.2f})$  & ${test_acc_average*100:.2f}(\pm{v5:.2f})$\\\\hline'  )


    else:
        print('Training a single run with seed', args.seed)
        main(args, name=name, seed=args.seed)
