
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
from sklearn.metrics import accuracy_score
from torch_geometric.nn.inits import glorot
from torch_geometric.data import Data
from torch_geometric.nn.conv import gcn2_conv
from torch_geometric.utils import to_networkx
from torch_geometric.loader import DataLoader,NeighborLoader
from pprint import pprint
import networkx as nx
sys.path.append('.')


import pandas as pd
import numpy as np
from tqdm import tqdm
import optuna
from sklearn import metrics


from models.gat import params as gat_params
from run_mlp import mlp_fit_predict
from utils.utils import *
from runners import tools

import networkx as nx
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import torch_geometric.transforms as T

from layer import GCNIIdenseConv
import math

       
# DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from torch_geometric.nn.conv import MessagePassing
from torch.nn import Parameter
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax


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

    
def test(model, X, edge_index,test_idx, test_y):
    
    """Evaluate the model on test set and print the accuracy score."""
    model.eval()
    acc=1
    test_x = X[test_idx]
   
    h,out = model(X,edge_index)

    with torch.no_grad():
            h,out = model(X,edge_index)
            
    probs = torch.sigmoid(out.argmax(dim=1))

    test_y = test_y.cpu().numpy()
                
    f1_score=metrics.f1_score(test_y, out[test_idx].argmax(dim=1))
    print(f'test f1 micro score: {f1_score*100:.2f}%')
    auc = metrics.roc_auc_score(test_y, out[test_idx].argmax(dim=1))
        
    return  probs, auc,acc


params = {
    'embedding_dim': 128,
    'walk_length': 64,
    'context_size': 64,
    'walks_per_node': 64,
    'num_negative_samples': 1,
}

class GIN(torch.nn.Module):
    """GIN"""
    def __init__(self,edge_index,dim_in,dim_h,dim_out, shared_weights=True):
        super(GIN, self).__init__()

        hl = [32, 32, 32];

        self.conv1 = GINConv(
            Sequential(Linear(dim_in, hl[0]),
                        BatchNorm1d(hl[0]),ReLU(),                
                       Linear(hl[0], hl[0]), ReLU()))

        self.conv2 = GINConv(
            Sequential(Linear(hl[0], hl[1]),
                       BatchNorm1d(hl[1]),ReLU(),               
                       Linear(hl[1], hl[1]), ReLU()))

        self.conv3 = GINConv(
            Sequential(Linear(hl[1], hl[1]),
                       BatchNorm1d(hl[1]),ReLU(),               
                       Linear(hl[1], hl[1]), ReLU()))

        self.conv4 = GINConv(
            Sequential(Linear(hl[1], hl[2]),
                       BatchNorm1d(hl[2]), ReLU(), 
                       Linear(hl[2], hl[2]), ReLU()))

        self.lin1 = Linear(hl[0]*4,hl[0]*4)
        self.lin2 = Linear(hl[0]*4, dim_out)
        
        self.weight = Parameter(torch.Tensor(
            dim_in, dim_in))
       
        
        self.optimizer = torch.optim.Adam(self.parameters(),
                                      lr=0.001,
                                      weight_decay=5e-4)
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)


    def forward(self, x, edge_index,size=None):
        
        x = torch.matmul(x, self.weight)
        DropOut1=0.1
        DropOut2=0.3


        h1 = self.conv1(x, edge_index)   
        h1 = F.dropout(h1, p=DropOut1, training=self.training)     

        h2 = self.conv2(h1, edge_index)
        h2 = F.dropout(h2, p=DropOut1, training=self.training)

        h3=self.conv3(h2, edge_index)
        h3= F.dropout(h3, p=DropOut1, training=self.training)

        h4 = self.conv4(h3, edge_index)
        h4= F.dropout(h4, p=DropOut1, training=self.training)
        
        h=torch.cat((h1, h2,h3,h4), dim=1)

        h = F.dropout(h, p=DropOut2, training=self.training)#0.3

        h = self.lin1(h)
        
        h = F.relu(h)
        
        h = F.dropout(h, p=DropOut2, training=self.training)#0.4
        
        h = self.lin2(h)
        
        
        return h,F.log_softmax(h, dim=1)



def train_model(model,X, edge_index,train_idx,train_y, val_idx,val_y,test_idx,test_y):

    """Train a GNN model and return the trained model."""
    criterion = torch.nn.CrossEntropyLoss()
  
    optimizer = model.optimizer

    model.to('cpu')
    DEVICE='cpu'
    X = X.to(DEVICE)
    edge_index = edge_index.to(DEVICE)
    train_y = train_y.to(DEVICE)
    val_y = val_y.to(DEVICE)

    mean_auc=0
    mean_val_auc=0
    loss=1

    epochs=5
    epoch=0
    model.train()

    base_weights=torch.tensor([1,2])#1,0human
    
    ones = torch.ones(train_y.shape[0], 2)  
  
    class_weights=torch.mul(ones, base_weights)

    val_losses = []
    train_losses = []
  
    while  epoch<500: #loss>0.177

        # Training
        epoch=epoch+1
        optimizer.zero_grad()

        G,out=model(X, edge_index)  
        
        train_y=train_y.type(torch.LongTensor)

        loss = criterion(out[train_idx],train_y)

        train_losses.append(loss.item())

        val_y=val_y.type(torch.LongTensor)

        val_loss = criterion(out[val_idx],val_y)

        val_losses.append(val_loss.item())

        loss.backward()
       
        optimizer.step()

        out = out.detach()
        G=G.detach()
       
        auc = metrics.roc_auc_score(train_y,out[train_idx].argmax(dim=1))
        acc = accuracy(train_y,out[train_idx].argmax(dim=1))

        mean_auc=mean_auc+auc

        # Validation

        val_acc = accuracy(val_y,out[val_idx].argmax(dim=1))

        val_auc = metrics.roc_auc_score(val_y,out[val_idx].argmax(dim=1))
                               
        mean_val_auc=mean_val_auc+val_auc
      
        # Print metrics every 10 epochs
        if(epoch % 10 == 0):
            print(f'Epoch {epoch:>3} |Train AUC: '
                  f'{auc*100:>6.2f}% | Train Loss: {loss:.3f} |  Train acc: {acc:.3f} | Val Loss: {val_loss:.2f} |  val acc: {val_acc:.3f} |'
                  f'val auc: {val_auc*100:.2f}%')

    plt.close('all')
    plt.figure(figsize=(10,5))
    plt.title("Training and Validation Loss")
    plt.plot(val_losses,label="val")
    plt.plot(train_losses,label="train")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    train_x = torch.cat([G[train_idx], X[train_idx]], dim=1)
    
    val_x = torch.cat([G[val_idx], X[val_idx]], dim=1)
    test_x = torch.cat([G[test_idx], X[test_idx]], dim=1)

    train_y=train_y.type(torch.FloatTensor)
    val_y=val_y.type(torch.FloatTensor)

    train_x=train_x.cpu()
    val_x=val_x.cpu()
    test_x=test_x.cpu()
    train_y = train_y.cpu()
    val_y = val_y.cpu()
    
    probs, val_probs,e = mlp_fit_predict(
        train_x, train_y, test_x, val=(val_x, val_y), return_val_probs=True)
    val_roc_auc = metrics.roc_auc_score(val_y.cpu().numpy(), val_probs)

    print('\nValidation ROC_AUC:', val_roc_auc)

    roc_aucs = []
    roc_auc = metrics.roc_auc_score(test_y, probs)
    roc_aucs.append(roc_auc)


    print('\nAuc(all):', roc_aucs)
    print(f'Auc Test: {np.mean(roc_aucs)*100:.2f}% ')

    test_acc=accuracy_score(test_y,probs.round())

    print(f'\nTest accracy: {test_acc*100:.2f}%')

    precision, recall, thresholds= metrics.precision_recall_curve(test_y, probs)
    
    # Use AUC function to calculate the area under the curve of precision recall curve
    auc_precision_recall = metrics.auc(recall, precision)
    print(f'\n Test precision recall curve: {auc_precision_recall*100:.2f}%' )

    auc_total = metrics.roc_auc_score(test_y,probs)

    # Average precision score 
    average_precision = metrics.average_precision_score(test_y, probs)
    print(f'\nTest average_precision: {average_precision*100:.2f}%')
    

    mcc=metrics.matthews_corrcoef(test_y, probs.round())
    print(f'\nTest matthews corrcoef: {mcc*100:.2f}%')

    m_auc=mean_auc/epochs
    m_val_auc=mean_val_auc/epochs
    print(f' mean auc: {m_auc*100:.2f}% | ' f'mean val auc: {m_val_auc*100:.2f}%')  

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

    # Building the modeel-----------------------------------------------

    gin = GIN(edge_index=edge_index,dim_in=X.shape[1],dim_h=32,dim_out=2)
    
    auc_total,auc_precision_recall,average_precision,mcc,test_acc=train_model(gin,X, edge_index,train_idx,train_y, val_idx,val_y,test_idx,test_y)


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

            print('random run number:',i)

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
