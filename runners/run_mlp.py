import os
import sys
import math 
sys.path.append('.')
import argparse

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.metrics import roc_auc_score

from utils.utils import set_seed
from sklearn.metrics import accuracy_score
from sklearn import metrics
import tools

#DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DEVICE = torch.device('cpu' )
class Loss():
    def __init__(self, y):
        self.y = y
        self.pos_mask = y == 1
        self.neg_mask = y == 0

    def __call__(self, out):
        pos_mask = self.pos_mask
        neg_mask = self.neg_mask
        loss_p = F.binary_cross_entropy_with_logits(
            out[pos_mask].squeeze(), self.y[self.pos_mask].cpu())
        loss_n = F.binary_cross_entropy_with_logits(
            out[neg_mask].squeeze(), self.y[neg_mask].cpu())
        loss = loss_p + loss_n
        return loss


def acc(t1, t2):
    return np.sum(t1*1 == t2*1) / len(t1)


def mlp_fit_predict(train_x, train_y, test_x, val=None, return_val_probs=False):
    epochs = 1000#1000

    in_feats = train_x.shape[1]
    model = nn.Sequential(
        nn.Linear(in_feats, 32),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(32, 1))
    optimizer = torch.optim.Adam(model.parameters())

    lossf = Loss(train_y)

    if val is not None:
        val_x, val_y = val
        lossf_val = Loss(val_y)

    model.train()
   

    patience, cur_es = 20, 0
    val_loss_old = np.Inf

    for i in range(epochs):
    
        out = model(train_x)
        loss = lossf(out)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i % 10) == 0:
            if val is not None:
                model.eval()
                with torch.no_grad():
                    loss_val = lossf_val(model(val_x))
                print(f'{i}. Train loss:', loss.detach().cpu().numpy(), ' |  Validation Loss:', loss_val.detach().cpu().numpy())
                model.train()

                if val_loss_old < loss_val:
                    cur_es += 1
                else:
                    cur_es = 0
                val_loss_old = loss_val

                if cur_es == patience:
                    print('**************')
                    break

    model.eval()
    with torch.no_grad():
        out = model(test_x).cpu()
    probs = torch.sigmoid(out).numpy()
    
    if return_val_probs:
        with torch.no_grad():
            out = model(val_x).cpu()
        val_probs = torch.sigmoid(out).numpy()

        return probs, val_probs

    return probs

def variance(data):
    # Number of observations
     n = len(data)
     
    # Mean of the data
     mean = sum(data) / n
     
    # Square deviations
     deviations = [(x - mean) ** 2 for x in data]
     
   # Variance
     variance = sum(deviations) / n
     

     return variance
def get_name(args):
    if args.name:
        return args.name

    name = 'MLP'
    if args.no_ppi:
        name += '_NO-PPI'
    if args.expression:
        name += '_EXP'
    if args.sublocs:
        name += '_SUB'
    if args.orthologs:
        name += '_ORT'

    return name
def main(args):
   
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
    
   

    #if args.hyper_search:
    #    hyper_search(args)

    if args.n_runs:
        args.train = True
        args.test = True

        scores = []
        for i in range(args.n_runs):
              seed = i
              set_seed(seed)
              mean_auc=0
              mean_val_auc=0
              print('random run number:',i)

              (edge_index, edge_weights), X,labels, (train_idx, train_y), \
              (val_idx, val_y), (test_idx, test_y), genes = tools.get_data(
                  args.__dict__, seed=seed)

              train_x = X[train_idx]
    
              val_x = X[val_idx]
              test_x =X[test_idx]

              train_y=train_y.type(torch.FloatTensor)
              val_y=val_y.type(torch.FloatTensor)

              train_x=train_x.cpu()
              val_x=val_x.cpu()
              test_x=test_x.cpu()
              train_y = train_y.cpu()
              val_y = val_y.cpu()

              probs, val_probs =  mlp_fit_predict(
              train_x, train_y, test_x, val=(val_x, val_y), return_val_probs=True)
              val_roc_auc = metrics.roc_auc_score(val_y.cpu().numpy(), val_probs)

              print('\nValidation ROC_AUC:', val_roc_auc)

              roc_aucs = []
              roc_auc = metrics.roc_auc_score(test_y, probs)
              roc_aucs.append(roc_auc)


              print('\nAuc(all):', roc_aucs)
              print(f'Auc Test: {np.mean(roc_aucs)*100:.2f}% ')

            

              
              test_acc=accuracy_score(test_y,probs.round())
              # test_acc = accuracy(test_y, torch.max(probs,1))
              print(f'\nTest accracy: {test_acc*100:.2f}%')

              precision, recall, thresholds= metrics.precision_recall_curve(test_y, probs)
              
              #Recall=metrics.recall_score(probs, test_y, average='micro')
              #print('\n Test recall:', recall)
              ##ValueError: Classification metrics can't handle a mix of continuous and binary targets

              # Use AUC function to calculate the area under the curve of precision recall curve

              auc_precision_recall = metrics.auc(recall, precision)
              print(f'\n Test precision recall curve: {auc_precision_recall*100:.2f}%' )

              auc_total = metrics.roc_auc_score(test_y,probs)
              # print(f'\n Test auc: {auc_total*100:.2f}%', )


              # Average precision score 
              average_precision = metrics.average_precision_score(test_y, probs)
              print(f'\nTest average_precision: {average_precision*100:.2f}%')
              

              mcc=metrics.matthews_corrcoef(test_y, probs.round())
              print(f'\nTest matthews corrcoef: {mcc*100:.2f}%')

              # m_auc=mean_auc/epochs
              # m_val_auc=mean_val_auc/epochs
              # print(f' mean auc: {m_auc*100:.2f}% | ' f'mean val auc: {m_val_auc*100:.2f}%')  
              auc_average=auc_total+auc_average

              auc_roc_varience.append(auc_total)


              auc_precision_recall_average=auc_precision_recall+auc_precision_recall_average

              auc_pr_varience.append(auc_precision_recall)


              average_precision_average=average_precision+average_precision_average

              precision_varience.append(average_precision)


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

        print('\n Test precision recall curve: {:.2f}%'.format(auc_precision_recall_average*100) )
        # print(f'\n Test precision recall curve: {auc_precision_recall_average*100:.2f}%' )

        print('\n precision average: {:.2f}%'.format(average_precision_average*100) )
    
        print('\n Test matthews corrcoef average: {:.2f}%'.format(mcc_average*100))

        print('\n Test accracy: {:.2f}%'.format(test_acc_average*100))
        

        v1=math.sqrt(variance(auc_roc_varience))

        v2=math.sqrt(variance(auc_pr_varience))
        
        v3=math.sqrt(variance(precision_varience))

        v4=math.sqrt(variance(mcc_varience))

        v5=math.sqrt(variance(acc_varience))

        print('********varience*********')
        print(f'\n auc-roc varience: {v1:.2f}%'  )

        print(f'\n auc-pr varience: {v2:.2f}%'  )

        print(f'\n precision varience: {v3:.2f}%' )

        print('\n matthews corrcoef varience: {:.2f}%'.format(v4)  )

        print('\n accracy varience: {:.2f}%'.format(v5)  )

        print(f'\n 	\lr{{MLP}} & ${auc_average*100:.2f}(\pm{v1:.2f})$& ${auc_precision_recall_average*100:.2f}(\pm{v2:.2f})$ & ${average_precision_average*100:.2f}(\pm{v3:.2f})$ & ${mcc_average*100:.2f}(\pm{v4:.2f})$  & ${test_acc_average*100:.2f}(\pm{v5:.2f})$\\\\hline'  )

    return probs, val_probs


def save_preds(preds, args, seed):
    name = get_name(args) + f'_{args.organism}_{args.ppi}_s{seed}.csv'
    name = name.lower()
    path = os.path.join('outputs/preds/', name)
    df = pd.DataFrame(preds, columns=['Gene', 'Pred'])
    df.to_csv(path)
    print('Saved the predictions to:', path)


if __name__ == '__main__':
    args = tools.get_args()

    main(args)

    name = get_name(args)

    df_path = 'outputs/results/results.csv'
    df = pd.read_csv(df_path)

    # df.loc[len(df)] = [name, args.organism, args.ppi, args.expression,
    #                    args.orthologs, args.sublocs, args.n_runs, mean, std]
    df.to_csv(df_path, index=False)
    # print(df.head())
