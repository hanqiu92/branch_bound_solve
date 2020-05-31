import math
import time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as splinalg
from scipy.sparse import csc_matrix,csr_matrix,coo_matrix
from enum import Enum,unique

#############################
## 一些与数值误差相关的常数的定义
INF = 1e16
MAX = 1e20
TOL = 1e-6
#############################

import pandas as pd
from cylp.cy import CyClpSimplex


def cbc_solve(fname,time_limit=3600,solve_type=0):
    tt = time.time()
    s = CyClpSimplex()
    s.readMps(fname)
    cbcModel = s.getCbcModel()
    cbcModel.maximumSeconds = time_limit
    if solve_type == 1:
        cbcModel.branchAndBound() ## BB
    else:
        cbcModel.solve() ## 默认求解方法    
    LB = cbcModel.bestPossibleObjValue
    obj = cbcModel.objectiveValue
    status = cbcModel.status
    dt = time.time() - tt
    return (dt,status,LB,obj)

def load_benchmark(fname):
    df = []
    with open(fname,'r+') as f:
        lines = f.readlines()
    for line in lines:
        df += [[line[:11].strip()[1:-1],line[11:52].strip(),line[52:].strip()]]
    df = pd.DataFrame(df,columns=['status_best','model','obj_best'])
    inf_idxs = df['obj_best'] == ''
    df.loc[inf_idxs,'obj_best'] = '1e50'
    df['obj_best'] = df['obj_best'].astype(float)
    return df

def model2fname(model_name):
    return 'data/test_problem/{}.mps'.format(model_name)

def process_result(result,save_fname=None):
    df = pd.DataFrame(result).T.reset_index()
    df.columns = ['model','time','status','LB','obj']
    df['opt_gap'] = df['obj'] - df['LB']
    df['opt_gap_rate'] = df['opt_gap'] / np.maximum(np.maximum(np.abs(df['obj']),np.abs(df['LB'])),1e-8)
    if save_fname is not None:
        df.to_csv(save_fname,index=False)
    return df

def process_result_new(result,save_fname=None):
    df = pd.DataFrame(result).T.reset_index()
    df.columns = ['model','time','status','LB','obj','total_node','remain_node']
    df['opt_gap'] = df['obj'] - df['LB']
    df['opt_gap_rate'] = df['opt_gap'] / np.maximum(np.maximum(np.abs(df['obj']),np.abs(df['LB'])),1e-8)
    if save_fname is not None:
        df.to_csv(save_fname,index=False)
    return df