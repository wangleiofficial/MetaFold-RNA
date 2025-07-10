import os
import math
import torch
import random
import argparse
import numpy as np
from itertools import product



perm = list(product(np.arange(4), np.arange(4)))


def one_hot(seq):
    RNN_seq = seq
    BASES = 'AUCG'
    bases = np.array([base for base in BASES])
    feat = np.concatenate(
            [[(bases == base.upper()).astype(int)] if str(base).upper() in BASES else np.array([[0] * len(BASES)]) for base
            in RNN_seq])
    return feat


def Gaussian(x):
    return math.exp(-0.5*(x*x))


def paired(x,y):
    if x == 'A' and y == 'U':
        return 2
    elif x == 'G' and y == 'C':
        return 3
    elif x == 'G' and y == 'U':
        return 0.8
    elif x == 'U' and y == 'A':
        return 2
    elif x == 'C' and y == 'G':
        return 3
    elif x == 'U' and y == 'G':
        return 0.8
    else:
        return 0
    

def creatmat(data):
    mat = np.zeros([len(data),len(data)])
    for i in range(len(data)):
        for j in range(len(data)):
            coefficient = 0
            for add in range(30):
                if i - add >= 0 and j + add <len(data):
                    score = paired(data[i - add],data[j + add])
                    if score == 0:
                        break
                    else:
                        coefficient = coefficient + score * Gaussian(add)
                else:
                    break
            if coefficient > 0:
                for add in range(1,30):
                    if i + add < len(data) and j - add >= 0:
                        score = paired(data[i + add],data[j - add])
                        if score == 0:
                            break
                        else:
                            coefficient = coefficient + score * Gaussian(add)
                    else:
                        break
            mat[[i],[j]] = coefficient
    return mat


def feature(matrix, l, seq):
    data_l = len(seq)
    data_fcn = np.zeros((16, l, l))
    for n, cord in enumerate(perm):
        i, j = cord
        data_fcn[n, :data_l, :data_l] = np.matmul(matrix[:data_l, i].reshape(-1, 1), matrix[:data_l, j].reshape(1, -1))
    data_fcn_1 = np.zeros((1,l,l))
    datalist = list(seq)
    data_fcn_1[0,:data_l,:data_l] = creatmat(datalist)
    data_fcn_2 = np.concatenate((data_fcn,data_fcn_1),axis=0) 
    return data_fcn_2


def metrics_fn(pred_a, true_a, eps=1e-11):
    tp_map = torch.sign(torch.Tensor(pred_a)*torch.Tensor(true_a))
    tp = tp_map.sum()
    pred_p = torch.sign(torch.Tensor(pred_a)).sum()
    true_p = true_a.sum()
    fp = pred_p - tp
    fn = true_p - tp
    recall = (tp + eps)/(tp+fn+eps)
    precision = (tp + eps)/(tp+fp+eps)
    f1_score = (2*tp + eps)/(2*tp + fp + fn + eps)
    return precision, recall, f1_score


def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def fastaprocess(file_path):
    data = {}
    bases = "AUCG"
    with open(file_path, 'r') as file:
        name = None
        sequence = ""
        for line in file:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                if name:
                    data[name] = sequence
                name = line[1:] or "input"
                sequence = ""
            else:
                line = line.replace(" ", "").upper().replace("T", "U")
                line = "".join(random.choice(bases) if char == "N" else char for char in line)
                sequence += line
        if name:
            data[name] = sequence
    return data


def save2bpseq(sequence, pred, bpseq_path):
    rna_length = len(sequence)
    paired_bases = {}
    for i in range(rna_length):
        for j in range(i + 1, rna_length):
            if pred[i, j].item() == 1:
                paired_bases[i] = j + 1
                paired_bases[j] = i + 1
    bpseq_lines = []
    for i in range(rna_length):
        base_index = i + 1
        base_char = sequence[i]
        pair_index = paired_bases.get(i, 0)
        bpseq_lines.append(f"{base_index} {base_char} {pair_index}")
    os.makedirs(os.path.dirname(bpseq_path), exist_ok=True)
    with open(bpseq_path, "w") as f:
        f.write("\n".join(bpseq_lines) + "\n")
    print(f'bpseq file saved: {bpseq_path}')








