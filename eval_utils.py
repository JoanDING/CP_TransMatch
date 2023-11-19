import math
import numpy as np

def getHIT_MRR(pred, target_items):
    hit= 0.
    mrr = 0.
    p_1 = []
    for p in range(len(pred)):
        pre = pred[p]
        if pre in target_items:
            hit += 1
            if pre not in p_1:
                p_1.append(pre)
                mrr = 1./(p+1)

    return hit, mrr


def getNDCG(rank_list, target_items):
    dcg = 0
    idcg = IDCG(len(target_items))
    for i in range(len(rank_list)):
        item_id = rank_list[i]
        if item_id not in target_items:
            continue
        rank = i + 1
        dcg += 1./math.log(rank+1, 2)

    return dcg/idcg


def IDCG(n):
    idcg = 0.
    for i in range(n):
        idcg += 1./math.log(i+2, 2)

    return idcg


def get_metrics(grd, grd_cnt, pred, topk):
    REC, MRR, NDCG = [],[],[]
    for each_grd, each_grd_cnt, each_pred in zip(grd, grd_cnt, pred):
        NDCG.append(getNDCG(each_pred[:topk], [each_grd][:each_grd_cnt]))
        hit, mrr = getHIT_MRR(each_pred[:topk], [each_grd][:each_grd_cnt])
        REC.append(hit)
        MRR.append(mrr)
        
    REC = np.mean(REC)
    MRR = np.mean(MRR)
    NDCG = np.mean(NDCG)

    return REC, MRR, NDCG