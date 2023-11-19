# coding=utf-8

import torch
from torch.optim import Adam
import numpy as np
from datetime import datetime
import argparse

import pdb
import shutil
import os
import yaml

from utility import Dataset
from TransMatch import TransMatch
from eval_utils import *

    
def evaluating(model, testData, device, conf, topks=[1]):
    model.eval()
    preds = []      
    for iteration, aBatch in enumerate(testData):
        aBatch = [x.to(device) for x in aBatch]
        scores = model.inference(aBatch).detach().cpu()
        _, tops = torch.topk(scores, k=topks[-1], dim=-1)
        preds.append(tops)
            
    preds = torch.cat(preds, dim=0)
    bs = preds.size(0)
    grd = [0] * bs
    grd_cnt = [1] * bs
    metrics = {}
    for topk in topks:
        metrics[topk] = {}
        REC, MRR, NDCG = get_metrics(grd, grd_cnt, preds.numpy(), topk)    
        metrics[topk]["recall"] = REC
        metrics[topk]["mrr"] = MRR
        metrics[topk]["ndcg"] = NDCG
    return metrics, preds


def Train_Eval(conf):
    dataset = Dataset(conf)
    conf["user_num"] = len(dataset.user_map)
    conf["item_num"] = len(dataset.item_map)
    conf["cate_num"] = len(dataset.cate_items)
    print("data prepared, %d users, %d items, %d train, %d test, %d validation data"%(len(dataset.user_map), len(dataset.item_map), len(dataset.traindata), len(dataset.testdata), len(dataset.valdata)))
    if conf["model"] == "TransMatch":
        model = TransMatch(conf, dataset.neighbor_params, dataset.visual_features.to(conf["device"]))
    model.to(conf["device"])

    optimizer = Adam([{'params': model.parameters(),'lr': conf["lr"], "weight_decay": conf["wd"]}])
    performance_files, result_path, model_path = get_save_file(conf, dataset.test_setting_list)
    model_name = "_".join(result_path.split("/")[-3:-1])
    if conf["save_results"] or conf["save_model"]:
        best_auc = 0
        best_hit = 0
        
    for epoch in range(conf["max_epoch"]):
        model.train()
        loss_scalar = 0.
        
        for iteration, aBatch in enumerate(dataset.train_loader):
            aBatch = [x.to(conf["device"]) for x in aBatch]
            loss = model.forward(aBatch)  
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_scalar += loss.detach().cpu()
        curr_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print("%s Epoch %d Loss: %.6f"%(curr_time, epoch, loss_scalar/iteration))
        test_results = {}
        if epoch % conf["evaluation_interval"] == 0:
            print(model_name)
            for test_setting, test_loader in zip(dataset.test_setting_list, dataset.test_loader_list):
                if "auc" in test_setting: # auc evalution
                    metrics, preds = evaluating(model, test_loader, conf["device"], conf)
                    curr_time = "%s "%datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    epoch_str = "Epoch %d"%epoch
                    result_str = ""
                    for met in metrics[1]:
                        auc = metrics[1][met]
                        result_str += " %s: %.4f"%("AUC", auc)
                        break
                    if conf["save_results"]:
                        if auc > best_auc:
                            best_auc = auc
                            np.save(result_path + "epoch_%d_%s_%.4f"%(epoch, test_setting, best_auc), preds.numpy())
       
                    print("%s"%test_setting[:-5], curr_time, result_str)
                    output_f = open(performance_files[test_setting], "a")
                    output_f.write(curr_time + "Epoch %d"%epoch + result_str + "\n")
                    output_f.close()
                else: # topk evaluation
                    metrics, preds = evaluating(model, test_loader, conf["device"], conf, conf["topk"])
                    curr_time = "%s "%datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    epoch_str = "Epoch %d"%epoch
                    output_f = open(performance_files[test_setting], "a")
                    result_str = ""
                    for topk in conf["topk"]:
                        for met in metrics[conf["topk"][0]]:
                            hit = metrics[topk][met]
                            result_str += " %s@%d: %.4f"%("Hit", topk, hit)
                            break
                            
                        if conf["save_results"]:
                            if hit > best_hit:
                                best_hit = hit
                                np.save(result_path + "epoch_%d_%s_hit@%d_%.4f"%(epoch, test_setting, topk, best_hit), preds.numpy())
                        if conf["save_model"]:
                            if hit > best_hit:
                                best_hit = hit
                                shutil.rmtree(model_path)
                                os.makedirs(model_path) 
                                torch.save(model, model_path + "epoch_%d_%s_hit@%d_%.4f_gpu_%s"%(epoch, test_setting, topk, best_hit, conf["gpu"])) 
                    print("%s"%(test_setting[:-5]), curr_time, result_str)
                    output_f.write(curr_time + "Epoch %d"%epoch + result_str + "\n")
                    output_f.close()

                
def get_save_file(conf, settings):
    if conf["model"] == "TransMatch":      
        f_name = "TransMatch_"+conf["score_type"]
        if conf["context"]:
            f_name += "_pcc"
            f_name += "_"+str(conf["context_hops"])
            f_name += "_"+str(conf["neighbor_samples"])
            f_name += "_%s"%conf["neighbor_agg"]
            f_name += "_%.2f"%conf["agg_param"]
        if conf["path"]:
            f_name += "_pcp_%d_%d_%.2f_%s"%(conf["path_num"], conf["max_path_len"], conf["path_weight"], conf["path_agg"])
                    
    f_name += "/"
    performance_path = conf["root_path"] + conf["performance_path"] + f_name
    model_path = conf["root_path"] + conf["model_path"] + f_name
    result_path = conf["root_path"] + conf["result_path"] + f_name
    f_list = {}
    if not os.path.exists(performance_path):
        os.makedirs(performance_path)
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    if not os.path.exists(model_path):
        os.makedirs(model_path)    
    for setting in settings:
        output_file = performance_path + setting
        f_list[setting] = output_file
    return f_list, result_path, model_path
        

def get_cmd(): 
    parser = argparse.ArgumentParser()
    # general params
    parser.add_argument("-d", "--dataset", default="iqon", type=str, help="polyvore, iqon, iqon_s")    
    parser.add_argument("-g", "--gpu", default="0", type=str, help="assign cuda device")    
    args = parser.parse_args()
    return args



if __name__ == "__main__": 
    paras = get_cmd().__dict__
    conf = yaml.safe_load(open("./train_model_config.yaml"))
    for k in paras:
        conf[k] = paras[k]  
    conf["device"] = torch.device("cuda:%s"%conf["gpu"] if torch.cuda.is_available() else "cpu")
    
    if conf["wide_evaluate"]:
        conf["test_batch_size"] = 64
        
    conf["performance_path"] += (conf["dataset"] + "/")
    conf["result_path"] += (conf["dataset"] + "/")
    conf["model_path"] += (conf["dataset"] + "/")
    Train_Eval(conf)

                               

