import os
import matplotlib.pyplot as plt
import pdb
import argparse
import yaml

def get_res_from_one_file(path):
    results = {}
    best_results = {}
    metrics = ["auc"]
    for met in metrics: 
        results[met] = []
        best_results[met] = 0
    best_results["epoch"] = 0
    performance = open(path).readlines()
    for ind, each_perform in enumerate(performance):
        res = each_perform.split(" ")
        epoch = int(res[3])
        auc = float(res[5])
        if auc > best_results["auc"]:
            best_results["auc"] = auc
            best_results["epoch"] = epoch
            best_results["index"] = ind
        results["auc"].append(auc)

    return results, best_results


def get_best_thr_cross_valid(root_path, model, anchor="val"):
    test_f = root_path + model + "/test_auc"
    val_f = root_path + model + "/val_auc"  
    val_results, val_best_results = get_res_from_one_file(val_f)
    test_results, test_best_results = get_res_from_one_file(test_f)

    if anchor == "val":
        best_index = val_best_results["index"]
        best_epoch = val_best_results["epoch"]
        best_result = {}
        for metric in test_results.keys():
            best_result[metric] = test_results[metric][best_index]
            break
    else:
        best_index = test_best_results["index"]
        best_epoch = test_best_results["epoch"]
        best_result = {}
        for metric in test_results.keys():
            try:
                best_result[metric] = val_results[metric][best_index]
            except Exception:
                pdb.set_trace()
            break
            
    return best_epoch, best_result

 
def show_multiple_cross_valid_results(performance_path, anchor, condition1=None, condition2=None):
    best_results = {}
    best_epochs = {}
    g = os.walk(performance_path)
    print("%67s  AUC    epoch "%"                                               ")
    
    for path, dir_list, file_list in g:
        for per_dir in dir_list:
            if per_dir == ".ipynb_checkpoints":
                continue
            if not os.listdir(path + "/" + per_dir):
                continue
            flag = True
            if condition1 is not None:
                for c in condition1:
                    if c not in per_dir:
                        flag = False
            if condition2 is not None:
                for c in condition2:
                    if c in per_dir:
                        flag = False
            if flag:
                best_epoch, best_result = get_best_thr_cross_valid(path + "/", per_dir, anchor)
                best_results[per_dir] = best_result["auc"]
                best_epochs[per_dir] = best_epoch
    sorted_best_results = sorted(best_results.items(), key=lambda x: x[1], reverse=True)
    for m, auc in sorted_best_results:   
        print("%65s  %.4f   %2d"%(m, auc, best_epochs[m]))
    return best_results


def get_cmd(): 
    parser = argparse.ArgumentParser()
    # general params
    parser.add_argument("-d", "--dataset", default="iqon", type=str, help="polyvore, iqon, iqon_s")    
    args = parser.parse_args()
    return args


if __name__ == "__main__": 
    paras = get_cmd().__dict__
    dataset = paras["dataset"]
    conf = yaml.safe_load(open("./train_model_config.yaml"))
    performance_path = "%s/performance/%s"%(conf["root_path"], dataset)
    res = show_multiple_cross_valid_results(performance_path, "test")