from torch.utils.data import Dataset, DataLoader
import random
import torch

###########################################################################
######## training and testing for general methods ######################### 
class Train_Data(Dataset):
    def __init__(self, train_data, cate_items, item_cates):
        self.train_data = train_data
        self.cate_items = {}
        for c, i_list in cate_items.items():
            self.cate_items[int(c)] = i_list
        self.item_cates = {}
        for i, c in item_cates.items():
            self.item_cates[int(i)] = c
        
    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, idx):
        u, i, j, _ = self.train_data[idx]
        j_cate = self.item_cates[j]
        candi_js = self.cate_items[j_cate]
        if len(candi_js) == 1: # there are some bad data in mg_pfcm_iqon
            neg_j = j
        else:
            neg_j = random.choice(candi_js)
            while neg_j == j:
                neg_j = random.choice(candi_js)   

        return u, i, j, neg_j, idx
    

class Test_Data(Dataset):
    def __init__(self, test_data):
        self.test_data = test_data
        
    def __len__(self):
        return len(self.test_data)

    def __getitem__(self, idx):
        u, i, j, neg_j = self.test_data[idx]
        return u, i, j, torch.LongTensor([neg_j])    

    
######## multi-negative test for general methods)##########################         
class Wide_Test_Data(Dataset):
    def __init__(self, test_data):
        self.test_data = test_data
    
    def __len__(self):
        return len(self.test_data)
    
    def __getitem__(self, idx):
        u = self.test_data[idx][0]
        i = self.test_data[idx][1]
        j = self.test_data[idx][2]
        neg_j = self.test_data[idx][3:]
        return u, i, j, neg_j    
    
    
###############################################################################
######## training and testing for PathCon with Path ###########################   
class Train_Data_Path(Dataset):
    def __init__(self, train_data, cate_items, item_cates, ht_dict, ht_paths, e2re, max_path_len, max_path_num, null_relation, neg_u_path_num=0):
        self.train_data = train_data
        self.cate_items = {}
        for c, i_list in cate_items.items():
            self.cate_items[int(c)] = i_list
        self.item_cates = {}
        for i, c in item_cates.items():
            self.item_cates[int(i)] = c
        self.ht_dict = ht_dict
        self.ht_paths = ht_paths
        self.e2re = e2re
        self.max_path_len = max_path_len
        self.max_path_num = max_path_num
        self.null_relation = null_relation
        self.neg_u_path_num = neg_u_path_num
        if self.neg_u_path_num != 0:
            all_path = set()
            for path in self.ht_paths:
                for p in path:
                    if p != [null_relation] * max_path_len:
                        p_str = "_".join([str(i) for i in sorted(p)])
                        all_path.add(p_str)
            self.all_path = []
            for p in all_path:
                self.all_path.append([int(k) for k in p.split("_")])
        
    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, idx):
        u, i, j, _ = self.train_data[idx]
        j_cate = self.item_cates[j]
        candi_js = self.cate_items[j_cate]
        if len(candi_js) == 1: # there are some bad data in mg_pfcm_iqon
            neg_j = j
        else:
            neg_j = random.choice(candi_js)
            while neg_j == j:
                neg_j = random.choice(candi_js)   
        ij = str(i) + "_" + str(j)
        try:
            ij_path = self.ht_paths[self.ht_dict[ij]]
        except Exception:
            ij_path = [[self.null_relation]*self.max_path_len] * self.max_path_num    
        ij_path_mask = torch.LongTensor(ij_path) - torch.LongTensor([self.null_relation])
        ij_path_mask = (ij_path_mask != 0).float()
        ik = str(i) + "_" + str(neg_j)
        try:
            ik_path = self.ht_paths[self.ht_dict[ik]]
        except Exception:
            ik_path = [[self.null_relation]*self.max_path_len] * self.max_path_num
        ik_path_mask = torch.LongTensor(ik_path) - torch.LongTensor([self.null_relation])
        ik_path_mask = (ik_path_mask != 0).float()
        if self.neg_u_path_num != 0:
            neg_u_path = []
            neg_u_path_mask = []
            path = random.choice(self.all_path)
            if path != ij_path:
                neg_u_path.append(path)
                mask = torch.LongTensor(path) - torch.LongTensor([self.null_relation])
                neg_u_path_mask.append((mask != 0).float())
            while len(neg_u_path) < self.neg_u_path_num:
                path = random.choice(self.all_path)
                if path != ij_path:
                    neg_u_path.append(path)
                    mask = torch.LongTensor(path) - torch.LongTensor([self.null_relation])
                    neg_u_path_mask.append((mask != 0).long())
            return u, i, j, neg_j, idx, torch.LongTensor(ij_path), torch.LongTensor(ik_path), ij_path_mask, ik_path_mask, torch.LongTensor(neg_u_path), torch.stack(neg_u_path_mask, dim=0)
        else:
            return u, i, j, neg_j, idx, torch.LongTensor(ij_path), torch.LongTensor(ik_path), ij_path_mask, ik_path_mask

 
                 
class Test_Data_Path(Dataset):
    def __init__(self, test_data, ht_dict, ht_paths, e2re, max_path_len, max_path_num, null_relation):
        self.test_data = test_data
        self.ht_dict = ht_dict
        self.ht_paths = ht_paths
        self.e2re = e2re
        self.max_path_len = max_path_len
        self.max_path_num = max_path_num
        self.null_relation = null_relation
        
    def __len__(self):
        return len(self.test_data)

    def __getitem__(self, idx):
        u, i, j, neg_j = self.test_data[idx]
        ij = str(i) + "_" + str(j)
        try:
            ij_path = self.ht_paths[self.ht_dict[ij]]
        except Exception:
            ij_path = [[self.null_relation]*self.max_path_len] * self.max_path_num
        ij_path_mask = torch.LongTensor(ij_path) - torch.LongTensor([self.null_relation])
        ij_path_mask = (ij_path_mask != 0).float()
        
        ik = str(i) + "_" + str(neg_j)
        try:
            ik_path = self.ht_paths[self.ht_dict[ik]]
        except Exception:
            ik_path = [[self.null_relation]*self.max_path_len] * self.max_path_num
        ik_path_mask = torch.LongTensor(ik_path) - torch.LongTensor([self.null_relation])
        ik_path_mask = (ik_path_mask != 0).float()
        return u, i, j, torch.LongTensor([neg_j]), torch.LongTensor(ij_path), torch.LongTensor(ik_path), ij_path_mask, ik_path_mask
     

######## multi-negative testing for PathCon with Path #######################  
class Wide_Test_Data_Path(Dataset):
    def __init__(self, test_data, ht_dict, ht_paths, e2re, max_path_len, max_path_num, null_relation):
        self.test_data = test_data
        self.ht_dict = ht_dict
        self.ht_paths = ht_paths
        self.e2re = e2re
        self.max_path_len = max_path_len
        self.max_path_num = max_path_num
        self.null_relation = null_relation
        
    def __len__(self):
        return len(self.test_data)

    def __getitem__(self, idx):
        u = self.test_data[idx][0]
        i = self.test_data[idx][1]
        j = self.test_data[idx][2]
        neg_js = self.test_data[idx][3:]
        
        ij = str(i) + "_" + str(j)
        try:
            ij_path = self.ht_paths[self.ht_dict[ij]]
        except Exception:
            ij_path = [[self.null_relation]*self.max_path_len] * self.max_path_num
        ij_path_mask = torch.LongTensor(ij_path) - torch.LongTensor([self.null_relation])
        ij_path_mask = (ij_path_mask != 0).float()
        
        ik_paths = []
        ik_path_masks = []
        for neg_j in neg_js:
            ik = str(i) + "_" + str(neg_j)
            try:
                ik_path = self.ht_paths[self.ht_dict[ik]]
            except Exception:
                ik_path = [[self.null_relation]*self.max_path_len] * self.max_path_num
            ik_path = torch.LongTensor(ik_path)
            ik_path_mask = torch.LongTensor(ik_path) - torch.LongTensor([self.null_relation])
            ik_path_mask = (ik_path_mask != 0).float()
            ik_paths.append(ik_path)
            ik_path_masks.append(ik_path_mask)
        return u, i, j, torch.LongTensor(neg_js), torch.LongTensor(ij_path), torch.stack(ik_paths, dim=0), ij_path_mask, torch.stack(ik_path_masks, dim=0)
  
    
    