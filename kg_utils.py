from collections import defaultdict
import pdb
import multiprocessing as mp
from datetime import datetime

def build_kg(train_data):
    # train_data[t] = u, i, j (r, h, t)
    edge2entities = []
    edge2relation = []
    e2re = defaultdict(set)
    entity2edge_set = defaultdict(set) 
    relation2entity_set = {}
    
    for edge_idx, triplet in enumerate(train_data):
        relation_id, head_id, tail_id, _ = triplet
        entity2edge_set[head_id].add(edge_idx)
        entity2edge_set[tail_id].add(edge_idx)
        edge2entities.append([head_id, tail_id])
        edge2relation.append(relation_id)
        if relation_id not in relation2entity_set:
            relation2entity_set[relation_id] = []
        relation2entity_set[relation_id].append(head_id)
        relation2entity_set[relation_id].append(tail_id)
        e2re[head_id].add((relation_id, tail_id))
        e2re[tail_id].add((relation_id, head_id))  
        
    return entity2edge_set, edge2entities, edge2relation, e2re, relation2entity_set


    
def shuffle_gragh(entity2edges, entity2edge_set, neighbor_samples):
    curr_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("%s shuffle kg ..."%curr_time)
    for i in range(len(entity2edge_set)):
        edges = torch.Tensor(list(entity2edge_set[i]))
        try:
            sampled_neighbors = torch.multinomial(edges, neighbor_samples, replacement=len(entity2edge_set[i]) < neighbor_samples)
        except Exception:
            continue
        entity2edges[int(i)] = edges[sampled_neighbors].long()
        
    return entity2edges


def get_path(head2tails, e2re, entity2edge_set, edge2entities, relation2entity_set, max_path_len, path_num):
    entity_path = {}
    entity_pair_path = {}
    relation_path = {}
    ht2paths = {}
    for cnt, (head, tails) in enumerate(head2tails):
        pdb.set_trace()
        ht2paths.update(bfs(head, tails, e2re, max_path_len))
    
def get_params_for_mp(n_triples):
    n_cores = mp.cpu_count()
    pool = mp.Pool(n_cores)
    avg = n_triples // n_cores

    range_list = []
    start = 0
    for i in range(n_cores):
        num = avg + 1 if i < n_triples - avg * n_cores else avg
        range_list.append([start, start + num])
        start += num

    return n_cores, pool, range_list
    
def get_h2t(train_triplets, test_triplets, val_triplets):
    head2tails = defaultdict(set)
    for relation, head, tail, _ in train_triplets + test_triplets + val_triplets:
        head2tails[head].add(tail)
#         head2tails[tail].add(head)
    return head2tails  
    
    
def count_all_paths(inputs):
    e2re, max_path_len, head2tails, item_cate, pid = inputs
    ht2paths = {}
    for i, (head, tails) in enumerate(head2tails):
#     for head, tails in head2tails.items():
        ht2paths.update(bfs(head, tails, e2re, max_path_len, item_cate))
    return ht2paths
    
    
def bfs(head, tails, e2re, max_path_len, item_cate):
    # put length-1 paths into all_paths
    # each element in all_paths is a path consisting of a sequence of (relation, entity)
    all_paths = [[i] for i in e2re[head]]

    p = 0
    for length in range(2, max_path_len + 1):
        while p < len(all_paths) and len(all_paths[p]) < length:
            path = all_paths[p]
            last_entity_in_path = path[-1][1]
            entities_in_path = set([head] + [i[1] for i in path])
            for edge in e2re[last_entity_in_path]:
                # append (relation, entity) to the path if the new entity does not appear in this path before
                if edge[1] not in entities_in_path:
                    all_paths.append(path + [edge])
            p += 1

    ht2paths = defaultdict(set)
    h_cate = item_cate[str(head)]
    for path in all_paths:
        tail = path[-1][1]
        t_cate = item_cate[str(tail)]
        if h_cate != t_cate:
#         if tail in tails:  # if this path ends at tail
            ht2paths[(head, tail)].add(tuple([i[0] for i in path])) #only contains relation, no node

    return ht2paths


            