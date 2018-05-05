import os
import html
import numpy as np
import pandas as pd
import tensorflow as tf

def get_up_dict(data_dir, u_index, p_index):
    f = open(data_dir, 'r', encoding='utf-8', errors='ignore')
    u_dict = {}
    u_dict['unk'] = len(u_dict)
    p_dict = {}
    p_dict['unk'] = len(p_dict)
    u_freq = {}
    u_freq['unk'] = 0
    p_freq = {}
    p_freq['unk'] = 0
    for line in f:
        line = line.split('\t\t')
        user = line[u_index]
        product = line[p_index]
        if user not in u_dict:
            u_dict[user] = len(u_dict)
            u_freq[user] = 0
        u_freq[user] += 1
        if product not in p_dict:
            p_dict[product] = len(p_dict)
            p_freq[product] = 0
        p_freq[product] += 1
        
    f.close()
    
    return u_dict, p_dict, u_freq, p_freq

def get_dict(data_dir, x_index, min_count=2):
    f = open(data_dir, 'r', encoding='utf-8', errors='ignore')
    count = {}
    for line in f:
        line = line.split('\t\t')
        x_par = line[x_index].split('<sssss>')
        for x_sen in x_par:
            x_inp = x_sen.strip().split()
            for term in x_inp:
                if term not in count:
                    count[term] = 0
                count[term] += 1
    f.close()

    f = open(data_dir, 'r', encoding='utf-8', errors='ignore')
    x_dict = {}
    x_dict['<PAD>'] = len(x_dict)
#    x_dict['<UNK>'] = len(x_dict)
    for line in f:
        line = line.split('\t\t')
        x_par = line[x_index].split('<sssss>')
        for x_sen in x_par:
            x_inp = x_sen.strip().split()
            for term in x_inp:
                if term not in x_dict:
                    if count[term] >= min_count:
                        x_dict[term] = len(x_dict)
    f.close()
    
    return x_dict

def get_flat_dict(data_dir, x_index, min_count=2):
    #TODO
    pass
    
def get_flat_data(data_dir, x_index, y_index, x_dict, num_classes,
                  u_index, p_index, u_dict, p_dict, u_freq, p_freq):
    print(data_dir)
    x_dat = []
    i_dat = []
    x_len = []
    y_dat = []
    u_dat = []
    p_dat = []
    u_count = []
    p_count = []
    
    f = open(data_dir, 'r', encoding='utf-8', errors='ignore')
    for line in f:
        line = line.split('\t\t')
        y = int(line[y_index])
        u_id = line[u_index]
        if u_id not in u_dict:
            u_id = 'unk'
        p_id = line[p_index]
        if p_id not in p_dict:
            p_id = 'unk'
        u = u_dict[u_id]
        p = p_dict[p_id]
        uc = u_freq[u_id]
        pc = p_freq[p_id]
        y_onehot = np.zeros(shape=[num_classes])
        y_onehot[y-1] = 1
        
        x_par = line[x_index].split()
        x = []
        for term in x_par:
            if term not in x_dict:
                continue
            else:
                x.append(x_dict[term])
        
        y_dat.append(y_onehot)
        u_dat.append([u])
        p_dat.append([p])
        u_count.append([uc])
        p_count.append([pc])
        x_dat.append(x)
        i_dat.append(np.array(range(1, len(x)+1)))
        x_len.append(len(x))
    
    f.close()
    
    x_dat = np.array(x_dat)
    i_dat = np.array(i_dat)
    x_len = np.array(x_len)
    y_dat = np.array(y_dat)
    u_dat = np.array(u_dat)
    u_count = np.array(u_count)
    p_dat = np.array(p_dat)
    p_count = np.array(p_count)
    
    return x_dat, i_dat, x_len, y_dat, u_dat, p_dat, u_count, p_count

def get_hier_data(data_dir, x_index, y_index, x_dict, num_classes, 
                  u_index, p_index, u_dict, p_dict, u_freq, p_freq, slice=100):
    print(data_dir)
    x_dat = []
    x_input_len = []
    x_paragraph_len = []
    y_dat = []
    u_dat = []
    p_dat = []
    u_count = []
    p_count = []
    
    f = open(data_dir, 'r', encoding='utf-8', errors='ignore')
    for line in f:
        line = line.split('\t\t')
        y = int(line[y_index])
        u_id = line[u_index]
        if u_id not in u_dict:
            u_id = 'unk'
        p_id = line[p_index]
        if p_id not in p_dict:
            p_id = 'unk'
        u = u_dict[u_id]
        p = p_dict[p_id]
        uc = u_freq[u_id]
        pc = p_freq[p_id]
        y_onehot = np.zeros(shape=[num_classes])
        y_onehot[y-1] = 1
        
        x_par = line[x_index].split('<sssss>')
        par_len = len(x_par)
        inp_len = []
        x = []
        for x_sen in x_par:
            x_inp = x_sen.strip().split()
            for i in range(0, len(x_inp), slice):
                inp_len.append(len(x_inp[i:i+slice]))
                x_indices = []
                for term in x_inp[i:i+slice]:
                    if term not in x_dict:
                        continue
                    else:
                        x_indices.append(x_dict[term])
                
                x.append(x_indices)
        
        y_dat.append(y_onehot)
        u_dat.append([u])
        p_dat.append([p])
        u_count.append([uc])
        p_count.append([pc])
        x_dat.append(x)
        x_input_len.append(inp_len)
        x_paragraph_len.append(par_len)
    
    f.close()
    
    x_dat = np.array(x_dat)
    x_input_len = np.array(x_input_len)
    x_paragraph_len = np.array(x_paragraph_len)
    y_dat = np.array(y_dat)
    u_dat = np.array(u_dat)
    u_count = np.array(u_count)
    p_dat = np.array(p_dat)
    p_count = np.array(p_count)
    
    return x_dat, x_input_len, x_paragraph_len, y_dat, u_dat, p_dat, u_count, p_count

def get_vectors(x_dict, vec_file, emb_size=100):
    x_vectors = np.random.uniform(-0.1, 0.1, (len(x_dict), emb_size))
    
    f = open(vec_file, 'r', encoding='utf-8', errors='ignore')
    for line in f:
        line = line.split()
        if line[0] in x_dict:
            x_vectors[x_dict[line[0]]] = np.array([float(x) for x in line[-emb_size:]])
    
    f.close()
    
    return x_vectors

def get_up_vectors(dict, list_file, vec_file, emb_size=300):
    vectors = np.random.uniform(-0.1, 0.1, (len(dict), emb_size))
    
    lst = []
    f = open(list_file, 'r', encoding='utf-8', errors='ignore')
    for line in f:
        lst.append(line.strip())
    
    f.close()
    
    vecs = np.load(vec_file)
    for id, vec in zip(lst, vecs):
        if id in dict:
            vectors[dict[id]] = vec
    
    return vectors