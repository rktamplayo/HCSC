import tensorflow as tf
import numpy as np
import hcsc
import utils
import sys

# FILE LOCATION
data_dir = sys.argv[1]
base_model = sys.argv[2]

if len(sys.argv) == 3:
    train_type = ''
else:
    train_type = sys.argv[3]
train_file = data_dir + '/train' + train_type + '.txt'
dev_file = data_dir + '/dev.txt'
test_file = data_dir + '/test.txt'
vec_file = 'glove.840B.300d.txt'
pickle_file = data_dir + '/model' + train_type + '.p'

# PARAMETERS
emb_size = 300
epoch = 300
x_index = 3
y_index = 2
u_index = 0
p_index = 1
if 'yelp' in data_dir:
    num_classes = 5
else:
    num_classes = 10
batch_size = 32

import os
import time
import pickle as cPickle
if os.path.isfile(pickle_file):
    a = time.time()
    x_train, i_train, x_train_len, y_train, \
        u_train, p_train, u_train_count, p_train_count, \
        x_dev, i_dev, x_dev_len, y_dev, \
        u_dev, p_dev, u_dev_count, p_dev_count, \
        x_test, i_test, x_test_len, y_test, \
        u_test, p_test, u_test_count, p_test_count, \
        x_dict, u_dict, p_dict, u_freq, p_freq, \
        x_vectors = cPickle.load(open(pickle_file, 'rb'))
else:
    x_dict = utils.get_dict(train_file, x_index)
    u_dict, p_dict, u_freq, p_freq = utils.get_up_dict(train_file, u_index, p_index)
    x_train, i_train, x_train_len, y_train, u_train, p_train, \
        u_train_count, p_train_count = utils.get_flat_data(train_file, x_index, y_index, x_dict, num_classes,
                                               u_index, p_index, u_dict, p_dict, u_freq, p_freq)
    x_dev, i_dev, x_dev_len, y_dev, u_dev, p_dev, \
        u_dev_count, p_dev_count = utils.get_flat_data(dev_file, x_index, y_index, x_dict, num_classes,
                                               u_index, p_index, u_dict, p_dict, u_freq, p_freq)
    x_test, i_test, x_test_len, y_test, u_test, p_test, \
        u_test_count, p_test_count = utils.get_flat_data(test_file, x_index, y_index, x_dict, num_classes,
                                               u_index, p_index, u_dict, p_dict, u_freq, p_freq)
    x_vectors = utils.get_vectors(x_dict, vec_file, emb_size)
    cPickle.dump([x_train, i_train, x_train_len, y_train, \
                  u_train, p_train, u_train_count, p_train_count, \
                  x_dev, i_dev, x_dev_len, y_dev, \
                  u_dev, p_dev, u_dev_count, p_dev_count, \
                  x_test, i_test, x_test_len, y_test, \
                  u_test, p_test, u_test_count, p_test_count, \
                  x_dict, u_dict, p_dict, u_freq, p_freq, \
                  x_vectors], open(pickle_file, 'wb'), protocol=4)

vocab_size = len(x_dict)
target_len = num_classes
user_vocab_size = len(u_dict)
prod_vocab_size = len(p_dict)
user_mean_freq = np.mean([u_freq[x] for x in u_freq])
prod_mean_freq = np.mean([p_freq[x] for x in p_freq])
i_max = np.max([np.max(np.concatenate(i_train)), np.max(np.concatenate(i_test))])
model = hcsc.Model(base_model,
                   vocab_size,
                   target_len,
                   user_vocab_size,
                   prod_vocab_size,
                   user_mean_freq,
                   prod_mean_freq,
                   embedding_size=emb_size,
                   batch_size=batch_size)

sess = tf.Session()

tf.set_random_seed(1234)
np.random.seed(1234)

sess.run(tf.global_variables_initializer())
sess.run(model.embedding.assign(x_vectors))

import time
step = 0
best_acc = 0
cur_time = time.time()
step_check = len(x_train)//10
with sess.as_default():
    for i in range(epoch):
        shuffle_indices = np.random.permutation(np.arange(len(y_train)))
        x_train_shuffle = x_train[shuffle_indices]
        x_train_input_shuffle = x_train_len[shuffle_indices]
        y_train_shuffle = y_train[shuffle_indices]
        u_train_shuffle = u_train[shuffle_indices]
        u_train_count_shuffle = u_train_count[shuffle_indices]
        p_train_shuffle = p_train[shuffle_indices]
        p_train_count_shuffle = p_train_count[shuffle_indices]
        
        train_loss = []
        for j in range(0, len(x_train_shuffle), batch_size):
            x_batch = x_train_shuffle[j:j+batch_size]
            if len(x_batch) != batch_size:
                continue
            x_batch_input = x_train_input_shuffle[j:j+batch_size]
            y_batch = y_train_shuffle[j:j+batch_size]
            u_batch = u_train_shuffle[j:j+batch_size]
            u_batch_count = u_train_count_shuffle[j:j+batch_size]
            p_batch = p_train_shuffle[j:j+batch_size]
            p_batch_count = p_train_count_shuffle[j:j+batch_size]
            
            loss, updates = model.step(sess, x_batch, y_batch, u_batch, p_batch, \
                                 x_batch_input, u_batch_count, p_batch_count)
            train_loss.append(loss)
            
            step += batch_size
            if False and step > step_check:
                test_count = 0
                test_tot = 0
                preds = []
                rmse = 0
                for jj in range(0, len(x_test), batch_size):
                    x_batch = x_test[jj:jj+batch_size]
                    x_batch_input = x_test_len[jj:jj+batch_size]
                    y_batch = y_test[jj:jj+batch_size]
                    u_batch = u_test[jj:jj+batch_size]
                    u_batch_count = u_test_count[jj:jj+batch_size]
                    p_batch = p_test[jj:jj+batch_size]
                    p_batch_count = p_test_count[jj:jj+batch_size]
                    k = len(x_batch)
                    if len(x_batch) != batch_size:
                        x_batch = np.concatenate((x_batch, x_batch[:-k+batch_size]), axis=0)
                        x_batch_input = np.concatenate((x_batch_input, x_batch_input[:-k+batch_size]), axis=0)
                        y_batch = np.concatenate((y_batch, y_batch[:-k+batch_size]), axis=0)
                        u_batch = np.concatenate((u_batch, u_batch[:-k+batch_size]), axis=0)
                        u_batch_count = np.concatenate((u_batch_count, u_batch_count[:-k+batch_size]), axis=0)
                        p_batch = np.concatenate((p_batch, p_batch[:-k+batch_size]), axis=0)
                        p_batch_count = np.concatenate((p_batch_count, p_batch_count[:-k+batch_size]), axis=0)
                    
                    predictions, count, k_u, lam_u, k_p, lam_p, \
                        a_ud, a_us, a_pd, a_ps, gu, gp, gup = model.step(sess, x_batch, y_batch, u_batch, p_batch, \
                                                    x_batch_input, u_batch_count, p_batch_count, \
                                                    training=False)
                    test_count += np.sum(count[:k])
                    test_tot += k
                    preds.append(predictions.flatten()[:k])
                    rmse += np.sum(np.power(predictions.flatten()[:k]-np.argmax(y_batch, 1).flatten()[:k], 2))
                
                test_acc = test_count / test_tot
                rmse = np.sqrt(rmse / test_tot)
                if best_acc < test_acc:
                    print('saving')
                    best_acc = test_acc
                    preds = np.array(preds).flatten()
                    f = open(data_dir + '/pred_model' + train_type + '_' + base_model + '.txt', 'w', encoding='utf-8', errors='ignore')
                    for p in zip(preds, a_ud, a_us:
                        for pp in p:
                            f.write(str(pp) + '\n')
                    f.close()
                    model.saver.save(sess, data_dir + '/model' + train_type + '_' + base_model)
                
                print('epoch', i, 'instance', j, 'train loss', np.mean(train_loss), 'test acc', test_acc, 'rmse', rmse, 'time', time.time()-cur_time)
                
                step -= step_check
                cur_time = time.time()
                train_loss = []
