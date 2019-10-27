import contextlib
import os, math, time
from random import shuffle as list_shuffle

import torch
import torch.nn as nn
from torch.utils.data.sampler import Sampler
import numpy as np
from fairseq.utils import get_len
from scipy.sparse import *


class AssistantIterator(object):

    def __init__(self, iterable, length, indices):
        self.itr = iterable
        self._sharded_len = length
        self.indices = indices

    def __len__(self):
        return self._sharded_len

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.itr)

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0) # only difference

def sigmoid(x):
    """Compute softmax values for each sets of scores in x."""
    return 1.0 / (1+ np.exp(-x))


class AssistantSamplerParallel(Sampler):
    r""" Generate instances based on Assistant model.
    Arguments:
        dic_src (Dictionary): dictionary for the source language
        dic_tgt (Dictionary): dictionary for the target language
        base_prob (float): ground probability for an instance to pass
        num_proc (int): number of assistant processes
        proc_id (int): the current process id 
        num_bins_per_proc (int): number of data bins in a single worker
        tfidf_feature (dict): TFIDF feature matrices
    """

    def __init__(self, dic_src, dic_tgt, base_prob = 0.3, num_proc = 8, proc_id = -1, num_bins_per_proc = 24,  tfidf_feature=None):
        self.base_prob = 1.0 # For first epoch, all instances are accepted
        self.real_base_prob = base_prob

        self.use_tfidf = tfidf_feature is not None
 
        if self.use_tfidf:
            print("Using TF-IDF version of Assistant")
            self.assistant = AssistantModelBinaryTfIdf( dic_src = dic_src, dic_tgt = dic_tgt, tfidf_feature = tfidf_feature)
        else:
            self.assistant = AssistantModelBinary( dic_src = dic_src, dic_tgt = dic_tgt)

        self.total_samples = 1
        self.total_success = 1

        self.sec_loss = 10.
        self.confident = 0

        self.epoch = 0
        self.shuffle = False

        self.num_proc = num_proc
        self.proc_rank = proc_id
        self.num_bins_per_proc = num_bins_per_proc

        self.token_num_batches = 0
        self.local_indices = None

        self.global_token_sum = -1
        self.local_token_sum = -1

        self.loss_bar = 0
        self.loss_epoch_sum = 0
        self.inst_epoch_sum = 0

    def mean_loss(self):
        return self.loss_bar

    def associate_data(self, data_source, all_indices):
        """
        Feed training feature into Assistant
        Arguments:
            data_source: if not use_tfidf, tokenized text data is used as feature,
                         otherwise the pre-constructed TF/IDF features
            all_indices: all the data indices
        """
        self.data_source = data_source
        self.all_indices = list(all_indices)
        self.len_idx = len(self.all_indices) #3961179

    def get_x_y(self, idx):
        ret = []
        for i in idx:
            ret.append(self.data_source[i])
        return ret
        
    def compute_iteration_length(self, max_sentences, max_tokens, num_tokens_fn):
        num_batches = [0 for x in range(self.num_proc)]
        num_tokens = [0 for x in range(self.num_proc)]
        for i in range(self.num_proc):
            cur_indices = self.local_indices
            for idx in cur_indices:
                num_tokens[i] += num_tokens_fn(idx)
            num_batch_tokens = math.ceil(num_tokens[i] / max_tokens)
            num_batch_sentences = math.ceil(len(cur_indices) / max_sentences)
            num_batches[i] = max(num_batch_tokens, num_batch_sentences)
        max_num_batch = max(num_batches)
        return max_num_batch, num_tokens[self.proc_rank]

    def batch_by_size(
        self, num_tokens_fn, max_tokens=None, max_sentences=None,
        required_batch_size_multiple=1, shard_num = 1, shard_id = 0, batch="sentences", shuffle = True,
    ):
        assert( shard_id == self.proc_rank, "Proc rank not same as shard_id!")
        self.proc_rank = shard_id
        max_sentences = max_sentences if max_sentences is not None else 10000
        self.batch_method = batch

        self.shuffle = shuffle

        if self.epoch ==0 :
            self.global_token_sum = sum([num_tokens_fn(idx) for idx in self.all_indices])
            if batch=="bins":
                # divide into bins
                avr_sentence_len = self.global_token_sum / len(self.all_indices)
                bin_size = math.ceil(self.len_idx / self.num_proc ) // self.num_bins_per_proc
                self.global_bins = np.array([self.all_indices[i * bin_size:(i + 1) * bin_size] for i in range((len(self.all_indices) + bin_size - 1) // bin_size )])
                self.global_bin_idcs = list(range(self.global_bins.shape[0]))
                print("Divided all indices into %d bins"%(self.global_bins.shape[0]))
                local_bins = self.global_bins[shard_id:len(self.global_bins):shard_num]
                print("Assistant %d assigned bins:"%(shard_id), self.global_bin_idcs[shard_id:len(self.global_bins):shard_num],  flush=True)
                self.local_indices = []
                for bb in local_bins:
                    self.local_indices += bb
            elif batch=="sentences":
                self.local_indices = self.all_indices[shard_id:self.len_idx:shard_num]


        num_batches, self.my_token_sum = self.compute_iteration_length(max_sentences, max_tokens, num_tokens_fn)


        batch_sampler = self._batch_generator(
        num_tokens_fn, max_tokens=max_tokens, max_sentences=max_sentences,
        required_batch_size_multiple=required_batch_size_multiple, indices = self.local_indices)

        print("Setting Assistant %d: num_batches=%d, num_tokens=%d, samples=%d/%d, confident=%f"%(shard_id,
            num_batches, self.my_token_sum,  self.total_success, self.total_samples, self.confident),  flush= True)
        return AssistantIterator( batch_sampler, num_batches, self.all_indices)


    def _batch_generator(
        self, num_tokens_fn, max_tokens=None, max_sentences=None,
        required_batch_size_multiple=1, indices = None,
    ):
        """
        Yield mini-batches of indices bucketed by size. Batches may contain
        sequences of different lengths.

        Args:
            num_tokens_fn (callable): function that returns the number of tokens at
                a given index
            max_tokens (int, optional): max number of tokens in each batch.
                Default: ``None``
            max_sentences (int, optional): max number of sentences in each
                batch. Default: ``None``
            required_batch_size_multiple (int, optional): require batch size to
                be a multiple of N. Default: ``1``
        """
        max_tokens = max_tokens if max_tokens is not None else float('Inf')
        max_sentences = max_sentences if max_sentences is not None else float('Inf')
        bsz_mult = required_batch_size_multiple
        num_batches = 0


        def is_batch_full(num_tokens):
            if len(batch) == 0:
                return False
            if len(batch) == max_sentences:
                return True
            if num_tokens > max_tokens:
                return True
            return False
        while True:
            batch = []
            sample_len = 0
            sample_lens = []
            
            if self.shuffle and self.epoch > 0:
                if self.batch_method =='bins':
                    indices = self.shuffle_bin_indices()
                else:
                    self.shuffle_local_indices()
            
            if self.epoch > 0:
                self.loss_bar = self.loss_epoch_sum / self.inst_epoch_sum
                self.loss_epoch_sum = 0
                self.inst_epoch_sum = 0
            for idx in indices:
                accept = self.accept( idx)
                if accept:
                    sample_lens.append(num_tokens_fn(idx))
                    sample_len = max(sample_len, sample_lens[-1])
                    num_tokens = (len(batch) + 1) * sample_len
                    if is_batch_full(num_tokens):
                        mod_len = max(
                            bsz_mult * (len(batch) // bsz_mult),
                            len(batch) % bsz_mult,
                        )
                        yield batch[:mod_len]
                        num_batches += 1
                        batch = batch[mod_len:]
                        sample_lens = sample_lens[mod_len:]
                        sample_len = max(sample_lens) if len(sample_lens) > 0 else 0

                    batch.append(idx)

            if len(batch) > 0:
                yield batch
                num_batches += 1
            self.epoch += 1
            self.base_prob = self.real_base_prob # after first epoch, accept with real_base_prob
            print("Assistant %d start over,  num_batches=%d, num_tokens=%d, samples=%d/%d, confident=%f"%(self.proc_rank,
            num_batches, self.my_token_sum,  self.total_success, self.total_samples, self.confident),  flush= True)

    def accept(self, index):
        base_prob = max(self.base_prob, (1.0 - self.confident))

        coin = np.random.uniform()
        self.total_samples += 1 # not thread safe!!


        if coin < base_prob:
            self.total_success += 1 # not thread safe!!
            return True
        else:
            #continue
            if not self.use_tfidf:
                cur_data = self.data_source[index]
                x = cur_data['source']
                y = cur_data['target']
            else:
                x = self.assistant.tfidf_feature['source'][index]
                y = self.assistant.tfidf_feature['target'][index]
            coin = (coin - base_prob) / (1.0 - base_prob) # renormalize coin, still independent variable

            # compute importance
            keep_prob = self.assistant.get_importance(x, y)
            if coin < keep_prob:
                self.total_success += 1 # not thread safe!!
                return True
        return False

    def __len__(self):
        return len(self.data_source)

    def rate(self):
        return float(self.total_success)/self.total_samples

    def loss(self):
        return self.confident

    def shuffle_bin_indices(self):
        np.random.seed(self.epoch + 100)
        np.random.shuffle(self.global_bin_idcs)
        local_bin_idcs = self.global_bin_idcs[self.proc_rank : len(self.global_bin_idcs) : self.num_proc]
        for b_idx in local_bin_idcs:
            np.random.shuffle(self.global_bins[b_idx])
        return np.concatenate(self.global_bins[local_bin_idcs], axis=0)


    def shuffle_local_indices(self):
        np.random.seed(self.epoch + 100)
        np.random.shuffle(self.local_indices)

    def train_tfidf_step(self, idcs, losses, n_steps = 1):
        batch_size = int(np.ceil(len(idcs) / n_steps))
        # losses are un-token-normalized losses, neet to normalize by number of tokens
        X = [ self.data_source[i]['source'].numpy() for i in idcs]
        Y = [ self.data_source[i]['target'].numpy() for i in idcs]
        y_len = np.array([len(yy) for yy in Y])
        norm_losses = np.divide(losses, y_len)

        self.loss_epoch_sum += norm_losses.sum()
        self.inst_epoch_sum += len(idcs)
        if self.epoch==0:# at epcoch 0, estimate mean loss with running average
            self.loss_bar = 0.99 * self.loss_bar + 0.01 * norm_losses.mean()

        sec_loss = []
        keep_probs = []
        pos_cnt = 0; pred_pos_cnt = 0
        for i in range(0, len(idcs), batch_size):
            cur_idcs = idcs[i:i+batch_size]
            cur_losses = norm_losses[i:i+batch_size]
            cur_sec_loss, cur_keep_probs, cur_real_pos, cur_pred_pos = self.assistant.train_step(cur_idcs, X[i:i+batch_size], Y[i:i+batch_size], \
                    norm_losses[i:i+batch_size], self.epoch, self.loss_bar)
            
            cur_batch_size =len(cur_idcs)
            sec_loss.append(cur_sec_loss * cur_batch_size)
            keep_probs.extend(cur_keep_probs)
            pos_cnt += cur_real_pos * cur_batch_size
            pred_pos_cnt += cur_pred_pos * cur_batch_size

        self.real_pos = pos_cnt / len(X)
        self.pred_pos = pred_pos_cnt / len(X)

        self.sec_loss = np.array(sec_loss).sum() / len(X)

        self.confident *= 0.95
        self.confident += self.sec_loss * 0.05
        return keep_probs

    def train_step(self, idcs, X, Y, losses, n_steps = 1):
        batch_size = int(np.ceil(len(X) / n_steps))

        def get_len(XX, PAD_idx):
            return np.array([ len(inst) - (inst==PAD_idx).sum() for inst in XX])
        
        # losses are un-token-normalized losses, neet to normalize by number of tokens
        y_len = get_len(Y, self.assistant.PAD_tgt)
        norm_losses = np.divide(losses, y_len)

        self.loss_epoch_sum += norm_losses.sum()
        self.inst_epoch_sum += len(X)
        if self.epoch==0:# at epcoch 0, estimate mean loss with running average
            self.loss_bar = 0.99 * self.loss_bar + 0.01 * norm_losses.mean()

        sec_loss = []
        keep_probs = []
        pos_cnt = 0; pred_pos_cnt = 0
        for i in range(0, len(X), batch_size):
            cur_sec_loss, cur_keep_probs, cur_real_pos, cur_pred_pos = self.assistant.train_step(idcs[i:i+batch_size], X[i:i+ batch_size],Y[i:i+batch_size], norm_losses[i:i+batch_size], self.epoch, self.loss_bar)
            cur_batch_size =len(X[i:i+ batch_size])
            sec_loss.append(cur_sec_loss * cur_batch_size)
            keep_probs.extend(cur_keep_probs)
            pos_cnt += cur_real_pos * cur_batch_size
            pred_pos_cnt += cur_pred_pos * cur_batch_size

        self.real_pos = pos_cnt / len(X)
        self.pred_pos = pred_pos_cnt / len(X)

        self.sec_loss = np.array(sec_loss).sum() / len(X)

        self.confident *= 0.95
        self.confident += self.sec_loss * 0.05
        return keep_probs


class AssistantModelBinary(nn.Module):
    """
    predict p( not_trivial | x_i,  y_i) = sigmoid( W*x_i + U[y_i] )
        where:
            not_trivial = ( loss_i > loss_mean - loss_stddev)
    Arguments:
        dic_src (Dictionary): dictionary for the source language
        dic_tgt (Dictionary): dictionary for the target language
    """
    def __init__(self, dic_src, dic_tgt):
        super(AssistantModelBinary, self).__init__()
        self.dim_src = len(dic_src)
        self.dim_tgt = len(dic_tgt)

        self.PAD_src = dic_src.pad()
        self.PAD_tgt = dic_tgt.pad()

        self.lr = 1
        self.lam = 1e-3
        self.fitted = 0

        self.W = 0.001 * np.random.randn( self.dim_src)
        self.U = 0.001 * np.random.randn( self.dim_tgt)
        #self.W = 0.001 * torch.randn( self.dim_src)
        #self.U = 0.001 * torch.randn( self.dim_tgt)

        self.W[self.PAD_src] = 0
        self.U[self.PAD_tgt] = 0

        self.b = 0.0

        self.loss_sum = 0
        self.num_instances = 1

    def get_importance(self, x, y):
        return sigmoid( self.W[x].sum() + self.U[y].sum() + self.b )

    def make_target(self, loss, epoch, mean):
        return np.array( loss < mean, dtype = int)

    def train_step(self, idcs, X, Y, loss, epoch, loss_bar):
        self.fitted += 1
        batch_size = Y.shape[0]
        lr = self.lr / Y.shape[0]
        label = self.make_target(loss, epoch, loss_bar)

        def compute(XX, W):
            return W[XX.reshape(-1)].reshape(XX.shape).sum(1)

        prob = sigmoid( compute(X, self.W) + compute(Y, self.U) + self.b)

        sec_pred = np.array( prob > 0.5, dtype=int)
        acc = np.sum(label == sec_pred) * 1.0
        predict_pos_rate = np.sum(sec_pred) / batch_size
        real_pos_rate = np.sum(label) / batch_size

        grad = (prob - label)

        # gradient update
        self.b    -= lr * ( grad.sum(0) + self.lam * self.b)
        self.W    -= lr * self.lam * self.W
        self.U    -= lr * self.lam * self.U
        def update(XX, Grad, W):
            for i in range(XX.shape[0]):
                for j in range(XX.shape[1]):
                    W[XX[i,j]] -= lr * Grad[i]

        update(X, grad, self.W)
        update(Y, grad, self.U)

        self.W[self.PAD_src] = 0
        self.U[self.PAD_tgt] = 0
        return acc / batch_size, prob, real_pos_rate, predict_pos_rate

class AssistantModelBinaryTfIdf(nn.Module):
    """
    predict p( not_trivial | x_i,  y_i) = sigmoid( W*x_i + U[y_i] )
        where:
            not_trivial = ( loss_i > loss_mean - loss_stddev)
    Arguments:
        dic_src (Dictionary): dictionary for the source language
        dic_tgt (Dictionary): dictionary for the target language
        tfidf_feature (Dictionary {'source': scipy.csr_matrix, 'target': scipy.csr_matrix}) TFIDF features of training data
    """
    def __init__(self, dic_src, dic_tgt, tfidf_feature = None):
        super(AssistantModelBinaryTfIdf, self).__init__()
        self.dim_src = len(dic_src)
        self.dim_tgt = len(dic_tgt)
        self.tfidf_feature = tfidf_feature

        self.xy_lengths = {'source':np.array(tfidf_feature['source'].getnnz(axis=1)), 'target':np.array(tfidf_feature['target'].getnnz(axis=1))}


        self.PAD_src = dic_src.pad()
        self.PAD_tgt = dic_tgt.pad()

        self.lr = 0.5
        self.lam = 1e-1
        self.fitted = 0
        

        self.W_tf = 0.0001 * np.random.randn( self.dim_src)
        self.U_tf = 0.0001 * np.random.randn( self.dim_tgt)
        self.W_tfidf = 0.0001 * np.random.randn( self.dim_src)
        self.U_tfidf = 0.0001 * np.random.randn( self.dim_tgt)
        
        self.zero_pad_weights()

        self.b = 0.0

        self.c_len_x = 0.0001
        self.c_len_y = 0.0001
        self.max_sen_len = 10

        self.loss_sum = 0
        self.num_instances = 1
    
    def zero_pad_weights(self): 
        self.W_tf[self.PAD_src] = 0
        self.U_tf[self.PAD_tgt] = 0
        self.W_tfidf[self.PAD_src] = 0
        self.U_tfidf[self.PAD_tgt] = 0

    def get_importance(self, x, y):
        """
        Compute the importance weight of given instance
        Arguments:
            x: scipy.sparse.csr_matrix 
            y: scipy.sparse.csr_matrix
        """
        linear_tf = self.W_tf[x.indices].sum() / x.getnnz() + self.U_tf[y.indices].sum() / y.getnnz()
        linear_tfidf = csr_matrix.dot(x, self.W_tfidf) + csr_matrix.dot(y, self.U_tfidf)
        return sigmoid( linear_tfidf + linear_tf + self.b + self.c_len_x * x.getnnz()/self.max_sen_len + self.c_len_y * y.getnnz()/self.max_sen_len)

    def make_target(self, loss, epoch, mean):
        return np.array( loss > mean, dtype = int)

    def train_step(self, idcs, X, Y, loss, epoch, loss_bar):
        self.fitted += 1
        batch_size = len(idcs)
        lr = self.lr / batch_size
        label = self.make_target(loss, epoch, loss_bar)
        X_tfidf = self.tfidf_feature['source'][idcs]
        Y_tfidf = self.tfidf_feature['target'][idcs]
        x_lengths = self.xy_lengths['source'][idcs]
        y_lengths = self.xy_lengths['target'][idcs]

        def compute_tf_linear(W, X, X_len):
            return np.array([ W[x.indices].sum() / xlen for x, xlen in zip(X, X_len)])

        tf_linear =  compute_tf_linear(self.W_tf, X_tfidf, x_lengths) + compute_tf_linear(self.U_tf, Y_tfidf, y_lengths)
        tfidf_linear =  csr_matrix.dot(X_tfidf, self.W_tfidf) + csr_matrix.dot(Y_tfidf, self.U_tfidf)
        prob = sigmoid( tfidf_linear + tf_linear + self.b + self.c_len_x * x_lengths / self.max_sen_len + self.c_len_y * y_lengths / self.max_sen_len)

        sec_pred = np.array( prob > 0.5, dtype=int)
        acc = np.sum(label == sec_pred) * 1.0
        predict_pos_rate = np.sum(sec_pred) / batch_size
        real_pos_rate = np.sum(label) / batch_size

        grad = (prob - label)

        # gradient update
        self.b    -= lr * ( grad.sum(0) + self.lam * self.b)
        self.c_len_x    -= lr * ( np.dot(grad, x_lengths) / self.max_sen_len + self.lam * self.c_len_x)
        self.c_len_y    -= lr * ( np.dot(grad, y_lengths) / self.max_sen_len + self.lam * self.c_len_y)
        self.W_tfidf    -= lr * self.lam * self.W_tfidf
        self.U_tfidf    -= lr * self.lam * self.U_tfidf
        self.W_tf    -= lr * self.lam * self.W_tf
        self.U_tf    -= lr * self.lam * self.U_tf

        
        def update_W_tf(idcs, XX, X_tfidf,  Grad, W):
            for i in range(len(XX)):
                W[X_tfidf[i].indices] -= lr * Grad[i] / X_tfidf[i].getnnz()

        def update_W_tfidf(idcs, XX, X_tfidf,  Grad, W):
            for i in range(len(XX)):
                for j in range(len(XX[i])):
                    W[XX[i][j]] -= lr * Grad[i] * X_tfidf[ i, XX[i][j]]

        update_W_tf(idcs, X, X_tfidf, grad, self.W_tf)
        update_W_tf(idcs, Y, Y_tfidf, grad, self.U_tf)
        update_W_tfidf(idcs, X, X_tfidf, grad, self.W_tfidf)
        update_W_tfidf(idcs, Y, Y_tfidf, grad, self.U_tfidf)

        self.zero_pad_weights()

        return acc / batch_size, prob, real_pos_rate, predict_pos_rate

