# -*- coding: utf-8 -*-
# @Author: Jie Yang
# @Date:   2017-12-04 23:19:38
# @Last Modified by:   Jie Yang,     Contact: jieynlp@gmail.com
# @Last Modified time: 2018-12-16 22:15:56
from __future__ import print_function
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
from torch_struct import LinearChainCRF
from torch_struct.semirings import LogSemiring
START_TAG = -2
STOP_TAG = -1
import torch

class CRF(nn.Module):

    def __init__(self, tagset_size, gpu):
        super(CRF, self).__init__()
        print("build CRF...")
        self.gpu = gpu
        # Matrix of transition parameters.  Entry i,j is the score of transitioning from i to j.
        self.tagset_size = tagset_size
        # # We add 2 here, because of START_TAG and STOP_TAG
        # # transitions (f_tag_size, t_tag_size), transition value from f_tag to t_tag
        init_transitions = torch.zeros(self.tagset_size+2, self.tagset_size+2)
        init_transitions[:,START_TAG] = -10000.0
        init_transitions[STOP_TAG,:] = -10000.0
        init_transitions[:,0] = -10000.0
        init_transitions[0,:] = -10000.0
        if self.gpu:
            init_transitions = init_transitions.cuda()
        self.transitions = nn.Parameter(init_transitions)

        # self.transitions = nn.Parameter(torch.Tensor(self.tagset_size+2, self.tagset_size+2))
        # self.transitions.data.zero_()
    def _viterbi_decode(self, feats, mask):
        """
            input:
                feats: (batch, seq_len, self.tag_size+2)
                mask: (batch, seq_len)
            output:
                decode_idx: (batch, seq_len) decoded sequence
                path_score: (batch, 1) corresponding score for each sequence (to be implementated)
        """
        batch_size = feats.size(0)
        tag_size = feats.size(2)
        assert(tag_size == self.tagset_size+2)
        ## calculate sentence length for each sentence
        length_mask = torch.sum(mask.long(), dim = 1).long()
        maxlen = mask.size(1)
        decoded_tag = torch.zeros(batch_size, maxlen).type(torch.long)
        log_potentials = self.log_potentials(feats, mask)
        dist = LinearChainCRF(log_potentials, lengths=length_mask+2)
        argmax = dist.argmax
        for i in range(batch_size):
            tmp = argmax[i].nonzero()[:-1,1]
            assert(tmp.size(0)==length_mask[i])
            decoded_tag[i,0:length_mask[i]]=tmp
        return None, decoded_tag

    def forward(self, feats):
        path_score, best_path = self._viterbi_decode(feats)
        return path_score, best_path

    def log_potentials(self, feats, mask):
        batch_size = feats.size(0)
        maxlen = mask.size(1)
        length_mask = torch.sum(mask.long(), dim=1).long() + 2
        log_potentials = self.transitions.repeat(batch_size, maxlen+1, 1, 1)
        log_potentials[:, 1::, :, :]+=feats.view(batch_size, maxlen, self.tagset_size + 2, 1)
        log_potentials[:, 0, tuple(i for i in range(self.tagset_size+2) if i != self.tagset_size + 2 + START_TAG), :] -= 1000
        log_potentials[torch.arange(batch_size),length_mask-2,:,0:-1] -= 1000
        log_potentials = log_potentials.transpose(2, 3).contiguous()
        return log_potentials
            
    def neg_log_likelihood_loss(self, feats, mask, tags):
        # nonegative log likelihood
        batch_size = feats.size(0)
        maxlen = mask.size(1)

        z = torch.zeros((batch_size, maxlen+1)+self.transitions.shape)

        for b in range(batch_size):
            for i in range(maxlen+1):
                if i==0:
                    cur = self.tagset_size 
                    nxt = tags[b,i]
                    if nxt == 0:
                        break
                elif i>=1 and tags[b,i-1]==0:
                    break
                elif i==maxlen or tags[b,i]==0:
                    cur = tags[b,i-1]
                    nxt = self.tagset_size + 1
                else:
                    cur = tags[b,i-1]
                    nxt = tags[b,i]
                z[b,i,cur,nxt]=1
                if self.transitions[cur,nxt]<-1000:
                    raise ("wrong",cur,nxt)
        z = z.transpose(2,3).contiguous()

        log_potentials = self.log_potentials(feats, mask)
        length_mask = torch.sum(mask.long(), dim=1).long() + 2
        dist = LinearChainCRF(log_potentials, length_mask)
        return -dist.log_prob(z).sum()



    def _viterbi_decode_nbest(self, feats, mask, nbest):
        """
            input:
                feats: (batch, seq_len, self.tag_size+2)
                mask: (batch, seq_len)
            output:
                decode_idx: (batch, nbest, seq_len) decoded sequence
                path_score: (batch, nbest) corresponding score for each sequence (to be implementated)
                nbest decode for sentence with one token is not well supported, to be optimized
        """

        batch_size = feats.size(0)
        tag_size = feats.size(2)
        assert(tag_size == self.tagset_size+2)
        ## calculate sentence length for each sentence
        length_mask = torch.sum(mask.long(), dim = 1).long()
        maxlen = mask.size(1)
        decoded_tag = torch.zeros(batch_size, maxlen).type(torch.long)
        log_potentials = self.log_potentials(feats, mask)
        dist = LinearChainCRF(log_potentials, lengths=length_mask+2)
        topk = dist.topk(nbest)
        for j in range(nbest):
            for i in range(batch_size):
                tmp = topk[j][i].nonzero()[:-1,1]
                assert(tmp.size(0)==length_mask[i])
                decoded_tag[i,0:length_mask[i]]=tmp
        return None, decoded_tag


















