#!/usr/bin/env python3

# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

from parlai.agents.memnn.modules import MemNN, opt_to_kwargs as opt_to_memnn_args
from parlai.agents.seq2seq.modules import Seq2seq, RNNDecoder, OutputLayer, opt_to_kwargs as opt_to_seq2seq_args

import math

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import torch.nn.functional as F

from parlai.core.utils import NEAR_INF


class Ctx2seq(Seq2seq):
    def __init__(self, opt, num_features, embeddingsize, hiddensize):
        """Initialize seq2seq model.

        See cmdline args in Seq2seqAgent for description of arguments.
        """
        s2s_args = opt_to_seq2seq_args(opt)
        super().__init__(num_features, embeddingsize, hiddensize, **s2s_args)
        mnn_args = opt_to_memnn_args(opt)
        self.memory_encoder = MemNNEncoder(num_features, embeddingsize, **mnn_args)
        self.reduce = nn.Linear(hiddensize, embeddingsize)
        self.enlarge = nn.Linear(embeddingsize, hiddensize)

    def transform(self, hidden, memories):
        """Transofrm encoder hidden state with memory attention."""
        cell = None
        if isinstance(hidden, tuple):
            hidden, cell = hidden

        if memories is not None:
            small = self.reduce(hidden)
            new_state = self.memory_encoder(small, memories)
        else:
            new_state = hidden
        full = self.enlarge(new_state)

        if cell is not None:
            full = (full, cell)
        # return full + hidden
        return full

    def forward(self, xs, ys=None, cands=None, prev_enc=None, maxlen=None,
                seq_len=None, memories=None):
        """Get output predictions from the model.

        :param xs:          (bsz x seqlen) LongTensor input to the encoder
        :param ys:          expected output from the decoder. used for teacher
                            forcing to calculate loss.
        :param cands:       set of candidates to rank
        :param prev_enc:    if you know you'll pass in the same xs multiple
                            times, you can pass in the encoder output from the
                            last forward pass to skip recalcuating the same
                            encoder output.
        :param maxlen:      max number of tokens to decode. if not set, will
                            use the length of the longest label this model
                            has seen. ignored when ys is not None.
        :param seq_len      this is the sequence length of the input (xs), i.e.
                            xs.size(1). we use this to recover the proper
                            output sizes in the case when we distribute over
                            multiple gpus
        :param memories:    (bsz x num_mems x seqlen) LongTensor memories

        :returns: scores, candidate scores, and encoder states
            scores contains the model's predicted token scores.
                (bsz x seqlen x num_features)
            candidate scores are the score the model assigned to each candidate
                (bsz x num_cands)
            encoder states are the (output, hidden, attn_mask) states from the
                encoder. feed this back in to skip encoding on the next call.
        """
        if ys is not None:
            # keep track of longest label we've ever seen
            # we'll never produce longer ones than that during prediction
            self.longest_label = max(self.longest_label, ys.size(1))

        encoder_output, hidden, attn_mask = self._encode(xs, prev_enc)
        new_hidden = self.transform(hidden, memories)
        encoder_states = encoder_output, new_hidden, attn_mask

        # rank candidates if they are available
        cand_scores = None
        if cands is not None:
            cand_inds = [i for i in range(cands.size(0))]
            cand_scores = self._rank(cands, cand_inds, encoder_states)

        if ys is not None:
            # use teacher forcing
            scores = self._decode_forced(ys, encoder_states)
        else:
            scores = self._decode(encoder_states, maxlen or self.longest_label)

        return scores, cand_scores, encoder_states


class MemNNEncoder(MemNN):
    def forward(self, xs, mems, cands=None):
        """One forward step.

        :param xs:    (bsz x esz) FloatTensor encoded queries to the model
        :param mems:  (bsz x num_mems x seqlen) LongTensor memories

        :returns: scores
            scores contains the model's similarity scores
                (bsz x esz) FloatTensor
        """
        state = xs
        # START COPY FROM MEMNN
        if mems is not None:
            # no memories available, `nomemnn` mode just uses query/ans embs
            in_memory_embs = self.in_memory_lt(mems).transpose(1, 2)
            out_memory_embs = self.out_memory_lt(mems)

            for _ in range(self.hops):
                state = self.memory_hop(state, in_memory_embs, out_memory_embs)
        # END COPY FROM MEMNN
        scores = state
        return scores
