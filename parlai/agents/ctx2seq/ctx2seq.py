#!/usr/bin/env python3

# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

from parlai.agents.memnn.memnn import MemnnAgent
from parlai.agents.seq2seq.seq2seq import Seq2seqAgent
from parlai.core.torch_agent import TorchAgent, Output, Beam, Batch
from parlai.core.utils import padded_tensor, round_sigfigs
from parlai.core.thread_utils import SharedTable
from .modules import Ctx2seq

import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import defaultdict

import os
import math
import json
import tempfile


class Ctx2seqAgent(TorchAgent):
    @classmethod
    def add_cmdline_args(cls, argparser):
        MemnnAgent.add_cmdline_args(argparser)
        Seq2seqAgent.add_cmdline_args(argparser)
        TorchAgent.add_cmdline_args(argparser)

    def __init__(self, opt, shared=None):
        model_file, opt = self._get_model_file(opt)
        super().__init__(opt, shared)

        # all instances may need some params
        self.id = 'Ctx2Seq'

        if shared:
            self.model = shared['model']
            self.metrics = shared['metrics']
        else:
            self.metrics = {'loss': 0.0, 'num_tokens': 0, 'correct_tokens': 0}
            self.build_model()
            if model_file:
                print('Loading existing model parameters from ' + model_file)
                self.load(model_file)

        # set up criteria
        if opt.get('numsoftmax', 1) > 1:
            self.criterion = nn.NLLLoss(
                ignore_index=self.NULL_IDX, size_average=False)
        else:
            self.criterion = nn.CrossEntropyLoss(
                ignore_index=self.NULL_IDX, size_average=False)

        if self.use_cuda:
            self.model.cuda()
            self.criterion.cuda()

        optim_params = [p for p in self.model.parameters() if p.requires_grad]
        self.init_optim(optim_params)

        self.reset()

    def build_model(self):
        opt = self.opt
        self.model = Ctx2seq(opt, len(self.dict), opt['embeddingsize'],
                             opt['hiddensize'])

    def zero_grad(self):
        """Zero out optimizer."""
        self.optimizer.zero_grad()

    def update_params(self):
        """Do one optimization step."""
        if self.clip > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()

    def reset_metrics(self):
        """Reset metrics for reporting loss and perplexity."""
        super().reset_metrics()
        self.metrics['loss'] = 0.0
        self.metrics['num_tokens'] = 0
        self.metrics['correct_tokens'] = 0

    def report(self):
        """Report loss and perplexity from model's perspective.

        Note that this includes predicting __END__ and __UNK__ tokens and may
        differ from a truly independent measurement.
        """
        m = {}
        num_tok = self.metrics['num_tokens']
        if num_tok > 0:
            if self.metrics['correct_tokens'] > 0:
                m['token_acc'] = self.metrics['correct_tokens'] / num_tok
            m['loss'] = self.metrics['loss'] / num_tok
            try:
                m['ppl'] = math.exp(m['loss'])
            except OverflowError:
                m['ppl'] = float('inf')
        for k, v in m.items():
            # clean up: rounds to sigfigs and converts tensors to floats
            m[k] = round_sigfigs(v, 4)
        return m

    def share(self):
        """Share internal states between parent and child instances."""
        shared = super().share()
        shared['model'] = self.model
        if self.opt.get('numthreads', 1) > 1:
            # we're doing hogwild so share the model too
            if isinstance(self.metrics, dict):
                # move metrics and model to shared memory
                self.metrics = SharedTable(self.metrics)
                self.model.share_memory()
        shared['metrics'] = self.metrics  # do after numthreads check
        return shared

    def vectorize(self, *args, **kwargs):
        """Override vectorize for seq2seq."""
        kwargs['add_start'] = False  # model does this in module code
        kwargs['add_end'] = True  # we do want this
        kwargs['split_lines'] = True  # we do want this
        return super().vectorize(*args, **kwargs)

    def batchify(self, *args, **kwargs):
        """Override batchify options for seq2seq."""
        kwargs['sort'] = True  # need sorted for pack_padded
        return super().batchify(*args, **kwargs)

    def _init_cuda_buffer(self, model, criterion, batchsize, maxlen):
        """Pre-initialize CUDA buffer by doing fake forward pass."""
        if self.use_cuda and not hasattr(self, 'buffer_initialized'):
            try:
                print('preinitializing pytorch cuda buffer')
                dummy_xs = torch.ones(batchsize, maxlen * 2 - 2).long().cuda()
                dummy_ys = torch.ones(batchsize, 2).long().cuda()
                out = model(dummy_xs, dummy_ys)
                sc = out[0]  # scores
                loss = criterion(sc.view(-1, sc.size(-1)), dummy_ys.view(-1))
                loss.backward()
                self.buffer_initialized = True
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    m = ('CUDA OOM: Lower batch size (-bs) from {} or lower '
                         ' max sequence length (-tr) from {}'
                         ''.format(batchsize, maxlen))
                    raise RuntimeError(m)
                else:
                    raise e

    def train_step(self, batch):
        """Train on a single batch of examples."""
        batchsize = batch.text_vec.size(0)
        # helps with memory usage
        self._init_cuda_buffer(self.model, self.criterion, batchsize,
                               self.truncate or 180)
        self.model.train()
        self.zero_grad()

        try:
            out = self.model(batch.text_vec, batch.label_vec)

            # generated response
            scores = out[0]
            _, preds = scores.max(2)

            score_view = scores.view(-1, scores.size(-1))
            loss = self.criterion(score_view, batch.label_vec.view(-1))
            # save loss to metrics
            notnull = batch.label_vec.ne(self.NULL_IDX)
            target_tokens = notnull.long().sum().item()
            correct = ((batch.label_vec == preds) * notnull).sum().item()
            self.metrics['correct_tokens'] += correct
            self.metrics['loss'] += loss.item()
            self.metrics['num_tokens'] += target_tokens
            loss /= target_tokens  # average loss per token
            loss.backward()
            self.update_params()
        except RuntimeError as e:
            # catch out of memory exceptions during fwd/bck (skip batch)
            if 'out of memory' in str(e):
                print('| WARNING: ran out of memory, skipping batch. '
                      'if this happens frequently, decrease batchsize or '
                      'truncate the inputs to the model.')
            else:
                raise e
    def _build_cands(self, batch):
        if not batch.candidates:
            return None, None
        cand_inds = [i for i in range(len(batch.candidates))
                     if batch.candidates[i]]
        cands = [batch.candidate_vecs[i] for i in cand_inds]
        max_cands_len = max(
            [max([cand.size(0) for cand in cands_i]) for cands_i in cands]
        )
        for i, c in enumerate(cands):
            cands[i] = padded_tensor(c,
                                     use_cuda=self.use_cuda,
                                     max_len=max_cands_len)[0].unsqueeze(0)
        cands = torch.cat(cands, 0)
        return cands, cand_inds

    def _pick_cands(self, cand_preds, cand_inds, cands):
        cand_replies = [None] * len(cands)
        for idx, order in enumerate(cand_preds):
            batch_idx = cand_inds[idx]
            cand_replies[batch_idx] = [cands[batch_idx][i] for i in order]
        return cand_replies

    def greedy_search(self, batch):
        cand_params = self._build_cands(batch)
        out = self.model(batch.text_vec, ys=None, cands=cand_params[0])
        return out, cand_params

    def eval_step(self, batch):
        """Evaluate a single batch of examples."""
        if batch.text_vec is None:
            return
        orig_batch = batch  # save for evaluation
        self.model.eval()
        cand_scores = None
        out, cand_params = self.greedy_search(batch)
        scores, cand_scores = out[0], out[1]
        _, preds = scores.max(2)

        if batch.label_vec is not None:
            # calculate loss on targets with teacher forcing
            out = self.model(batch.text_vec, batch.label_vec)
            f_scores = out[0]  # forced scores
            _, f_preds = f_scores.max(2)  # forced preds
            score_view = f_scores.view(-1, f_scores.size(-1))
            loss = self.criterion(score_view, orig_batch.label_vec.view(-1))
            # save loss to metrics
            notnull = orig_batch.label_vec.ne(self.NULL_IDX)
            target_tokens = notnull.long().sum().item()
            correct = ((orig_batch.label_vec == f_preds) * notnull).sum().item()
            self.metrics['correct_tokens'] += correct
            self.metrics['loss'] += loss.item()
            self.metrics['num_tokens'] += target_tokens

        cand_choices = None
        if cand_scores is not None:
            cand_preds = cand_scores.sort(1, descending=True)[1]
            # now select the text of the cands based on their scores
            cand_choices = self._pick_cands(cand_preds, cand_params[1],
                                            orig_batch.candidates)

        text = [self._v2t(p) for p in preds]
        return Output(text, cand_choices)
