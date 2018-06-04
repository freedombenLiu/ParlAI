# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

from parlai.core.torch_agent import TorchAgent
from parlai.core.dict import DictionaryAgent
from .modules import GenImageCaption


class GenImageCaptionAgent(TorchAgent):
    """
    """

    @staticmethod
    def add_cmdline_args(argparser):
        """Add command-line arguments specifically for this agent."""
        agent = argparser.add_argument_group('Image Caption Model Arguments')
        agent.add_argument('--embed_size', type=int , default=256,
                           help='dimension of word embedding vectors')
        agent.add_argument('--hidden_size', type=int , default=512,
                           help='dimension of lstm hidden states')
        agent.add_argument('--num_layers', type=int , default=1,
                           help='number of layers in lstm')
        agent.add_argument('--max_pred_length', type=int, default=20,
                           help='maximum length of predicted caption')
        GenImageCaptionAgent.dictionary_class().add_cmdline_args(argparser)

    @staticmethod
    def dictionary_class():
        return DictionaryAgent

    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        self.optimizer = None
        self.model = None

    def observe(self, observation):
        pass

    def batch_act(self, observations):
        pass

    def predict(self, xs, ys=None):
        is_training = ys is not None
        if is_training:
            self.model.train()
            self.optimizer.zero_grad()
            pred = self.model(xs, ys)

        else:
            self.model.eval()

        pass

    def share(self):
        pass

    def save(self, path):
        pass

    def act(self):
        pass

    def shutdown(self):
        pass
