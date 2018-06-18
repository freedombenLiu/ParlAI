# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

from parlai.core.torch_agent import TorchAgent
from parlai.core.build_data import modelzoo_path
from parlai.core.dict import DictionaryAgent
from .modules import GenImageCaption
from parlai.core.utils import round_sigfigs

import torch
import torch.nn as nn

import torchvision.transforms as transforms
from torch.nn.utils.rnn import pack_padded_sequence

import math
import os
import random


class GenImageCaptionAgent(TorchAgent):
    """
    """

    @staticmethod
    def add_cmdline_args(argparser):
        """Add command-line arguments specifically for this agent."""
        TorchAgent.add_cmdline_args(argparser)
        agent = argparser.add_argument_group('Image Caption Model Arguments')
        agent.add_argument('--embed_size', type=int , default=256,
                           help='dimension of word embedding vectors')
        agent.add_argument('--hidden_size', type=int , default=512,
                           help='dimension of lstm hidden states')
        agent.add_argument('--num_layers', type=int , default=1,
                           help='number of layers in lstm')
        agent.add_argument('--max_pred_length', type=int, default=20,
                           help='maximum length of predicted caption in eval mode')
        agent.add_argument('-lr', '--learning_rate', type=float, default=0.001,
                           help='learning rate')
        agent.add_argument('-opt', '--optimizer', default='adam',
                           choices=['sgd', 'adam'],
                           help='Choose either sgd or adam as the optimizer.')
        agent.add_argument('--use_feature_state', type='bool',
                           default=True,
                           help='Initialize LSTM state with image features')
        agent.add_argument('--concat_img_feats', type='bool', default=True,
                           help='Concat resnet feats to each token during generation')
        GenImageCaptionAgent.dictionary_class().add_cmdline_args(argparser)

    @staticmethod
    def dictionary_class():
        return DictionaryAgent

    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)

        if not shared:
            self.image_size = opt['image_size']
            self.crop_size = opt['image_cropsize']

            # initialize the transform function using torch vision.
            self.transform = transforms.Compose([
                transforms.Scale(self.image_size),
                transforms.CenterCrop(self.crop_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
            self.model = GenImageCaption(opt, self.dict, self.use_cuda)
            self.metrics = {'loss': 0.0, 'num_tokens': 0}

            load_model = None
            states = {}
            if opt.get('model_file') and os.path.isfile(opt['model_file']):
                load_model = opt['model_file']
            if load_model is not None:
                # load model parameters if available
                print('[ Loading existing model params from {} ]'.format(load_model))
                states = self.load(opt['model_file'])


            if not states:
                # set up preinitialized embeddings
                try:
                    import torchtext.vocab as vocab
                except ModuleNotFoundError as ex:
                    print('Please install torch text with `pip install torchtext`')
                    raise ex

                init = 'glove'
                embs = vocab.GloVe(name='840B', dim=300,
                                   cache=modelzoo_path(self.opt.get('datapath'),
                                                       'models:glove_vectors'))

                if opt['embed_size'] != 300:
                    rp = torch.Tensor(300, opt['embed_size']).normal_()
                    t = lambda x: torch.mm(x.unsqueeze(0), rp)
                else:
                    t = lambda x: x
                cnt = 0
                for w, i in self.dict.tok2ind.items():
                    if w in embs.stoi:
                        vec = t(embs.vectors[embs.stoi[w]])
                        self.model.decoder.embed.weight.data[i] = vec
                        cnt += 1
                print('Gen_Image_Caption: initialized embeddings for {} tokens from {}.'
                      ''.format(cnt, init))
            if states:
                # set loaded states if applicable
                self.model.load_state_dict(states['model'])

            self.criterion = nn.CrossEntropyLoss(
                                ignore_index=self.NULL_IDX, size_average=False)

            if self.use_cuda:
                self.model.cuda()
                self.criterion.cuda()

            self.optimizer = self.model.get_optim()
            if 'optimizer' in states:
                try:
                    self.optimizer.load_state_dict(states['optimizer'])
                except ValueError:
                    print('WARNING: not loading optim state since model '
                          'params changed.')
                if self.use_cuda:
                    for state in self.optimizer.state.values():
                        for k, v in state.items():
                            if isinstance(v, torch.Tensor):
                                state[k] = v.cuda()

        self.reset()

    def reset(self):
        self.observation = None
        self.episode_done = False
        if hasattr(self, "metrics"):
            self.reset_metrics()

    def reset_metrics(self):
        """Reset metrics for reporting loss and perplexity."""
        self.metrics['loss'] = 0.0
        self.metrics['num_tokens'] = 0

    def observe(self, observation):
        """Save observation for act."""
        # shallow copy observation (deep copy can be expensive)
        observation = observation.copy()
        self.observation = observation
        self.episode_done = observation['episode_done']
        return observation

    def batch_act(self, observations):
        batch_size = len(observations)
        # initialize a table of replies with this agent's id
        batch_reply = [{'id': self.getID()} for _ in range(batch_size)]

        is_training = any(['labels' in obs for obs in observations])

        vec_obs = [self.vectorize(obs, addStartIdx=False)
                   for obs in observations]

        # Need to copy over the labels and vectors into the text field so that
        # map_valid will order the batch based on the captions, which are in
        # the label field. We do this after vectorize so that the START_IDX and
        # END_IDX are added to the captions and included in the packed tensor
        # returned by map_valid.
        for item in observations:
            if is_training:
                if 'labels' in item:
                    item['text'] = item['labels'][0]
                    item['text_vec'] = item['labels_vec'][0]
            else:
                if 'eval_labels' in item:
                    item['text'] = item['eval_labels'][0]
                    item['text_vec'] = item['eval_labels_vec'][0]

        xs, x_lens, _, labels, valid_inds = self.map_valid(vec_obs)

        if xs is None:
            return batch_reply

        valid_img_obs = [obs for obs in observations if 'text' in obs ]
        # Prepare the images
        for ex in valid_img_obs:
            ex['image'] = self.transform(ex['image'])
            # if self.use_cuda:
            #     ex['image'] = ex['image'].cuda(async=True)

        images = torch.stack([ex['image'] for ex in valid_img_obs])
        if self.use_cuda:
            images = images.cuda(async=True)

        predictions, loss = self.predict(images, xs, x_lens, is_training=is_training)

        if loss is not None:
            batch_reply[0]['metrics'] = {'loss': loss.item()}

        unmap_pred = self.unmap_valid(predictions, valid_inds, batch_size)
        unmap_labels = self.unmap_valid(labels, valid_inds, batch_size)
        # Format the predictions into reply format
        for rep, pred in zip(batch_reply, unmap_pred):
            if pred is not None:
                output_tokens = []
                # Remove the final END_TOKEN that is appended to predictions
                for i, token in enumerate(pred):
                    if token == self.END_IDX and i != 0:
                        break
                    else:
                        output_tokens.append(token)
                rep['text'] = self.dict.vec2txt(output_tokens)
        ran = random.random()
        if (not is_training and xs is not None and ran < 0.05) or (is_training and ran < 0.25):
            pred_text = [o.get('text', None) for o in batch_reply]
            for pred, label in list(zip(pred_text, unmap_labels))[:5]:
                if label is not None:
                    print("Predicted: {} \tActual: {}".format(pred, label))

        return batch_reply

    def predict(self, xs, ys=None, y_lens=None, is_training=False):
        loss = None
        longest_label = None if ys is None else ys.shape[1]
        if is_training:
            self.model.train()
            self.optimizer.zero_grad()
            tokens, scores = self.model(longest_label, xs, ys, y_lens)
            loss = self.criterion(scores.float(), ys)
            # save loss to metrics
            target_tokens = ys.ne(self.NULL_IDX).long().sum().item()
            self.metrics['loss'] += loss.item()
            self.metrics['num_tokens'] += target_tokens
            loss /= target_tokens  # average loss per token
            loss.backward()
            self.optimizer.step()

        else:
            self.model.eval()
            tokens, scores = self.model(longest_label, xs, None)
            if ys is not None:
                loss = self.criterion(scores.float(), ys)
                target_tokens = ys.ne(self.NULL_IDX).long().sum().item()
                self.metrics['loss'] += loss.item()
                self.metrics['num_tokens'] += target_tokens
                loss /= target_tokens  # average loss per token

        return tokens, loss

    def report(self):
        """Report loss and perplexity from model's perspective.

        Note that this includes predicting __END__ and __UNK__ tokens and may
        differ from a truly independent measurement.
        """
        m = {}
        if self.metrics['num_tokens'] > 0:
            m['loss'] = self.metrics['loss'] / self.metrics['num_tokens']
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
        shared['metrics'] = self.metrics
        shared['model'] = self.model
        shared['states'] = {  # only need to pass optimizer states
            'optimizer': self.optimizer.state_dict()
        }
        return shared

    def save(self, path):
        """Save model parameters if model_file is set."""
        path = self.opt.get('model_file', None) if path is None else path

        if path and hasattr(self, 'model'):
            model = {}
            model['model'] = self.model.state_dict()
            model['optimizer'] = self.optimizer.state_dict()
            model['opt'] = self.opt

            with open(path, 'wb') as write:
                torch.save(model, write)

    def load(self, path):
        """Return opt and model states."""
        states = torch.load(path, map_location=lambda cpu, _: cpu)
        return states

    def act(self):
        # call batch_act with this batch of one
        return self.batch_act([self.observation])[0]

    def shutdown(self):
        """Save the state of the model when shutdown."""
        path = self.opt.get('model_file', None)
        if path is not None:
            self.save(path + '.shutdown_state')
        super().shutdown()
