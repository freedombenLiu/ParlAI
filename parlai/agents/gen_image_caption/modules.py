# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

import torch
from torch import optim
import torch.nn as nn
import torchvision.models as models

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

OPTIM_OPTS = {
    'adam': optim.Adam,
    'sgd': optim.SGD
}

class GenImageCaption(nn.Module):
    def __init__(self, opt, dict):
        super().__init__()
        self.opt = opt
        self.dict = dict
        vocab_size = len(dict.tok2ind)
        self.use_state = opt['use_feature_state']
        self.encoder = EncoderCNN(embed_size=opt['embed_size'],
                                  hidden_size=opt['hidden_size'])
        self.decoder = DecoderRNN(embed_size=opt['embed_size'],
                                  hidden_size=opt['hidden_size'],
                                  vocab_size=vocab_size,
                                  dict=dict,
                                  num_layers=opt['num_layers'],
                                  max_seq_length=opt['max_pred_length'])


    def forward(self, longest_label, images, captions=None, caption_lens=None):
        features, state = self.encoder(images)
        state = state if self.use_state else None
        tokens, preds = self.decoder(features, state, captions, caption_lens, longest_label)
        return tokens, preds

    def get_optim(self):
        optim_class = OPTIM_OPTS[self.opt['optimizer']]
        kwargs = {'lr': self.opt['learning_rate']}
        if self.opt['optimizer'] == 'adam':
            # https://openreview.net/forum?id=ryQu7f-RZ
            kwargs['amsgrad'] = True

        params = list(self.decoder.parameters()) + \
                 list(self.encoder.linear.parameters()) + \
                 list(self.encoder.bn.parameters()) + \
                 list(self.encoder.hidden.parameters()) + \
                 list(self.encoder.cell.parameters())
        optim = optim_class(params, **kwargs)
        return optim


class EncoderCNN(nn.Module):
    def __init__(self, embed_size, hidden_size):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(EncoderCNN, self).__init__()
        resnet = models.resnet152(pretrained=True)
        modules = list(resnet.children())[:-1]      # delete the last fc layer.
        self.resnet = nn.Sequential(*modules)
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        self.hidden = nn.Linear(resnet.fc.in_features, hidden_size)
        self.cell = nn.Linear(resnet.fc.in_features, hidden_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)

    def forward(self, images):
        """Extract feature vectors from input images."""
        with torch.no_grad():
            features = self.resnet(images)
        features = features.reshape(features.size(0), -1)
        hidden_state = self.hidden(features)
        cell_state = self.cell(features)
        features = self.bn(self.linear(features))

        return features, (hidden_state.unsqueeze(0), cell_state.unsqueeze(0))


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers,
                 dict, max_seq_length=20, dropout=0.1):
        """Set the hyper-parameters and build the layers."""
        super(DecoderRNN, self).__init__()
        self.vocab_size = vocab_size
        self.START_IDX = dict[dict.start_token]
        self.dropout = nn.Dropout(p=dropout)
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size*2, hidden_size,
                            num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        # maybe do one linear from hidden to embed and one from embed to vocab,
        # and then make that share the weights with the embedding layer
        self.max_seg_length = max_seq_length

    def forward(self, features, state, captions, caption_lens, longest_label):
        """Decode image feature vectors and generates captions."""
        features = features.unsqueeze(1)
        if captions is not None:
            features = features.repeat(1, captions.shape[1], 1)
            # Training
            embeddings = self.dropout(self.embed(captions))
            # import pdb; pdb.set_trace()
            embeddings = torch.cat((features, embeddings), 2)
            packed = pack_padded_sequence(embeddings, caption_lens, batch_first=True)
            hiddens, _ = self.lstm(packed, state)
            unpacked_outputs, _ = pad_packed_sequence(hiddens, batch_first=True)
            outputs = self.linear(unpacked_outputs) # (bsz x vocab_size x longest sequence)
            outputs = self.dropout(outputs.permute(0,2,1))
            _, ids = outputs.max(1)
            return ids, outputs
        else:
            prev_token = torch.LongTensor(features.shape[0]).fill_(self.START_IDX)
            prev_token = self.embed(prev_token)
            prev_token = prev_token.unsqueeze(1)

            states = None
            sampled_preds = []
            max_seg_length = self.max_seg_length if longest_label is None \
                                                 else longest_label

            for i in range(max_seg_length):
                feature_tokens = torch.cat((features, prev_token), 1)
                # import pdb; pdb.set_trace()
                hidden, states = self.lstm(feature_tokens, states)
                outputs = self.linear(hidden.squeeze(1))    # outputs: (bsz, vocab_size)
                _, predicted = outputs.max(1)                # predicted: (bsz)
                # print(predicted)
                sampled_preds.append(outputs)

                if i < max_seg_length - 1:
                    inputs = self.embed(predicted)  # inputs: (batch_size, embed_size)
                    inputs = inputs.unsqueeze(1)  # inputs: (batch_size, 1, embed_size)

            # sampled_ids: (batch_size, max_seg_length)
            sampled_preds = torch.stack(sampled_preds, 2)

            _, sampled_ids = sampled_preds.max(1)
            return sampled_ids, sampled_preds
