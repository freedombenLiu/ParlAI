# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

import torch
import torch.nn as nn
import torchvision.models as models

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class GenImageCaption(nn.Module):
    def __init__(self, opt, dict):
        super().__init__()
        self.opt = opt
        self.dict = dict
        vocab_size = len(dict.tok2ind)
        self.encoder = EncoderCNN(embed_size=opt['embed_size'])
        self.decoder = DecoderRNN(embed_size=opt['embed_size'],
                                  hidden_size=opt['hidden_size'],
                                  vocab_size=vocab_size,
                                  num_layers=opt['num_layers'],
                                  max_seq_length=opt['max_pred_length'])


    def forward(self, longest_label, images, captions=None, caption_lens=None):
        features = self.encoder(images)
        tokens, preds = self.decoder(features, captions, caption_lens, longest_label)
        return tokens, preds

    def get_optim(self):
        params = list(self.decoder.parameters()) + \
                 list(self.encoder.linear.parameters()) + \
                 list(self.encoder.bn.parameters())
        optim = torch.optim.Adam(params, lr=self.opt['learning_rate'])
        return optim


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(EncoderCNN, self).__init__()
        resnet = models.resnet152(pretrained=True)
        modules = list(resnet.children())[:-1]      # delete the last fc layer.
        self.resnet = nn.Sequential(*modules)
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)

    def forward(self, images):
        """Extract feature vectors from input images."""
        with torch.no_grad():
            features = self.resnet(images)
        features = features.reshape(features.size(0), -1)
        features = self.bn(self.linear(features))
        return features


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, max_seq_length=20):
        """Set the hyper-parameters and build the layers."""
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size,
                            num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.max_seg_length = max_seq_length

    def forward(self, features, captions, caption_lens, longest_label):
        """Decode image feature vectors and generates captions."""
        inputs = features.unsqueeze(1)
        if captions is not None:
            # Training
            embeddings = self.embed(captions)
            embeddings = torch.cat((inputs, embeddings), 1)
            packed = pack_padded_sequence(embeddings, caption_lens, batch_first=True)
            hiddens, _ = self.lstm(packed)
            unpacked_outputs, _ = pad_packed_sequence(hiddens, batch_first=True)
            outputs = self.linear(unpacked_outputs)
            outputs = outputs.permute(0,2,1)
            _, ids = outputs.max(1)
            return ids, outputs
        else:
            states = None
            sampled_preds = []
            max_seg_length = self.max_seg_length if longest_label is None \
                                                 else longest_label

            for i in range(max_seg_length):
                hidden, states = self.lstm(inputs, states)
                outputs = self.linear(hidden.squeeze(1))    # outputs: (bsz, vocab_size)
                _, predicted = outputs.max(1)                # predicted: (bsz)
                sampled_preds.append(outputs)

                if i < max_seg_length - 1:
                    inputs = self.embed(predicted)  # inputs: (batch_size, embed_size)
                    inputs = inputs.unsqueeze(1)  # inputs: (batch_size, 1, embed_size)

            # sampled_ids: (batch_size, max_seg_length)
            sampled_preds = torch.stack(sampled_preds, 2)

            _, sampled_ids = sampled_preds.max(1)
            return sampled_ids, sampled_preds
