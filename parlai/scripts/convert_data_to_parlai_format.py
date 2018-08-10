# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
"""Convert a dataset into the ParlAI text format.
E.g.:
`python convert_data_to_parlai_format.py -t babi:task1k:1 --outfile /tmp/dump `
"""
# py parlai/scripts/convert_data_to_parlai_format.py -t internal:Reddit:redditPytorchData -n 1000 --ignore-fields "label_candidates,id" --max_hist_len 3

from parlai.core.params import ParlaiParser
from parlai.agents.repeat_label.repeat_label import RepeatLabelAgent
from parlai.core.worlds import create_task
from parlai.core.utils import msg_to_str

import random


def dump_data(opt):
    # create repeat label agent and assign it to the specified task
    agent = RepeatLabelAgent(opt)
    world = create_task(opt, agent)
    ignorefields = opt.get('ignore_fields', '')

    print('[ starting to convert.. ]')
    fw = open(opt['outfile'], 'w')
    for _ in range(opt['num_examples']):
        world.parley()
        world.acts[0]['labels'] = world.acts[0].get(
            'labels', world.acts[0].pop('eval_labels', None))
        #import pdb; pdb.set_trace()
        ##
        t = world.acts[0].copy()
        ss = t['text'].split('\n')
        a = []
        for s in ss:
            if 'persona:' not in s:
                a.append(s)
        t['text'] = '\n'.join(a)
        t['text'] = t['text'].replace('__START__', '')
        t['text'] = t['text'].replace('__END__', '\n') 
        if t['text'][-1] == '\n':
            t['text'] = t['text'][:-1]
        txt = msg_to_str(t, ignore_fields=ignorefields)
        ##
        fw.write(txt + '\n')
        if world.acts[0].get('episode_done', False):
            fw.write('\n')

        if world.epoch_done():
            print('EPOCH DONE')
            break
    fw.close()


def main():
    random.seed(42)
    # Get command line arguments
    parser = ParlaiParser()
    parser.add_argument('-n', '--num-examples', default=1000000000, type=int)
    parser.add_argument('-of', '--outfile', default='/tmp/dump', type=str)
    parser.add_argument('-if', '--ignore-fields', default='id', type=str)
    parser.set_defaults(datatype='train:stream')
    opt = parser.parse_args()
    dump_data(opt)


if __name__ == '__main__':
    main()
