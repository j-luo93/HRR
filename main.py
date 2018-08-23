from __future__ import print_function, division

import os
import argparse
from datetime import datetime
from pprint import pprint
from models import BaseLM
from datasets import Datasets
from trainer import Trainer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=str, help='data directory')
    parser.add_argument('--msg', '-M', type=str, help='message', metavar='')
    parser.add_argument('--num_layers', '-nl', metavar='', default=1, type=int, help='number of layers')
    parser.add_argument('--cell_dim', '-cd', metavar='', default=512, type=int, help='cell dimensionality')
    parser.add_argument('--num_epochs', '-ne', metavar='', default=10, type=int, help='num of epochs')
    parser.add_argument('--batch_size', '-bs', metavar='', default=128, type=int, help='batch size')
    parser.add_argument('--model', '-m', metavar='', default='base', type=str, help='model type')
    parser.add_argument('--vocab_size', '-vs', metavar='', default=0, type=int, help='vocabulary size')
    parser.add_argument('--sample_size', '-ss', metavar='', default=0, type=int, help='sample size for sampled softmax')
    parser.add_argument('--gpu', '-g', metavar='', default='', nargs='+', help='which gpu(s) to use')
    parser.add_argument('--print_interval', '-pi', metavar='', default=100, type=int, help='print training info after this many steps')
    parser.add_argument('--eval_interval', '-ei', metavar='', default=100, type=int, help='evaluate after this many steps')
    parser.add_argument('--learning_rate', '-lr', metavar='', default=0.0002, type=float, help='learning rate')
    parser.add_argument('-untied_io', dest='tied_io', action='store_false', help='untie input and output embeddings')

    args = parser.parse_args()
    assert args.model in ['baseLM'], 'model type not supported or implemented'

	# set up log directory
    now = datetime.now()
    date = now.strftime("%m-%d")
    timestamp = now.strftime("%H:%M:%S")
    if not args.msg:
        args.log_dir = 'log/%s/%s' %(date, timestamp)
    else:
        args.log_dir = 'log/%s/%s-%s' %(date, args.msg, timestamp)
	assert not os.path.isdir(args.log_dir)
	os.makedirs(args.log_dir)

    return args

def construct_model(args):
    if args.model == 'baseLM':
        model = BaseLM(**vars(args))
    return model

def prepare_datasets(args):
    datasets = Datasets(**vars(args))
    args.vocab_size = len(datasets.vocab)
    return datasets

if __name__ == '__main__':
    args = parse_args()
    pprint(args)

    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(args.gpu)

    # get data
    datasets = prepare_datasets(args)

    # init model
    model = construct_model(args)

    # get trainer
    trainer = Trainer(**vars(args))
    trainer.train(model, datasets)
