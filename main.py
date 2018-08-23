from __future__ import print_function, division

import argparse
from pprint import pprint
from models import BaseLM
from datasets import Datasets
from trainer import Trainer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=str, help='data directory')
    parser.add_argument('--num_layers', '-nl', metavar='', default=1, type=int, help='number of layers')
    parser.add_argument('--cell_dim', '-cd', metavar='', default=512, type=int, help='cell dimensionality')
    parser.add_argument('--num_epochs', '-ne', metavar='', default=10, type=int, help='num of epochs')
    parser.add_argument('--batch_size', '-bs', metavar='', default=128, type=int, help='batch size')
    parser.add_argument('--model', '-m', metavar='', default='base', type=str, help='model type')
    parser.add_argument('--vocab_size', '-vs', metavar='', default=0, type=int, help='vocabulary size')
    parser.add_argument('--sample_size', '-ss', metavar='', default=0, type=int, help='sample size for sampled softmax')
    parser.add_argument('--learning_rate', '-lr', metavar='', default=0.0002, type=int, help='learning rate')
    parser.add_argument('-untied_io', dest='tied_io', action='store_false', help='untie input and output embeddings')

    args = parser.parse_args()
    assert args.model in ['baseLM'], 'model type not supported or implemented'
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

    # get data
    datasets = prepare_datasets(args)

    # init model
    model = construct_model(args)

    # get trainer
    trainer = Trainer(**vars(args))
    trainer.train(model, datasets)
