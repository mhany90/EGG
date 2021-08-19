import argparse
import numpy as np
from os import listdir
from os.path import isfile, join
import re
import json
from collections import defaultdict, OrderedDict
import pandas as pd


import egg.core as core

def get_params(params):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results_dir",
        type=str,
        help="Directory with results",
    )
    parser.add_argument(
        "--exclude_failed_runs",
        default=False,
        action="store_true",
        help="Exclude failed runs from result summary",
    )
    args = core.init(parser, params)
    return args

def parse_file(file):
    """
    takes results file as input and returns lists of train and dev statistics per epoch
    """
    file_string = open(file, 'r').read()
    # split log file by section
    log_split = \
        re.split('INPUTS|LABELS|MESSAGES|OUTPUTS', file_string)
    results, inputs, labels, messages, outputs = log_split

    ## read result dictionaries
    # create lists for storing result items
    train_acc_list, train_sender_entropy_list, train_length_list, \
    dev_acc_list, dev_sender_entropy_list, dev_length_list, = [], [], [], [], [], []
    # split and process line by line
    results_list =  re.split('\n', results)
    # remove first two lines of results section, which are not relevant and last empty line
    results_list = results_list[2:-1]

    for result_line in results_list:
        result_dict = json.loads(result_line)
        if result_dict['mode'] == 'train':
            train_acc_list.append(result_dict['acc'])
            train_sender_entropy_list.append(result_dict['sender_entropy'])
            train_length_list.append(result_dict['length'])
        else:
            dev_acc_list.append(result_dict['acc'])
            dev_sender_entropy_list.append(result_dict['sender_entropy'])
            dev_length_list.append(result_dict['length'])
    return (train_acc_list, train_sender_entropy_list, train_length_list),\
           (dev_acc_list, dev_sender_entropy_list, dev_length_list)


def get_summary_stats(results_tuple):
    """
    takes results tuple and returns summary statistics
    """
    acc_list, sender_entropy_list, lengths_list = results_tuple
    # get max acc and max acc index
    max_acc, max_acc_index = float(np.max(acc_list)), int(np.argmax(acc_list))
    # get mean sender entropy
    mean_sender_entropy = float(np.mean(sender_entropy_list))
    # get mean message len
    mean_message_len = float(np.mean(lengths_list))
    return {"max_acc":max_acc, "max_acc_index":max_acc_index, "mean_sender_entropy":mean_sender_entropy, "mean_message_len":mean_message_len}

def main(params):
    opts = get_params(params)

    # read all result files in dir
    log_files = [join(opts.results_dir, f) for f in listdir(opts.results_dir)
                 if isfile(join(opts.results_dir, f)) and 'stats' not in f]

    # init nested default dictionaries for storing results
    train_results_dict = defaultdict(dict)
    dev_results_dict = defaultdict(dict)

    # iterate over files
    for file in log_files:
        # extract experiment settings from log file name
        seed=file.split('/')[-1].split('_')[0].replace('seed-','')
        vocab=file.split('/')[-1].split('_')[1].replace('vocab-','')
        maxlen=file.split('/')[-1].split('_')[2].replace('maxlen-','')
        entropy_coef = file.split('/')[-1].split('_')[3].replace('ec-','').replace('.txt', '')
        # parse file
        train_results_tuple, dev_results_tuple = parse_file(file)
        # get summary stats
        train_summary_stats = get_summary_stats(train_results_tuple)
        dev_summary_stats = get_summary_stats(dev_results_tuple)
        # calculate mean and store, only include succesful run acc. to max_acc of train > 0.10
        if opts.exclude_failed_runs:
            if train_summary_stats['max_acc'] > 0.10:
                df = pd.DataFrame([train_results_dict[vocab + '-' + maxlen + '-' + entropy_coef], train_summary_stats])
                train_results_dict_of_means = dict(df.mean())
                train_results_dict[vocab + '-' + maxlen + '-' + entropy_coef] = train_results_dict_of_means
                # same for dev
                df = pd.DataFrame([dev_results_dict[vocab + '-' + maxlen + '-' + entropy_coef], dev_summary_stats])
                dev_results_dict_of_means = dict(df.mean())
                dev_results_dict[vocab + '-' + maxlen + '-' + entropy_coef] = dev_results_dict_of_means
        else:
            df = pd.DataFrame([train_results_dict[vocab + '-' + maxlen + '-' + entropy_coef], train_summary_stats])
            train_results_dict_of_means = dict(df.mean())
            train_results_dict[vocab + '-' + maxlen + '-' + entropy_coef] = train_results_dict_of_means
            # same for dev
            df = pd.DataFrame([dev_results_dict[vocab + '-' + maxlen + '-' + entropy_coef], dev_summary_stats])
            dev_results_dict_of_means = dict(df.mean())
            dev_results_dict[vocab + '-' + maxlen + '-' + entropy_coef] = dev_results_dict_of_means

    # sort by max acc
    train_results_dict = OrderedDict(sorted(train_results_dict.items(), key=lambda x: x[1]['max_acc'], reverse=True))
    dev_results_dict = OrderedDict(sorted(dev_results_dict.items(), key=lambda x: x[1]['max_acc'], reverse=True))

    # prep write out files
    train_file = open(join(opts.results_dir,"train_summary_stats.txt"), 'w')
    dev_file = open(join(opts.results_dir,"dev_summary_stats.txt"), 'w')

    # write
    json.dump(train_results_dict, train_file, indent=6)
    train_file.close()
    json.dump(dev_results_dict, dev_file, indent=6)
    dev_file.close()

if __name__ == "__main__":
    import sys

    main(sys.argv[1:])
