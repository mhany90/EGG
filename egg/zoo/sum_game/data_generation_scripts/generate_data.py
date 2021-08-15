import argparse
import numpy as np
import random

import egg.core as core

# parameters for data generation
def get_params(params):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--no_dev_examples",
        type=int,
        default=1000,
        help="Number of validation examples to generate",
    )
    parser.add_argument(
        "--no_test_examples",
        type=int,
        default=1000,
        help="Number of test examples to generate",
    )
    parser.add_argument(
        "--input_range",
        type=int,
        default=1000,
        help="Max range of input integers",
    )
    parser.add_argument(
        "--holdout_pairs",
        default=False,
        action="store_true",
        help="If this flag is passed, input pair combinations (e.g. [302,101]) are heldout from the train set but included in the development and test sets.",
    )
    args = core.init(parser, params)
    return args


def main(params):
    opts = get_params(params)

    # init list of examples and lists to store keys for holding out pair combinations
    train_examples, dev_examples, test_examples = [], [], []
    dev_set_keys, test_set_keys = [], []

    # generate all combinations of two lists of range(n_range)
    comb_array = np.array(np.meshgrid(list(range(opts.input_range)), list(range(opts.input_range)))).T.reshape(-1, 2).tolist()

    # shuffle list of examples
    random.shuffle(comb_array)

    # start with generating test and dev examples to enable holding them out
    # test set
    for example in comb_array[:opts.no_test_examples]:
        first_int = example[0]
        second_int = example[1]
        # compute sum
        summation = first_int + second_int
        # append inputs and sum to list of test examples
        test_examples.append([first_int, second_int, summation])
        # append pair combination to list of test keys
        test_set_keys.append(str(first_int) + '-' + str(second_int))
    # dev set
    for example in comb_array[opts.no_test_examples: opts.no_test_examples + opts.no_dev_examples]:
        first_int = example[0]
        second_int = example[1]
        # compute sum
        summation = first_int + second_int
        # append inputs and sum to list of all examples
        dev_examples.append([first_int, second_int, summation])
        # append pair combination to list of test keys
        dev_set_keys.append(str(first_int) + '-' + str(second_int))
    # train set
    for example in comb_array[opts.no_test_examples + opts.no_dev_examples:]:
        first_int = example[0]
        second_int = example[1]
        # compute sum
        summation = first_int + second_int
        # append inputs and sum to list of all examples
        train_examples.append([first_int, second_int, summation])

    print("test set len: {}, dev set len: {}, train set len: {}".format(len(test_examples), len(dev_examples), len(train_examples)))

    # prep write out files
    train_file = "train_file_range-{}_examples-{}.txt".format(str(opts.input_range), str(len(train_examples)))
    dev_file = "dev_file_range-{}_examples-{}.txt".format(str(opts.input_range), str(opts.no_dev_examples))
    test_file = "test_file_range-{}_examples-{}.txt".format(str(opts.input_range), str(opts.no_test_examples))

    # write train file
    out_train = open(train_file, 'w')
    for train_example in train_examples:
        out_train.write(str(train_example[0]) + ' ' + str(train_example[1]) + ' ' + str(train_example[2]) + '\n')
    # write dev file
    out_dev = open(dev_file, 'w')
    for dev_example in dev_examples:
        out_dev.write(str(dev_example[0]) + ' ' + str(dev_example[1]) + ' ' + str(dev_example[2]) + '\n')
    # write testtest file
    out_test = open(test_file, 'w')
    for test_example in test_examples:
        out_test.write(str(test_example[0]) + ' ' + str(test_example[1]) + ' ' + str(test_example[2]) + '\n')


if __name__ == "__main__":
    import sys

    main(sys.argv[1:])
