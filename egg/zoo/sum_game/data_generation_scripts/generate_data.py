import argparse
import numpy as np
import random

import egg.core as core

# parameters for data generation
def get_params(params):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--no_train_examples",
        type=int,
        default=10000,
        help="Number of train examples to generate",
    )
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

    # start with generating test and dev examples to enable holding them out
    # test set
    for _ in range(opts.no_test_examples):
        # uniformly sample two integers from range
        first_int = random.randint(0, opts.input_range - 1)
        second_int = random.randint(0, opts.input_range - 1)
        # compute sum
        summation = first_int + second_int
        # append inputs and sum to list of all examples
        test_examples.append([first_int, second_int, summation])
        # append pair combination to list of test keys
        test_set_keys.append(str(first_int) + '-' + str(second_int))
    # dev set
    for _ in range(opts.no_dev_examples):
        # uniformly sample two integers from range
        first_int = random.randint(0, opts.input_range - 1)
        second_int = random.randint(0, opts.input_range - 1)
        # compute sum
        summation = first_int + second_int
        # append inputs and sum to list of all examples
        dev_examples.append([first_int, second_int, summation])
        # append pair combination to list of test keys
        dev_set_keys.append(str(first_int) + '-' + str(second_int))
    # train set
    # holdout pair combinations from trainset
    if opts.holdout_pairs:
        holdout_keys = set(test_set_keys + dev_set_keys)
        for _ in range(opts.no_train_examples):
            # uniformly sample two integers from range
            first_int = random.randint(0, opts.input_range - 1)
            second_int = random.randint(0, opts.input_range - 1)
            #check if pair in dev or test sets (in either order)
            order_1 = str(first_int) + '-' + str(second_int)
            order_2 = str(second_int) + '-' + str(first_int)
            if order_1 not in holdout_keys and order_2 not in holdout_keys:
                # compute sum
                summation = first_int + second_int
                # append inputs and sum to list of all examples
                train_examples.append([first_int, second_int, summation])
    else:
        for _ in range(opts.no_train_examples):
            # uniformly sample two integers from range
            first_int = random.randint(0, opts.input_range - 1)
            second_int = random.randint(0, opts.input_range - 1)
            summation = first_int + second_int
            # append inputs and sum to list of all examples
            train_examples.append([first_int, second_int, summation])

    # prep write out files
    train_file = "train_file_range-{}_examples-{}_holdout-{}.txt".format(str(opts.input_range), str(opts.no_train_examples), str(opts.holdout_pairs))
    dev_file = "dev_file_range-{}_examples-{}_holdout-{}.txt".format(str(opts.input_range), str(opts.no_dev_examples), str(opts.holdout_pairs))
    test_file = "test_file_range-{}_examples-{}_holdout-{}.txt".format(str(opts.input_range), str(opts.no_test_examples), str(opts.holdout_pairs))

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
