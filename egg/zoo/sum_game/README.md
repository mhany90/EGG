## Introduction

The `play.py` script in this directory implements a arithmetic addition game where Two agents, a sender and a receiver are jointly optimised to solve an arithmetic addition task (e.g. 2 + 5 = 7). As input, the sender receivers two integers (2 and 5) and communicates a (discrete) message to the receiver, which is supposed to output their sum (7). In particular, we consider the case here when inputs are vectors of discrete elements (integer pairs), and we let the user pass these inputs through text files. [This directory](data_generation_scripts) contains scripts that can generate input files in the right format, as well as samples from their outputs.

## Sum game

The sum game reads input from files that have an input item (two integers) on each line.


Here is an example of how to run the game:

```bash
python -m egg.zoo.sum_game.play --mode 'rf' --train_data "egg/zoo/sum_game/data_generation_scripts/train_file_range-100_examples-8000.txt"   --validation_data "egg/zoo/sum_game/data_generation_scripts/dev_file_range-100_examples-1000.txt" --n_range 500 --n_epochs 10 --batch_size 64 --validation_batch_size 1000 --max_len 7 --vocab_size 10 --sender_hidden 20 --receiver_hidden 40 --sender_embedding 10 --receiver_embedding 10 --receiver_cell "gru" --sender_cell "gru" --lr 0.01 --print_validation_event
```

In this particular instance, the following parameters are invoked:
 * `mode` -- tells whether to use Reinforce (`rf`) or Gumbel-Softmax (`gs`) for training.
 * `train_data/validation_data` -- paths to the files containing training data and validation data (the latter used at each epoch to track the progress of training); both files are in the same format.
 * `n_range` -- the maximum range of integers 
 * `n_epochs` -- how many times the data in the input training file will be traversed during training: note that they will be traversed in a different random order each time.
 * `batch_size` -- batch size for training data (can't be smaller than number of items in training file).
 * `validation_batch_size` -- batch size for validation data, provided as a separate argument as it is often convenient to traverse the whole validation set in a single step.
 * `max_len` -- after `max_len` symbols without `<eos>` have been emitted by the Sender, an `<eos>` is forced; consequently, the longest possible message will contain `max_len` symbols, followed by `<eos>`.
 * `vocab_size` -- the number of unique symbols in the Sender vocabulary (inluding `<eos>`!).
 * `sender_hidden/receiver_hidden` -- the size of the hidden layers of the agents.
 * `sender_embedding/receiver_embedding` -- output dimensionality of the layer that embeds symbols produced at previous step by the Sender message-emitting/Receiver message-processing recurrent networks, respectively.
 * `sender_cell/receiver_cell` -- type of cell of recurrent networks agents use to emit/process the message.
 * `lr` -- learning rate.
 * `print_validation_events` -- if this flag is passed, after training is done the script will print the validation input, as well as the corresponding messages emitted by Sender and the corresponding Receiver outputs.
 
 To see all arguments that can be passed (and for more information on the ones above), run:
 
 ```bash
python -m egg.zoo.basic_games.play -h
```

## Output

The script will write output to STDOUT, including information on the chosen parameters, and training and validation statistics at each epoch. The exact output might change depending on the chosen parameters and independent EGG development, but it will always contain information about the loss and accuracy (proportion of successful game rounds).

When the `printer_validation_events` flag is passed, the script prints detailed information about the last validation pass. In particular, for all inputs in the validation set, the script prints the following lists: the inputs, the corresponding gold labels, the messages produced by Sender and the outputs of Receiver. The exact nature of each of these lists will change depending on game type and other parameters, but the following considerations hold in general:
* **INPUTS** -- The input items are printed in one-hot vector format. In discrimination games, only Sender inputs (that is, the target items) are printed.
* **LABELS** -- In discrimination games, these are the same indices of target location present in the input file. In recognition games, they are identical to the input items (in the original input format).
* **MESSAGES** -- For technical reasons, these are represented in integer format when using Reinforce (with 0 as `<eos>` delimiter) and as one-hot vectors for Gumbel-Softmax (with the 0-th position denoting `<eos>`). Note that, at validation/test time, any symbol following the first occurrence of `<eos>` can be ignored.
* **OUTPUTS** -- These are the non-normalized (pre-softmax) scores produced by the Receiver. In reconstruction games, each output will be the concatenation of the distributions over the values of each input attribute: for example, if inputs are two-attribute vectors with 5 possible values, each output item will be a list of 10 numbers, the first half to be interpreted as a non-normalized probability distribution over the values of the first attribute, and the second half as the same for the second attribute. In discrimination games, the output represents non-normalized probabilities for the possible position of the target in the input item array. When training with Gumbel-Softmax, this distribution is printed for each symbol produced by Sender. The one that is taken as the effective Receiver output for evaluation purposes is the distribution emitted in correspondance to the first `<eos>` emitted by Sender.
 
