import torch
from experiments import train_unseen_alpha, train_unseen_omega, train_seen
from intentSpace import IntentSpace
from helper import get_word2vec, read_file
from training_details import TrainingDetails, Output, DataSet

BATCH_SIZE = 50
learning_rate_decay = 0.9
changing_condition = 5
accuracy_min_inc = 0.1
weight_decay = 10e-5
lr = 0.001 * 100
max_epochs = 50
acc_threshold = 0.95
max_attempts = 10
INPUT_DIR = "input/"
UNSEEN_TRAIN_INTENT_FILE = INPUT_DIR + "/Data-Unseen/train_intent.txt"
word2vec = get_word2vec(INPUT_DIR + "glove.42B.300d.txt")
WORD_DIMENSION = word2vec[str(list(word2vec)[0])].shape[0]
HIDDEN_DIM = WORD_DIMENSION
data_points = [1, 10, 100, 500, 1000, 1500]

intents = sorted(set(read_file(INPUT_DIR + "/Data-Seen/train_intent.txt")))
int2intent = dict(enumerate(intents))
total_intents = len(int2intent)

train_text = "train_text.txt"
train_intent = "train_intent.txt"
valid_text = "valid_text.txt"
valid_intent = "valid_intent.txt"
test_text = "test_text.txt"
test_intent = "test_intent.txt"


def vertify_data_set(dataset, expected_count):
    loaders = [dataset.unseen_train_loader]

    for loader in loaders:
        if loader is None:
            print("WARNING: Couldn't vertify loader, value is NONE")
        else:
            count = dict()
            for text, labels in loader:
                for label in labels:
                    intent = int(label.item())
                    if intent in count:
                        count[intent] += 1
                    else:
                        count[intent] = 1

            for value in count.values():
                assert expected_count == value




model = IntentSpace(input_size=WORD_DIMENSION, hidden_dim=HIDDEN_DIM, B=total_intents,
                        simplex=True, int2intent=int2intent)

output = Output("experiment1", "model.pt", "stats.csv", "alpha.csv")
data = TrainingDetails(model, output, lr, max_epochs, word2vec, learning_rate_decay, weight_decay, max_attempts)

# dataset = DataSet(data, INPUT_DIR + "Data-Seen", INPUT_DIR + "Data-Unseen", BATCH_SIZE)
# data.dataset = dataset
#
# train_seen(data, changing_condition, accuracy_min_inc)
for data_point in data_points:
    model = torch.load("experiment1/model.pt")
    data.model = model

    output_dir_name = "output_" + str(data_point) + "_experiment"

    dataset = DataSet(data, INPUT_DIR + "Data-Seen/", INPUT_DIR + "Data-Unseen/" + str(data_point) + "sentences",
                      BATCH_SIZE)

    data.dataset = dataset

    # Experiment 2
    data.max_epochs = 150
    data.lr = 0.001 * 100
    data.output.update_directory(output_dir_name + str(2))
    intents = sorted(set(read_file(UNSEEN_TRAIN_INTENT_FILE)))

    train_unseen_alpha(data, intents, acc_threshold)
    vertify_data_set(data.dataset, data_point)

    # Experiment 3 Details
    data.max_epochs = 500
    data.lr = 0.001 * 1000
    data.output.update_directory(output_dir_name + str(3))
    train_unseen_omega(data, acc_threshold, add_unseen=True)
