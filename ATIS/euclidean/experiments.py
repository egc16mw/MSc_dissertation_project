import torch

from helper import calculate_score, read_file, get_word2vec
from intentSpace import IntentSpace
from training import train
from training_details import TrainingDetails, Output, DataSet


# Experiment 1
def train_seen(data, changing_condition, accuracy_min_inc):
    train(data=data, seen=True, debug=True, changing_condition=changing_condition, accuracy_min_inc=accuracy_min_inc)


# Experiment 2
def train_unseen_alpha(data, intents, accuracy_thresold):
    # Freeze current parameters
    print("Freeze all parameters")
    for name, param in data.model.named_parameters():
        param.requires_grad = False
    # Update Intent Dictionary and add alpha
    data.model.add_intents(intents)
    data.dataset.update()
    train(data=data, seen=False, debug=True, regularisation_function=experiment2_calculate_regularisation, accuracy_thresold=accuracy_thresold)


# Experiment 3
# Omega list of alphas to add omegas too, zero indexed
def train_unseen_omega(data, accuracy_thresold, omegas=None, add_unseen=None):
    print("Freeze all parameters")
    for name, param in data.model.named_parameters():
        param.requires_grad = False

    if omegas is None and add_unseen is None:
        print("Please specify where to add omega")
        exit(0)
    elif (not (omegas is None)) and (not (add_unseen is None)):
        print("Please only specify one method")
        exit(0)

    # Add omegas from list
    if add_unseen is None:
        for omega in omegas:
            data.model.add_omegas(omega)

    # Add omega to all unseen intents
    if omegas is None:
        for i in range(data.model.B, data.model.c):
            data.model.add_omegas(i)

    train(data=data, seen=False, debug=True, regularisation_function=compute_equation54, accuracy_thresold=accuracy_thresold)


def compute_equation54(data, model):
    assert isinstance(data, TrainingDetails)
    seen_loader = data.dataset.seen_train_loader
    modifier = 0.20
    for text, labels in seen_loader:
        total = 0
        current_batch_size = text.size(0)
        max_seen = 0
        max_unseen = 0
        for i in range(current_batch_size):
            for c in range(model.c):
                output = model(text[i], c)
                score = calculate_score(model, output)
                if not c < model.B:
                    max_unseen = max(max_unseen, score)
                else:
                    max_seen = max(max_seen, score)
            total += torch.log(max_seen / max_unseen)
        return -(total / current_batch_size) * modifier


# Not optimal if adding more than 1 alpha currently
def experiment2_calculate_regularisation(data, model):
    regularisation = 0
    for i in range(model.B, model.c):
        alpha = model.get_alpha(i)
        for coordinate in alpha:
            regularisation += (coordinate - (1 / model.B)) ** 2
        regularisation /= model.B
    return regularisation


# RUN ALL THE EXPERIMENTS, CALLED TO RUN EXPERIMENTS
def run_experiments(word_vector_path, seen_input, unseen_input, output_dir=None, euclidean=None):
    batch_size = 50
    learning_rate_decay = 0.9
    changing_condition = 5
    accuracy_min_inc = 0.1
    weight_decay = 10e-5
    lr = 0.001 * 100
    max_epochs = 50
    acc_threshold = 0.95
    max_attempts = 10
    unseen_train_intent_file = unseen_input + "/train_intent.txt"
    word2vec = get_word2vec(word_vector_path)
    word_dimension = word2vec[str(list(word2vec)[0])].shape[0]
    hidden_dim = word_dimension

    intents = sorted(set(read_file(seen_input + "/train_intent.txt")))
    int2intent = dict(enumerate(intents))
    total_intents = len(int2intent)

    simplex = True if euclidean is None else False
    print("Simplex Model: ", simplex)
    model = IntentSpace(input_size=word_dimension, hidden_dim=hidden_dim, B=total_intents,
                        simplex=simplex, int2intent=int2intent)

    model.freeze_alpha()
    if output_dir is None:
        output_dir = "output"
    output = Output(output_dir + str(1), "model.pt", "stats.csv", "alpha.csv")
    data = TrainingDetails(model, output, lr, max_epochs, word2vec, learning_rate_decay, weight_decay, max_attempts)
    dataset = DataSet(data, seen_input, unseen_input, batch_size)
    data.dataset = dataset
    train_seen(data, changing_condition, accuracy_min_inc)

