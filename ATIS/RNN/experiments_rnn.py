from helper import read_file, get_word2vec
from myRNN import MyRNN
from training import train
from training_details import TrainingDetails, Output, DataSet


# Experiment 1
def train_seen(data, accuracy_min_inc):
    train(data=data, debug=True, accuracy_min_inc=accuracy_min_inc)


# RUN ALL THE EXPERIMENTS, CALLED TO RUN EXPERIMENTS
def run_seen_experiment(word_vector_path, seen_input, unseen_input, output_dir=None):
    batch_size = 50
    learning_rate_decay = 0.9
    accuracy_min_inc = 0.1
    weight_decay = 10e-5
    lr = 0.001 * 10
    max_epochs = 50
    max_attempts = 50
    word2vec = get_word2vec(word_vector_path)
    word_dimension = word2vec[str(list(word2vec)[0])].shape[0]
    hidden_dim = word_dimension

    intents = sorted(set(read_file(seen_input + "/train_intent.txt")))
    int2intent = dict(enumerate(intents))
    total_intents = len(int2intent)

    model = MyRNN(input_size=word_dimension, hidden_dim=hidden_dim, total_intents=total_intents, int2intent=int2intent)

    if output_dir is None:
        output_dir = "output"
    output = Output(output_dir + str(1), "model.pt", "stats.csv", "alpha.csv")
    data = TrainingDetails(model, output, lr, max_epochs, word2vec, learning_rate_decay, weight_decay, max_attempts)
    dataset = DataSet(data, seen_input, unseen_input, batch_size)
    data.dataset = dataset
    train_seen(data, accuracy_min_inc)

