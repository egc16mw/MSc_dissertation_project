import os
from helper import get_loader
from myRNN import MyRNN


class TrainingDetails:
    def __init__(self, model, output, lr, max_epochs, word2vec, learning_rate_decay, weight_decay, max_attempts, dataset=None):
        '''
        Parameters
        ----------
        model: The model that should be trained, IntentSpace class
        output: Output class
        lr: Learning Rate
        max_epochs: Max epochs to train for
        word2vec: A dictionary converting words to vectors
        learning_rate_decay: A multiplier to decrease lr with when reverting (e.g. 0.95 to decrease 5%)
        weight_decay: Weight decay to use with optimiser
        max_attempts: Max attempts allowed to recover/revert
        dataset: A class representing the dataset
        '''
        self.output = output
        self.learning_rate_decay = learning_rate_decay
        self.weight_decay = weight_decay
        if not isinstance(dataset, DataSet):
            print("Remember to set Dataset before starting to train")
        assert isinstance(output, Output)
        assert isinstance(model, MyRNN)
        self.lr = lr
        self.word2vec = word2vec
        self.max_epochs = max_epochs
        self.model = model
        self.dataset = dataset
        self.max_attempts = max_attempts


class DataSet:
    def __init__(self, training_details, seen_input_dir, unseen_input_dir, batch_size):
        '''
        Parameters
        ----------
        training_details: Class for training details
        seen_input_dir: Directory where seen data is located
        unseen_input_dir: Directory where unseen data is located
        batch_size: batch size to use in all loaders
        '''
        self.training_details = training_details
        self.seen_input_dir = seen_input_dir + "/"
        self.unseen_input_dir = unseen_input_dir + "/"
        self.batch_size = batch_size

        self.train_text = "train_text.txt"
        self.train_intent = "train_intent.txt"
        self.valid_text = "valid_text.txt"
        self.valid_intent = "valid_intent.txt"
        self.test_text = "test_text.txt"
        self.test_intent = "test_intent.txt"
        self.seen_train_loader = None
        self.seen_valid_loader = None
        self.seen_test_loader = None
        self.unseen_train_loader = None
        self.unseen_valid_loader = None
        self.unseen_test_loader = None
        self.update()

    def update(self):
        self.seen_train_loader = get_loader(self.training_details.model, self.training_details.word2vec, self.seen_input_dir + self.train_text, self.seen_input_dir + self.train_intent, self.batch_size)
        self.seen_valid_loader = get_loader(self.training_details.model, self.training_details.word2vec, self.seen_input_dir + self.valid_text, self.seen_input_dir + self.valid_intent, self.batch_size)
        self.seen_test_loader = get_loader(self.training_details.model, self.training_details.word2vec, self.seen_input_dir + self.test_text, self.seen_input_dir + self.test_intent, self.batch_size)
        try:
            self.unseen_train_loader = get_loader(self.training_details.model, self.training_details.word2vec, self.unseen_input_dir + self.train_text, self.unseen_input_dir + self.train_intent, self.batch_size)
            self.unseen_valid_loader = get_loader(self.training_details.model, self.training_details.word2vec, self.unseen_input_dir + self.valid_text, self.unseen_input_dir + self.valid_intent, self.batch_size)
            self.unseen_test_loader = get_loader(self.training_details.model, self.training_details.word2vec, self.unseen_input_dir + self.test_text, self.unseen_input_dir + self.test_intent, self.batch_size)
        except (KeyError, FileNotFoundError):
            print("Not all intents are the dictionary, only created seen loaders")


class Output:
    def __init__(self, directory, model, stats, alpha):
        '''
        Parameters
        ----------
        directory: The directory where the files should be located
        model: Where the output model should be saved
        stats: Where the training stats should be saved
        alpha: Where the file containing alphas should be saved
        '''
        self.directory = directory + "/"
        self.modelz = model
        self.statsz = stats
        self.alphaz = alpha
        self.model = self.directory + model
        self.stats = self.directory + stats
        self.alpha = self.directory + alpha
        create_directory(self.directory)

    def update_directory(self, directory):
        '''
        Parameters
        ----------
        directory: Update the output directory along with all output files (e.g. model)
        '''
        self.directory = directory + "/"
        self.model = self.directory + self.modelz
        self.stats = self.directory + self.statsz
        self.alpha = self.directory + self.alphaz
        create_directory(self.directory)


# Create a directory if not already created
def create_directory(directory):
    if not os.path.exists(directory):
        print("Creating directory: " + str(directory))
        os.makedirs(directory)