from experiments_rnn import run_seen_experiment

# Seed must be updated in:

# main/intentSpace.py
# main/training.py
# main/helper.py

run_seen_experiment(word_vector_path="input/glove.42B.300d.txt", seen_input="input/Data-Seen", unseen_input="input/Data-Unseen")
