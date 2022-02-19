from experiments import run_experiments

# Seed must be updated in:

# main/intentSpace.py
# main/training.py
# main/helper.py

run_experiments(word_vector_path="input/glove.42B.300d.txt", seen_input="input/Data-Seen", unseen_input="input/Data-Unseen", euclidean=True)
