import csv
from collections import Counter

import torch
from intentSpace import IntentSpace

from evaluate import calculate_entropy
import numpy as np

model = torch.load("input/model_euclidean.pt")
text_file = "input/test_text.txt"
intent_file = "input/test_intent.txt"

def get_word2vec(file):
    output_dict = dict()
    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], "float32")
            output_dict[word] = vector
    return output_dict


word2vec = get_word2vec("input/glove.42B.300d.txt")

entropy, label = calculate_entropy(model, word2vec, text_file, intent_file)
label, entropy = (list(t) for t in zip(*sorted(zip(label, entropy))))
counter = Counter(label)
TOTAL_SEEN = counter[0]
TOTAL_UNSEEN = counter[1]
print("Seen Intents: ", TOTAL_SEEN)
print("Unseen Intents: ", TOTAL_UNSEEN)

with open("data.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Label", "Entropy"])
    for x, y in zip(label, entropy):
        writer.writerow([x, y])