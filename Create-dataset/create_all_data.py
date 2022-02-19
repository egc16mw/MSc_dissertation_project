import os, codecs


def read_file(intent_file):
    with codecs.open(intent_file, "r", encoding="utf-8") as file:
        read_intent = file.read()

    return read_intent.split('\n')

INPUT_FOLDER = "Data/"
UNIQUE_INTENTS = [x for x in set(read_file("Data/train_intent.txt")) if x]
for intent in UNIQUE_INTENTS:
    print("Creating a dataset for " + intent)
    output_root = "output/" + intent
    seen_output = output_root + "/seen/"
    unseen_output = output_root + "/unseen/"
    os.system("python create-corpus.py " + INPUT_FOLDER + " " + seen_output + " " + unseen_output + " " + intent)