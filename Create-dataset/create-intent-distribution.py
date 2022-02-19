import sys, csv, os

if len(sys.argv) != 3:
    print("Usage: python SRCDIR OUTPUT_FILE")
    exit(0)

def read_file(input_file):
    with open(input_file, "r") as file:
        read_file = file.read()

    read_file = read_file.lower().split('\n')
    return [line for line in read_file if line] # remove empty lines

SRDIR = sys.argv[1]
OUTPUT_FILE = sys.argv[2]

# Input Files
TRAINING = SRDIR + "/" + "train_intent.txt"
VALIDATION = SRDIR + "/" + "valid_intent.txt"
TESTING = SRDIR + "/" + "test_intent.txt"

start_files = [TRAINING, VALIDATION, TESTING]
start_labels = ["Training", "Validation", "Testing"]
files, labels = [], []
for file, label in zip(start_files, start_labels):
    if os.path.isfile(file):
        files.append(read_file(file))
        labels.append(label)

unique_intents = set()

all_distributions = []

# Find Distribution of each file
for file in files:
    unique_intents.update(file)
    distribution = dict()
    for line in file:
        if line in distribution:
            distribution[line] += 1
        else:
            distribution[line] = 1
    all_distributions.append(distribution)

# Write file with the distribution
with open(OUTPUT_FILE, "w", newline="") as file:
    print("Writing intent distribution file called: ", OUTPUT_FILE)
    writer = csv.writer(file)
    writer.writerow(["Intents"] + labels)
    for intent in unique_intents:
        row = []
        for x in all_distributions:
            result = ""
            if intent in x:
                result = x[intent]
            row.append(result)
        writer.writerow([intent] + row)
