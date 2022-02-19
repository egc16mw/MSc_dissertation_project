import sys, codecs, os

TRAINING_TEXT = "train_text.txt"
TRAINING_INTENT = "train_intent.txt"
VALIDATION_TEXT = "valid_text.txt"
VALIDATION_INTENT = "valid_intent.txt"
TESTING_TEXT = "test_text.txt"
TESTING_INTENT = "test_intent.txt"

ENCODING_METHOD = "utf-8"

if len(sys.argv) != 5:
    print("Usage: python INPUT_FOLDER SEEN_OUTPUT_FOLDER UNSEEN_OUTPUT_FOLDER UNSEEN_INTENT")
    exit(0)

INPUT_FOLDER = sys.argv[1]
SEEN_OUTPUT_FOLDER = sys.argv[2]
UNSEEN_OUTPUT_FOLDER = sys.argv[3]
IGNORED_INTENT = sys.argv[4]
print("Ignoring Intent: " + IGNORED_INTENT)

#############################################################################
######################## VALIDATING DATA IS CORRECT #########################
#############################################################################
def read_files(text_file, intent_file, ignored_intents=None):
    if ignored_intents is None:
        ignored_intents = []

    seen_output_text, seen_output_intent = [],[]
    unseen_output_text, unseen_output_intent = [], []
    with codecs.open(text_file, "r", encoding=ENCODING_METHOD) as file:
        read_text = file.read()
    with codecs.open(intent_file, "r", encoding=ENCODING_METHOD) as file:
        read_intent = file.read()

    read_text = read_text.split('\n')
    read_intent = read_intent.split('\n')
    for text, intent in zip(read_text, read_intent):
        # Filter out empty lines
        if not text:
            if intent:
                raise Exception("Text line is empty while intent line is not")
        elif not intent:
            if text:
                raise Exception("Intent line is empty while text line is not")
        elif intent in ignored_intents:
            unseen_output_text.append(text)
            unseen_output_intent.append(intent)
        else:
            seen_output_text.append(text)
            seen_output_intent.append(intent)

    assert len(seen_output_text) == len(seen_output_intent), "Text and Intent lengths don't match."
    assert len(unseen_output_text) == len(unseen_output_intent), "Text and Intent lengths don't match."
    return seen_output_text, seen_output_intent, unseen_output_text, unseen_output_intent

def write_file(file, data):
    with codecs.open(file, "w", encoding=ENCODING_METHOD) as f:
        for line in data:
            if line:
                f.write(line + "\n")


seen_training_text, seen_training_intents, unseen_training_text, unseen_training_intents = read_files(INPUT_FOLDER + TRAINING_TEXT, INPUT_FOLDER + TRAINING_INTENT, [IGNORED_INTENT])
seen_valid_text, seen_valid_intents, unseen_valid_text, unseen_valid_intents = read_files(INPUT_FOLDER + VALIDATION_TEXT, INPUT_FOLDER + VALIDATION_INTENT, [IGNORED_INTENT])
seen_test_text, seen_test_intents, unseen_test_text, unseen_test_intents = read_files(INPUT_FOLDER + TESTING_TEXT, INPUT_FOLDER + TESTING_INTENT, [IGNORED_INTENT])

if not os.path.exists(SEEN_OUTPUT_FOLDER):
    print("Creating directory: " + str(SEEN_OUTPUT_FOLDER))
    os.makedirs(SEEN_OUTPUT_FOLDER)

write_file(SEEN_OUTPUT_FOLDER + TRAINING_TEXT, seen_training_text)
write_file(SEEN_OUTPUT_FOLDER + TRAINING_INTENT, seen_training_intents)
write_file(SEEN_OUTPUT_FOLDER + VALIDATION_TEXT, seen_valid_text)
write_file(SEEN_OUTPUT_FOLDER + VALIDATION_INTENT, seen_valid_intents)
write_file(SEEN_OUTPUT_FOLDER + TESTING_TEXT, seen_test_text)
write_file(SEEN_OUTPUT_FOLDER + TESTING_INTENT, seen_test_intents)

os.system("python create-intent-distribution.py " + SEEN_OUTPUT_FOLDER + " " + SEEN_OUTPUT_FOLDER + "/overview.csv")

if not os.path.exists(UNSEEN_OUTPUT_FOLDER):
    print("Creating directory: " + str(UNSEEN_OUTPUT_FOLDER))
    os.makedirs(UNSEEN_OUTPUT_FOLDER)

write_file(UNSEEN_OUTPUT_FOLDER + TRAINING_TEXT, unseen_training_text)
write_file(UNSEEN_OUTPUT_FOLDER + TRAINING_INTENT, unseen_training_intents)
write_file(UNSEEN_OUTPUT_FOLDER + VALIDATION_TEXT, unseen_valid_text)
write_file(UNSEEN_OUTPUT_FOLDER + VALIDATION_INTENT, unseen_valid_intents)
write_file(UNSEEN_OUTPUT_FOLDER + TESTING_TEXT, unseen_test_text)
write_file(UNSEEN_OUTPUT_FOLDER + TESTING_INTENT, unseen_test_intents)

os.system("python create-intent-distribution.py " + UNSEEN_OUTPUT_FOLDER + " " + UNSEEN_OUTPUT_FOLDER + "/overview.csv")
