import torch, csv, sys, copy, os, math
from datetime import datetime
from evaluate import evaluate
from intentSpace import IntentSpace
from helper import get_word2vec, read_files, get_loader, calculate_score, write_alpha_file, compute_equation58, compute_equation54, write_omega_files
import numpy as np

# Experiment 3
# v 0.2
torch.manual_seed(0)
np.random.seed(0)

if len(sys.argv) != 6:
    print("Usage: SEEN_DATA_DIR UNSEEN_DATA_DIR INPUT_DIR INPUT_DIR/<WORDMODEL> OUTPUT_DIR")
    exit()

# TODO call main.run_experiments()

# Input Params
SEEN_DATA_DIR = sys.argv[1] + "/"
UNSEEN_DATA_DIR = sys.argv[2] + "/"
INPUT_DIR = sys.argv[3] + "/"
OUTPUT_DIR_BASE = sys.argv[5]
word2vec = get_word2vec(INPUT_DIR + sys.argv[4])
int2intent = np.load(INPUT_DIR + "intents2.npy", allow_pickle=True).item()
intent2int = {intent: index for index, intent in int2intent.items()}

# Learning Params
MAX_EPOCHS = 50
BATCH_SIZE = 50
WORD_DIMENSION = word2vec[str(list(word2vec)[0])].shape[0]
HIDDEN_DIM = WORD_DIMENSION
MAX_ATTEMPTS = 10
WEIGHT_DECAY = 10e-5
ACCURACY_THRESHOLD = 0.95
LEARNING_RATE_DECAY = 0.9
lr = 0.001 * 1000
EQUATION54_MODIFIER_LIST = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25]
EQUATION58_MODIFIER_LIST = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25]
highest_performance_num = 0
highest_performance_name = ""
experiment_counter = 0
TOTAL_EXPERIMENT_COUNTS = len(EQUATION54_MODIFIER_LIST) * len(EQUATION58_MODIFIER_LIST)
START_TIME = datetime.now()


# Input Data Files
SEEN_TRAIN_TEXT_FILE = SEEN_DATA_DIR + "train_text.txt"
SEEN_TRAIN_INTENT_FILE = SEEN_DATA_DIR + "train_intent.txt"

SEEN_VALID_TEXT_FILE = SEEN_DATA_DIR + "valid_text.txt"
SEEN_VALID_INTENT_FILE = SEEN_DATA_DIR + "valid_intent.txt"

SEEN_TEST_TEXT_FILE = SEEN_DATA_DIR + "test_text.txt"
SEEN_TEST_INTENT_FILE = SEEN_DATA_DIR + "test_intent.txt"

UNSEEN_TRAIN_TEXT_FILE = UNSEEN_DATA_DIR + "train_text.txt"
UNSEEN_TRAIN_INTENT_FILE = UNSEEN_DATA_DIR + "train_intent.txt"

UNSEEN_VALID_TEXT_FILE = UNSEEN_DATA_DIR + "valid_text.txt"
UNSEEN_VALID_INTENT_FILE = UNSEEN_DATA_DIR + "valid_intent.txt"

UNSEEN_TEST_TEXT_FILE = UNSEEN_DATA_DIR + "test_text.txt"
UNSEEN_TEST_INTENT_FILE = UNSEEN_DATA_DIR + "test_intent.txt"

# Get weighting between seen and unseen intents
temp, _ = read_files(SEEN_VALID_TEXT_FILE, SEEN_VALID_INTENT_FILE)
SEEN_VALID_COUNT = len(temp)
temp, _ = read_files(UNSEEN_VALID_TEXT_FILE, UNSEEN_VALID_INTENT_FILE)
UNSEEN_VALID_COUNT = len(temp)
TOTAL_VALID_COUNT = SEEN_VALID_COUNT + UNSEEN_VALID_COUNT

# Data Loaders
train_loader = get_loader(word2vec, UNSEEN_TRAIN_TEXT_FILE, UNSEEN_TRAIN_INTENT_FILE, intent2int, BATCH_SIZE)

seen_test_loader = get_loader(word2vec, SEEN_TEST_TEXT_FILE, SEEN_TEST_INTENT_FILE, intent2int, BATCH_SIZE)
seen_train_loader = get_loader(word2vec, SEEN_TRAIN_TEXT_FILE, SEEN_TRAIN_INTENT_FILE, intent2int, BATCH_SIZE)
seen_valid_loader = get_loader(word2vec, SEEN_VALID_TEXT_FILE, SEEN_VALID_INTENT_FILE, intent2int, BATCH_SIZE)

unseen_valid_loader = get_loader(word2vec, UNSEEN_VALID_TEXT_FILE, UNSEEN_VALID_INTENT_FILE, intent2int, BATCH_SIZE)
unseen_test_loader = get_loader(word2vec, UNSEEN_TEST_TEXT_FILE, UNSEEN_TEST_INTENT_FILE, intent2int, BATCH_SIZE)

for equation54_modifier in EQUATION54_MODIFIER_LIST:
    for equation58_modifier in EQUATION58_MODIFIER_LIST:
        no_improvement_counter = 0
        # Setup file structure for output
        experiment_name = "eq54_" + str(equation54_modifier) + "___eq58_" + str(equation58_modifier)
        output_dir = OUTPUT_DIR_BASE + "/" + experiment_name + "/"
        if not os.path.exists(output_dir):
            print("Creating directory: " + str(output_dir))
            os.makedirs(output_dir)
        output_model = output_dir + "model3.pt"
        output_stats = output_dir + "training_stats3.csv"

        experiment_time = datetime.now()
        model = torch.load(INPUT_DIR + "model2.pt")

        # Add Omega
        print("Freeze all parameters")
        for name, param in model.named_parameters():
            param.requires_grad = False

        model.add_omegas(6)

        # Training Setup
        current_attempts, epoch, validation_accuracy = 0, 0, 0
        file_array = []
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)

        # Training
        print("Epoch \t Equation 54 \t Equation 58 \t Total Loss \t Validation Accuracy Seen \t Validation Accuracy Unseen\t Accuracy All Data")
        while current_attempts <= MAX_ATTEMPTS and MAX_EPOCHS > epoch:
            model.check_finite()
            model.train()
            epoch += 1
            totalLoss, total_equation58, total_equation54 = [], [], []
            previous_model = copy.deepcopy(model)
            previous_validation_accuracy = validation_accuracy
            for text, labels in train_loader:
                model.zero_grad()
                temp_loss = 0
                current_batch_size = text.size(0)
                for i in range(current_batch_size):
                    intent = int(labels[i].item())
                    total, correct_loss = 0, 0
                    for c in range(model.c):
                        output = model(text[i], c, c != intent)
                        total += calculate_score(model, output)
                        if c == intent:
                            correct_loss = calculate_score(model, output)
                    loss = -torch.log(correct_loss / total)
                    temp_loss += loss

                equation54 = compute_equation54(model, seen_train_loader, equation54_modifier)
                equation58 = compute_equation58(model, equation58_modifier)
                loss = temp_loss / current_batch_size + equation58 + equation54
                loss.backward()
                optimizer.step()
                totalLoss.append(loss)
                total_equation54.append(equation54)
                total_equation58.append(equation58)


            # Evaluating
            loss = torch.mean(torch.stack(totalLoss))
            equation54 = torch.mean(torch.stack(total_equation54))
            equation58 = torch.mean(torch.stack(total_equation58))
            seen_accuracy = evaluate(model, seen_valid_loader)
            unseen_accuracy = evaluate(model, unseen_valid_loader)
            all_data_accuracy = (SEEN_VALID_COUNT * seen_accuracy + UNSEEN_VALID_COUNT * unseen_accuracy) / TOTAL_VALID_COUNT
            if all_data_accuracy > highest_performance_num:
                print("New high score: " + str(all_data_accuracy) + " and dir: " + output_dir)
                highest_performance_num = all_data_accuracy
                highest_performance_name = output_dir

            temp_file_array = [epoch, equation54.item(), equation58.item(), loss.item(), seen_accuracy, unseen_accuracy, all_data_accuracy]

            print("{:3}\t\t".format(epoch), end=" ")
            print("{:.5f}\t\t".format(equation54.item()), end=" ")
            print("{:.5f}\t\t".format(equation58.item()), end=" ")
            print("{:.5f}\t\t".format(loss.item()), end=" ")
            print("{:.2f}%\t\t\t\t".format(seen_accuracy), end=" ")
            print("{:.2f}%\t\t\t\t".format(unseen_accuracy), end=" ")
            print("{:.2f}%".format(all_data_accuracy))

            # Smart Training
            validation_accuracy = unseen_accuracy
            # Stop if no improvements
            if math.floor(previous_validation_accuracy * 100) >= math.floor(validation_accuracy * 100):
                no_improvement_counter += 1
            else:
                no_improvement_counter = 0 if no_improvement_counter == 0 else no_improvement_counter - 1

            if no_improvement_counter > 10:
                print("Exiting loop, no_improvement_counter is greater than " + str(no_improvement_counter))
                break

            # Smart reverting
            if validation_accuracy < (previous_validation_accuracy * ACCURACY_THRESHOLD):
                temp_file_array.append("Reverting to previous model")
                current_attempts += 1
                model = previous_model
                validation_accuracy = previous_validation_accuracy
                print("Reverting to previous model, attempt: " + str(current_attempts))
                lr *= LEARNING_RATE_DECAY
                optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
                print("New Learning Rate: ", lr)
            file_array.append(temp_file_array)

        # Writing Stats
        print("Experiment took a total of {:.2f} minutes".format((datetime.now() - experiment_time).seconds / 60))
        print("Writing model to file called: ", output_model)
        torch.save(model, output_model)

        seen_final_valid_accuracy = evaluate(model, seen_valid_loader)
        seen_final_test_accuracy = evaluate(model, seen_test_loader)

        unseen_final_valid_accuracy = evaluate(model, unseen_valid_loader)
        unseen_final_test_accuracy = evaluate(model, unseen_test_loader)
        file_array.append(["Accuracy on Seen Validation set", "", seen_final_valid_accuracy])
        file_array.append(["Accuracy on Seen Test set", "", seen_final_test_accuracy])
        file_array.append(["Accuracy on Unseen Validation set", "", unseen_final_valid_accuracy])
        file_array.append(["Accuracy on Unseen Test set", "", unseen_final_test_accuracy])
        file_array.append(["Equation54 Modifier", equation54_modifier])
        file_array.append(["Equation58 Modifier", equation58_modifier])

        print("Writing training stats to file called: ", output_stats)
        with open(output_stats, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Epoch", "Equation54", "Equation58", "Final Loss", "Seen Validation Accuracy", "Unseen Validation Accuracy", "Accuracy All Data", "Additional Info"])
            for row in file_array:
                writer.writerow(row)
        write_alpha_file(int2intent, model, output_dir + "alpha.csv")
        write_omega_files(model, output_dir)
        experiment_counter += 1
        print("Finished Experiment " + str(experiment_counter) + "/" + str(TOTAL_EXPERIMENT_COUNTS))
print("High score: " + str(highest_performance_num) + ", location: " + highest_performance_name)
print("It took a total of {:.2f} minutes".format((datetime.now() - START_TIME).seconds / 60))
