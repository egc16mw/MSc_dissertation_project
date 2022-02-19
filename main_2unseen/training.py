import csv, copy, torch
from datetime import datetime
from evaluate import evaluate
from helper import calculate_score, write_alpha_file, write_omega_files
from training_details import TrainingDetails

torch.manual_seed(0)


"""
BEFORE CALLING THIS METHOD YOU MUST FREEZE THINGS THAT SHOULD BE FROZEN AND ADD PARAMETERS THAT SHOULD BE ADDED

Train the model given the information provided. Updates the model located in data.model
"""
def train(data, seen, regularisation_function=None, debug=False, changing_condition=None, accuracy_min_inc=None, accuracy_thresold=None):
    '''
    Parameters
    ----------
    data: An instance of TrainingDetails
    seen: True if training seen intents, False otherwise (unseen)
    regularisation_function: A function to use for regularisation
    debug: Debug mode
    changing_condition: Specify if training W and alpha interleaved
    accuracy_min_inc: Minimum accuracy increase. Specify if using changing_condition
    accuracy_thresold: Accuracy threshold before reverting model. Should specifcy if training not using accuracy_min_inc

    Returns
    ----------
    seen_final_test_accuracy: Test accuracy for seen intents
    unseen_final_test_accuracy: Test accuracy for unseen intents (None if training seen)

    '''
    assert isinstance(data, TrainingDetails)
    # Setup parameters
    start_time = datetime.now()
    unseen = not seen
    using_regularisation = not (regularisation_function is None)
    lr = data.lr
    model = data.model
    increase_changing_condition = changing_condition
    unseen_valid_count = total_valid_count = unseen_accuracy = columns = seen_valid_count = unseen_final_test_accuracy = None
    epochs_without_improvement = 0  # Unseen intent accuracy on validation dataset is baseline
    max_unseen_accuracy = 0
    print("Started training, starting time: ", start_time)
    if debug:
        for name, param in model.named_parameters():
            if param.requires_grad:
                print("Training Param: ", name)

    current_attempts, epoch, validation_accuracy, file_array = 0, 0, 0, []
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=data.weight_decay)

    if unseen:
        train_loader = data.dataset.unseen_train_loader
        seen_valid_count = 0
        unseen_valid_count = 0
        for text, labels in data.dataset.seen_valid_loader:
            seen_valid_count += text.size(0)
        for text, labels in data.dataset.unseen_valid_loader:
            unseen_valid_count += text.size(0)
        total_valid_count = seen_valid_count + unseen_valid_count

        if debug:
            print("Seen Valid Count: " + str(seen_valid_count))
            print("Unseen Valid Count: " + str(unseen_valid_count))
    else:
        train_loader = data.dataset.seen_train_loader

    while current_attempts <= data.max_attempts and data.max_epochs > epoch and epochs_without_improvement < 50:
        epoch += 1
        model.check_finite()
        model.train()
        total_loss, total_regularisation = [], []
        previous_model = copy.deepcopy(model)
        previous_validation_accuracy = validation_accuracy
        for text, labels in train_loader:
            model.zero_grad()
            current_batch_size = text.size(0)
            temp_loss = 0
            for i in range(current_batch_size):
                intent = int(labels[i].item())
                total, correct_loss = 0, 0
                for c in range(model.c):
                    output = model(x=text[i], alpha_int=c, detach=c != intent)
                    total += calculate_score(model, output)
                    if c == intent:
                        correct_loss = calculate_score(model, output)
                loss = -torch.log(correct_loss / total)
                if debug:
                    if not loss.requires_grad:
                        print("Loss don't have a grad function!!!")

                temp_loss += loss
            if using_regularisation:
                regularisation = regularisation_function(data, model)
                if debug and not regularisation.requires_grad:
                    print("Regularisation don't have grad: ", regularisation)
                loss = temp_loss / current_batch_size + regularisation
                total_regularisation.append(regularisation)
            else:
                loss = temp_loss / current_batch_size
            loss.backward()
            optimizer.step()
            total_loss.append(loss)

        # Evaluating
        loss = torch.mean(torch.stack(total_loss))
        temp_file_array = [epoch, loss.item()]
        columns = ["Epoch", "Loss"]
        if not (regularisation_function is None):
            regularisation = torch.mean(torch.stack(total_regularisation))
            temp_file_array.append(regularisation.item())
            columns.append("Regularisation")

        seen_accuracy = evaluate(model, data.dataset.seen_valid_loader)
        columns.append("Seen Validation Accuracy")
        temp_file_array.append(seen_accuracy)

        if unseen:
            unseen_accuracy = evaluate(model, data.dataset.unseen_valid_loader)
            all_data_accuracy = (seen_valid_count * seen_accuracy + unseen_valid_count * unseen_accuracy) / total_valid_count
            columns.append("Unseen Validation Accuracy")
            temp_file_array.append(unseen_accuracy)
            columns.append("All Data Validation Accuracy")
            temp_file_array.append(all_data_accuracy)
            # Smart stop
            if unseen_accuracy > max_unseen_accuracy:
                epochs_without_improvement = 0
                max_unseen_accuracy = unseen_accuracy
            else:
                epochs_without_improvement += 1

        columns.append("Revert")

        # Smart training/stop
        validation_accuracy = unseen_accuracy if unseen else seen_accuracy
        revert = False
        if changing_condition is None:
            if validation_accuracy < (previous_validation_accuracy * accuracy_thresold):
                revert = True
        else:
            if (validation_accuracy < previous_validation_accuracy + accuracy_min_inc) and (epoch != increase_changing_condition + 1):
                revert = True

        if revert:
            current_attempts += 1
            model = previous_model
            validation_accuracy = previous_validation_accuracy
            lr *= data.learning_rate_decay
            optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=data.weight_decay)
            temp_file_array.append(True)
        else:
            temp_file_array.append(False)

        if not (changing_condition is None):
            # Add to file array
            columns.append("Training alpha/W")
            if model.is_alpha_frozen():
                temp_file_array.append("W")
            else:
                temp_file_array.append("Alpha")

            # Change what is being trained
            if epoch == changing_condition and epoch != increase_changing_condition + 1:
                if model.is_alpha_frozen():
                    model.freeze_W()
                    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=data.weight_decay)
                    changing_condition += increase_changing_condition
                else:
                    model.freeze_alpha()
                    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=data.weight_decay)
                    changing_condition += increase_changing_condition
        if epoch == 1:
            print_epoch(columns, columns)
        print_epoch(temp_file_array, columns)
        file_array.append(temp_file_array)

    # END OF TRAINING, WRITE FILES
    print("It took a total of {:.2f} minutes".format((datetime.now() - start_time).seconds / 60))
    print("Writing model to file called: ", data.output.model)
    torch.save(model, data.output.model)

    seen_final_valid_accuracy = evaluate(model, data.dataset.seen_valid_loader)
    seen_final_test_accuracy = evaluate(model, data.dataset.seen_test_loader)

    file_array.append(["Accuracy on Seen Validation set", "", seen_final_valid_accuracy])
    file_array.append(["Accuracy on Seen Test set", "", seen_final_test_accuracy])
    if unseen:
        unseen_final_valid_accuracy = evaluate(model, data.dataset.unseen_valid_loader)
        unseen_final_test_accuracy = evaluate(model, data.dataset.unseen_test_loader)
        file_array.append(["Accuracy on Unseen Validation set", "", unseen_final_valid_accuracy])
        file_array.append(["Accuracy on Unseen Test set", "", unseen_final_test_accuracy])

    print("Writing training stats to file called: ", data.output.stats)
    with open(data.output.stats, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        for row in file_array:
            writer.writerow(row)

    data.model = model
    write_alpha_file(model, data.output.alpha)
    write_omega_files(model, data.output.directory)
    return seen_final_test_accuracy, unseen_final_test_accuracy


def print_epoch(file_array, columns):
    '''
    Parameters
    ----------
    file_array: What should be printed
    columns: Columns (used to calculate formatting)
    '''
    row_width, new_file_array = [], []
    for row in columns:
        highest = max(len(row) + 3, 8)
        row_width.append(highest)
    row_format = "".join(["{:>" + str(width) + "}" + "{:>3}" for width in row_width])
    for x in file_array:
        try:
            decimal = False if x - int(x) == 0 else True
            x = round(x, 5) if decimal else x
        except ValueError:
            pass
        new_file_array.append(x)
        new_file_array.append("|")
    print(row_format.format(*new_file_array))


