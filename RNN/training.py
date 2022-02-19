import csv, copy, torch
from datetime import datetime
from evaluate import evaluate
from training_details import TrainingDetails
torch.manual_seed(0)


"""
BEFORE CALLING THIS METHOD YOU MUST FREEZE THINGS THAT SHOULD BE FROZEN AND ADD PARAMETERS THAT SHOULD BE ADDED

Train the model given the information provided. Updates the model located in data.model
"""
def train(data, debug=False, accuracy_min_inc=None):
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
    lr = data.lr
    model = data.model
    print("Started training, starting time: ", start_time)
    if debug:
        for name, param in model.named_parameters():
            if param.requires_grad:
                print("Training Param: ", name)

    current_attempts, epoch, validation_accuracy, file_array = 0, 0, 0, []
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=data.weight_decay)


    train_loader = data.dataset.seen_train_loader
    columns = []
    while current_attempts <= data.max_attempts and data.max_epochs > epoch:
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
                output = model(x=text[i])
                probability = torch.softmax(output.squeeze(), dim=0)
                loss = -torch.log(probability[intent])
                if debug:
                    if not loss.requires_grad:
                        print("Loss don't have a grad function!!!")
                temp_loss += loss

            loss = temp_loss / current_batch_size
            loss.backward()
            optimizer.step()
            total_loss.append(loss)

        # Evaluating
        loss = torch.mean(torch.stack(total_loss))
        temp_file_array = [epoch, loss.item()]
        columns = ["Epoch", "Loss"]

        seen_accuracy = evaluate(model, data.dataset.seen_valid_loader)
        columns.append("Seen Validation Accuracy")
        temp_file_array.append(seen_accuracy)

        columns.append("Revert")

        # Smart training/stop
        validation_accuracy = seen_accuracy
        revert = False
        if validation_accuracy < previous_validation_accuracy + accuracy_min_inc:
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

    print("Writing training stats to file called: ", data.output.stats)
    with open(data.output.stats, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        for row in file_array:
            writer.writerow(row)

    data.model = model
    return seen_final_test_accuracy


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


