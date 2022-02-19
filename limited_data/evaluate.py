from helper import calculate_score
import torch

def evaluate(model, data_loader):
    model.eval()
    total, correct = 0, 0

    with torch.no_grad():
        for text, labels in data_loader:
            current_batch_size = text.size(0)
            total += current_batch_size
            for i in range(current_batch_size):
                intent = int(labels[i].item())
                sentence = text[i]
                predicted_intent = predict(model, sentence)
                if predicted_intent == intent:
                    correct += 1
    return correct / total * 100


def predict(model, sentence):
    most_likely, highest_score = 0, 0

    # Calculate probability distribution and the most likely one
    with torch.no_grad():
        for x in range(model.c):
            output = model(sentence.squeeze(), x)
            score = calculate_score(model, output)
            if score > highest_score:
                highest_score = score
                most_likely = x

    return most_likely
