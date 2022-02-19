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
                output = model(sentence)
                probability = torch.softmax(output.squeeze(), dim=0)
                predicted_intent = torch.max(probability, dim=0)[1].item()
                if predicted_intent == intent:
                    correct += 1
    return correct / total * 100
