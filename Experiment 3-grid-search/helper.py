import numpy as np
import torch, csv
from torch.utils.data import TensorDataset, DataLoader

torch.manual_seed(0)
np.random.seed(0)


def get_word2vec(file):
    output_dict = dict()
    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], "float32")
            output_dict[word] = vector
    return output_dict


def read_files(text_file, intent_file):
    output_text, output_intent = [],[]
    with open(text_file, "r", encoding="utf-8") as file:
        read_text = file.read()
    with open(intent_file, "r", encoding="utf-8") as file:
        read_intent = file.read()

    read_text = read_text.lower().split('\n')
    read_intent = read_intent.lower().split('\n')
    for text, intent in zip(read_text, read_intent):
        # Filter out empty lines
        if not text:
            if intent:
                raise Exception("Text line is empty while intent line is not")
        else:
            output_text.append(text)
            output_intent.append(intent)

    assert len(output_text) == len(output_intent), "Text and Intent lengths don't match."
    return output_text, output_intent


def get_loader(word2vec, text_file, intent_file, intent2int, batch_size):
    text, intents = read_files(text_file, intent_file)
    maxlen = max([len(item.split()) for item in text])
    sentences = encode_sentence(word2vec, text, maxlen)
    intents = torch.tensor([intent2int[x] for x in intents])

    data = TensorDataset(sentences, intents)
    return DataLoader(data, shuffle=True, batch_size=batch_size)


def encode_sentence(word2vec, text, maxlen):
    average_vector = 0
    for word in word2vec.keys():
        average_vector += word2vec[word]
    average_vector = average_vector / len(word2vec)
    average_vector = torch.from_numpy(average_vector)
    sentences = []
    WORD_DIMENSIONS = word2vec[str(list(word2vec)[0])].shape[0]
    tensor = torch.zeros(WORD_DIMENSIONS)
    for i in range(WORD_DIMENSIONS):
        tensor[i] = float("Inf")

    for sentence in text:
        encoded = []
        for word in sentence.split():
            try:
                encoded.append(torch.from_numpy(word2vec[word]))
            except KeyError:
                encoded.append(average_vector.float())
        encoded = add_padding(encoded, maxlen, tensor)
        encoded = torch.stack(encoded)
        sentences.append(encoded)
    return torch.stack(sentences)


def add_padding(sentence, maxlen, tensor):
    while len(sentence) < maxlen:
        sentence.append(tensor)
    return sentence


def calculate_score(model, output):
    return torch.sigmoid(torch.mm(output, model.a) + model.b_output)


def write_alpha_file(int2intent, model, file_name):
    output = []
    if not hasattr(model, "c"):
        raise Exception("C is undefined")

    temp = [""]
    for b in range(model.B):
        temp.append(int2intent[b])
    output.append(temp)

    for c in range(model.c):
        alpha = model.get_alpha(c)
        result = [int2intent[c]]
        for coordinate in alpha:
            result.append(coordinate.item())
        output.append(result)

    with open(file_name, "w", newline="") as f:
        print("Writing Alpha to file called: ", file_name)
        writer = csv.writer(f)
        for alpha in output:
            writer.writerow(alpha)


def write_omega_files(model, output_dir):
    for i in range(model.B, model.c):
        for b in range(model.B):
            file_name = output_dir + "alpha" + str(i) + "omega" + str(b) + ".csv"
            print("Writing file for omega at: ", file_name)
            with open(file_name, "w", newline="") as f:
                writer = csv.writer(f)
                omega = model.get_omega(i, b)
                for j in range(model.hidden_dim):
                    row = []
                    for k in range(model.hidden_dim):
                        row.append(omega[j][k].item())
                    writer.writerow(row)


# Experiment 3 Regularisation
def compute_equation58(model, modifier=1):
    total = 0
    # Equation 58
    for i in range(model.B, model.c):
        temp_norm_loss_array = []
        for j in range(model.B):
            omega = model.get_omega(i, j)
            omega = omega - torch.eye(model.hidden_dim, model.hidden_dim)
            temp_norm_loss_array.append(torch.norm(input=omega, p="fro"))
        norm_loss = torch.mean(torch.stack(temp_norm_loss_array)) * modifier
        total += norm_loss
    return total


def compute_equation54(model, seen_loader, modifier=1):
    for text, labels in seen_loader:
        total = 0
        current_batch_size = text.size(0)
        max_seen = 0
        max_unseen = 0
        for i in range(current_batch_size):
            for c in range(model.c):
                output = model(text[i], c)
                score = calculate_score(model, output)
                if not c < model.B:
                    max_unseen = max(max_unseen, score)
                else:
                    max_seen = max(max_seen, score)
            total += torch.log(max_seen / max_unseen)
        return -(total / current_batch_size) * modifier
