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


def read_file(input_file):
    with open(input_file, "r", encoding="utf-8") as file:
        read_file = file.read()

    read_file = read_file.lower().split('\n')
    return [line for line in read_file if line] # remove empty lines


def get_loader(model, word2vec, text_file, intent_file, batch_size):
    intent2int = model.intent2int
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


def write_alpha_file(model, file_name):
    int2intent = model.int2intent
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
    for i in range(model.c):
        omega_name = "Omega" + str(i) + str(0)
        if hasattr(model, omega_name):
            print("Writing file for omega " + str(i) + " at " + str(output_dir))
            for b in range(model.B):
                file_name = output_dir + "alpha" + str(i) + "omega" + str(b) + ".csv"
                with open(file_name, "w", newline="") as f:
                    writer = csv.writer(f)
                    omega = model.get_omega(i, b)
                    for j in range(model.hidden_dim):
                        row = []
                        for k in range(model.hidden_dim):
                            row.append(omega[j][k].item())
                        writer.writerow(row)



