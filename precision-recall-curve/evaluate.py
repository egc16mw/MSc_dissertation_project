import torch
from torch.utils.data import TensorDataset, DataLoader

SEEN_INTENT = 0
UNSEEN_INTENT = 1


def calculate_score(model, output):
    return torch.sigmoid(torch.mm(output, model.a) + model.b_output)


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


def get_loader(model, word2vec, text_file, intent_file, batch_size):
    intent2int = model.intent2int
    text, intents = read_files(text_file, intent_file)
    maxlen = max([len(item.split()) for item in text])
    sentences = encode_sentence(word2vec, text, maxlen)
    temp_intents = []
    for x in intents:
        if x in intent2int:
            temp_intents.append(intent2int[x])
        else:
            temp_intents.append(100)
    intents = torch.tensor(temp_intents)

    data = TensorDataset(sentences, intents)
    return DataLoader(data, shuffle=True, batch_size=batch_size)


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


def predict(model, sentence, correct_label):
    entropy = torch.zeros(1)
    total = torch.zeros(1)

    score_list, probability_list = [], []

    # Calculate probability distribution and the most likely one
    with torch.no_grad():
        for x in range(model.c):
            output = model(sentence.squeeze(), x)
            score = calculate_score(model, output).squeeze()
            score_list.append(score)
            total += score

    for x in score_list:
        probability_list.append(x/total)

    for x in probability_list:
        entropy += x * torch.log(x)

    label = SEEN_INTENT if correct_label.item() in model.int2intent else UNSEEN_INTENT
    return -entropy.item(), label


def calculate_entropy(model, word2vec, text_file, intent_file):
    model.eval()
    model.training = False
    entropy_list, label_list = [], []
    loader = get_loader(model, word2vec, text_file, intent_file, 50)
    for texts, labels in loader:
        for sentence, intent in zip(texts, labels):
            entropy, label = predict(model, sentence, intent)
            entropy_list.append(entropy)
            label_list.append(label)
    return entropy_list, label_list
