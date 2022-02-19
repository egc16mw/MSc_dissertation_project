import torch

from evaluate import evaluate
from helper import get_word2vec, get_loader
from intentSpace import IntentSpace

batch_size = 50


def changed_model_parameters(model_1, model_2):
    for name, parameters in model_2.named_parameters():

        value = None
        for n, p in model_1.named_parameters():
            if n == name:
                value = p

        if value is None:
            print("Couldn't find: ", name)
        else:
            are_equal = torch.all(torch.eq(value, parameters)).item()
            if not are_equal:
                print("Not equal: " + name + ", requires_grad: " + str(value.requires_grad))


model2 = torch.load("output1/model.pt")
model = torch.load("output3/model.pt")
word2vec = get_word2vec("input/glove.42B.300d.txt")
text_file = "input/Data-Seen/test_text.txt"
intent_file = "input/Data-Seen/test_intent.txt"
data_loader = get_loader(model, word2vec, text_file, intent_file, batch_size)
print("Seen Accuracy: " + str(evaluate(model=model, data_loader=data_loader)))
text_file = "input/Data-Unseen/test_text.txt"
intent_file = "input/Data-Unseen/test_intent.txt"
data_loader = get_loader(model, word2vec, text_file, intent_file, batch_size)
print("Unseen Accuracy: " + str(evaluate(model=model, data_loader=data_loader)))
changed_model_parameters(model2, model)
