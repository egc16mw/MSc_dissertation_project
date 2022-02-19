import os

from experiments import run_experiments

datasets = ["AddToPlaylist", "BookRestaurant", "GetWeather", "PlayMusic", "RateBook", "SearchCreativeWork", "SearchScreeningEvent"]
input_root = "input/"
output_root = "output/"


def create_path(base, seen):
    '''
    Parameters
    ----------
    base: base path
    seen: seen or unseen intent

    Returns
    -------

    '''
    if seen:
        return input_root + base + "/seen"
    else:
        return input_root + base + "/unseen"


# Some validation to avoid wasting time
for data in datasets:
    if not os.path.exists(create_path(base=data, seen=True)):
        print("Couldn't find directory " + create_path(base=data, seen=True))
        exit()
    if not os.path.exists(create_path(base=data, seen=False)):
        print("Couldn't find directory " + create_path(base=data, seen=False))
        exit()

data = datasets[0]
seen_dir = create_path(base=data, seen=True)
unseen_dir = create_path(base=data, seen=False)
output_dir = output_root + data
run_experiments(word_vector_path="input/glove.42B.300d.txt", seen_input=seen_dir, unseen_input=unseen_dir, output_dir=output_dir)
