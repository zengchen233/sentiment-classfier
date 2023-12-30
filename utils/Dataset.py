import os
from datasets import load_from_disk, load_dataset


def get_dataset():
    global dataset
    if os.path.exists('./datasets'):
        dataset = load_from_disk('./datasets/')  # 从磁盘中加载imdb数据集
    else:
        dataset = load_dataset('imdb')
        dataset.save_to_disk('./datasets/')
    return dataset
