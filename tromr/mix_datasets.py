
from typing import List
import numpy as np


def _calc_number_of_files_to_take(data_sources, number_of_files: int):
    files_to_take = [0 for _ in data_sources]
    while number_of_files > 0:
        total_weight = sum([s["weight"] for s in data_sources])
        number_of_files_per_source = [int(number_of_files * s["weight"] / total_weight) for s in data_sources]
        number_of_files_per_source[-1] = number_of_files - sum(number_of_files_per_source[:-1])

        max_available_ratios = []
        for i, source in enumerate(data_sources):
            max_available_ratios.append(min(source["len"] / number_of_files_per_source[i], 1))

        limiting_ratio = min(max_available_ratios)
        number_of_files_per_source = [int(limiting_ratio * n) for n in number_of_files_per_source]

        for i, n in enumerate(number_of_files_per_source):
            number_of_files -= n
            id = data_sources[i]["id"]
            files_to_take[id] += n
            data_sources[i]["len"] -= n
        
        data_sources = [s for s in data_sources if s["len"] > 0]
    return files_to_take

    
def _take_all_training_sets(indexes):
    train_index = []
    for index in indexes:
        train_index += index
    np.random.shuffle(train_index)
    return train_index

    
def mix_training_sets(data_sources: List[List[str]], weights: List[float], number_of_files: int):
    for data_source in data_sources:
        np.random.shuffle(data_source)
    if number_of_files < 0:
        return _take_all_training_sets(data_sources)
    number_of_files_per_index = _calc_number_of_files_to_take([{"len": len(index), "weight": weights[i], "id": i} for i, index in enumerate(data_sources)], number_of_files)
    mixed_source = []
    for i, data_source in enumerate(data_sources):
        mixed_source += data_source[:number_of_files_per_index[i]]
    np.random.shuffle(mixed_source)
    return mixed_source