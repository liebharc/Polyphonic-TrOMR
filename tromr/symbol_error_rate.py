from staff2score import Staff2Score
import re
import editdistance

from split_merge_symbols import merge_symbols

def _load_semantic_file(semantic_path):
    with open(semantic_path, 'r') as f:
        return f.readlines()

def calc_symbol_error_rate_for_list(checkpoint_file_path, dataset, config):
    len_limit = 100
    if len(dataset) > len_limit:
        print(f"The dataset is too large ({len(dataset)} > {len_limit})")
        dataset = dataset[:len_limit]
    model = Staff2Score(config, checkpoint_file_path)
    all_sers = []
    i = 0
    total = len(dataset)
    for sample in dataset:
        img_path, semantic_path = sample.strip().split(",")
        expected = re.split("\s+", _load_semantic_file(semantic_path)[0].strip())
        predrhythms, predpitchs, predlifts = model.predict(img_path)
        actual = merge_symbols(predrhythms, predpitchs, predlifts)[0].split("+")
        distance = editdistance.eval(expected, actual)
        ser = distance / len(expected)
        all_sers.append(ser)
        ser = round(100 * ser)
        ser_avg = round(100 * sum(all_sers) / len(all_sers))
        i += 1
        percentage = round(i / total * 100)
        print(f"Progress: {percentage}%, SER: {ser}%, SER avg: {ser_avg}%")
    ser_avg = round(100 * sum(all_sers) / len(all_sers))
    print(f"Done, SER avg: {ser_avg}")

    result_file = checkpoint_file_path.replace(".pth", "_ser.txt")
    with open(result_file, 'w') as f:
        f.write(f"SER avg: {ser_avg}\n")