import interleaving as il

import numpy as np
import os
from collections import defaultdict
import matplotlib.pyplot as plot
import pandas as pd
import datetime
import glob
import yaml
import csv
from multiprocessing import Pool
import multiprocessing as multi


ITERATION = 25
ITER_PER_QUERY = 10000
EVALUATION_SPAN = 100
TOPK = 10
CLICK_PROBS = {
    "perfect": [0.0, 0.5, 1.0],
    "navigational": [0.05, 0.5, 0.95],
    "informational": [0.4, 0.7, 0.9],
    "random": [0.5, 0.5, 0.5]
}
STOP_PROBS = {
    "perfect": [0.0, 0.0, 0.0],
    "navigational": [0.2, 0.5, 0.9],
    "informational": [0.1, 0.3, 0.5],
    "random": [0.0, 0.0, 0.0]
}

SETTINGS = glob.glob("./settings/*.yaml")
METHODS = [il.TeamDraft, il.PairwisePreference]

OUTPUT_FILEPATH_PREFIX = os.path.join("./results/",
                                      str(datetime.datetime.now())
                                      .replace(':', '.')
                                      .replace(' ', '_'))

def create_user_types():
    user_types = {}
    for usertype in CLICK_PROBS:
        user_types[usertype] = il.simulation.User(
            click_probs=CLICK_PROBS[usertype],
            stop_probs=STOP_PROBS[usertype]
        )
    return user_types

def main():
    usertypes = create_user_types()
    args = []
    for setting_filepath in SETTINGS:
        setting = yaml.load(open(setting_filepath))
        folderpaths = glob.glob(setting["folderpath"])
        for folderpath in folderpaths:
            filepaths = glob.glob(folderpath + "/test*.txt")
            foldname = os.path.basename(folderpath)
            args.append((usertypes, setting, filepaths, foldname))

    n_cores = multi.cpu_count()
    p = Pool(n_cores)
    p.map(parallel_process, args)


def parallel_process(arg):
    usertypes, setting, filepaths, foldname = arg
    sim = il.simulation.Simulator(filepaths, ITER_PER_QUERY, TOPK)
    for method in METHODS:
        print(setting["name"], foldname,
              method.__name__, datetime.datetime.now())
        simulation(setting, foldname, sim, usertypes, method)

def simulation(setting, foldname, sim, usertypes, method):
    ranker_names, rankers = generate_rankers(setting)
    ndcg_result = sim.ndcg(rankers, TOPK)

    output_filepath = OUTPUT_FILEPATH_PREFIX + str(id(sim)) + ".csv"
    for i in range(ITERATION):
        for usertype, user in usertypes.items():
            il_result = sim.evaluate(rankers, user, method)
            metadata = [
                usertype,
                setting["name"],
                foldname,
                method.__name__,
            ]
            evaluation(output_filepath, metadata, il_result, ndcg_result)

def evaluation(output_filepath, metadata, il_result, ndcg_result):
    lines = []
    upto = EVALUATION_SPAN
    while upto <= len(il_result):
        il_result_subset = il_result[:upto]
        error = il.simulation.Simulator.measure_error(il_result_subset,
                                                      ndcg_result)
        lines.append(metadata + [upto, error])
        upto += EVALUATION_SPAN

    with open(output_filepath, 'a') as f:
        writer = csv.writer(f)
        writer.writerows(lines)

def generate_rankers(setting):
    rankers = []
    ranker_names = []
    for ranker_name, idx in setting["rankers"].items():
        ranker_names.append(ranker_name)
        ranker = il.simulation.Ranker(lambda x, y=idx: x[y])
        rankers.append(ranker)
    return (ranker_names, rankers)


if __name__ == '__main__':
    main()
