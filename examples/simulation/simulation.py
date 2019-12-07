import interleaving as il
# Cython version
from interleaving.cmethods import Probabilistic
il.Probabilistic = Probabilistic

import numpy as np
import os
from collections import defaultdict
import matplotlib.pyplot as plot
import pandas as pd
import datetime
import glob
import yaml

ITERATION = 25
ITER_PER_QUERY = 5000
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
#METHODS = [il.TeamDraft]
METHODS = [il.Probabilistic]

def main():
    usertypes = { usertype: il.simulation.User(
        click_probs=CLICK_PROBS[usertype],
        stop_probs=STOP_PROBS[usertype]
    ) for usertype in CLICK_PROBS}
    for setting_filepath in SETTINGS:
        setting = yaml.load(open(setting_filepath))
        folderpaths = glob.glob(setting["folderpath"])
        for folderpath in folderpaths:
            filepaths = glob.glob(folderpath + "/test*.txt")
            foldname = os.path.basename(folderpath)
            sim = il.simulation.Simulator(filepaths, usertypes,
                ITER_PER_QUERY, TOPK)
            for method in METHODS:
                print(setting_filepath, foldname, method.__name__, datetime.datetime.now())
                simulation(setting, foldname, method, sim)

def simulation(setting, foldname, method, sim):
    ranker_names, rankers = generate_rankers(setting)

    for i in range(ITERATION):
        print(i, "/", ITERATION)
        res = sim.run(rankers, method)
        for usertype in res:
            df = pd.DataFrame(res[usertype])
            df.columns = ranker_names
            df.to_csv("./results/%s__%s__%s__%s__%03d.csv" % (
                usertype,
                setting["name"],
                foldname,
                method.__name__,
                i
            ))

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
