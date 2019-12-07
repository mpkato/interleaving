import argparse
import csv
import glob
import pandas as pd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_filepath_prefix')
    args = parser.parse_args()

    result_filepaths = glob.glob("{}*".format(args.input_filepath_prefix))
    all_df = None
    for result_filepath in result_filepaths:
        df = pd.read_csv(result_filepath, header=None)
        df.columns = ['User', 'Data', 'Fold', 'Method', 'Impression', 'Error']
        all_df = df if all_df is None else pd.concat([all_df, df])

    all_df['Dataset'] = all_df.Data.replace({
        '2003_td_dataset': 'LETOR 3.0',
        '2004_td_dataset': 'LETOR 3.0',
        '2003_hp_dataset': 'LETOR 3.0',
        '2004_hp_dataset': 'LETOR 3.0',
        '2003_np_dataset': 'LETOR 3.0',
        '2004_np_dataset': 'LETOR 3.0',
    })
    all_df = all_df.groupby(['User', 'Dataset', 'Method', 'Impression']).mean()
    all_df = all_df.reset_index()
    all_df = all_df[all_df.Impression == 10000]
    print(all_df)

if __name__ == '__main__':
    main()
