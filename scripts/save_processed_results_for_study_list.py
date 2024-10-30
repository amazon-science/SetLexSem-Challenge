import pandas as pd
from tqdm import tqdm

from setlexsem.constants import HPS, STUDY2MODEL
from setlexsem.utils import save_processed_results

if __name__ == "__main__":
    STUDY_LIST = list(STUDY2MODEL.keys())
    print(f"The list of studies that will be processed :\n {STUDY_LIST}")

    results_all = pd.DataFrame()
    df_all = pd.DataFrame()
    for study_name in tqdm(STUDY_LIST):
        df, results = save_processed_results(study_name, hps=HPS)
        df_all = pd.concat([df_all, df])
        results_all = pd.concat([results_all, results])
