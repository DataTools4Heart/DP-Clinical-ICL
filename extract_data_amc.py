import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from dateutil.relativedelta import relativedelta
from faker import Faker
import random
from transformers import set_seed
from sklearn.model_selection import train_test_split
from functools import partial
import numpy as np
from collections import Counter

# Constants
TRAIN_RATIO = 0.9
PAD_TOKEN = "<PAD>"
UNKNOWN_TOKEN = "<UNK>"
MIN_TARGET_COUNT = 10

ID_COLUMN = "_id"
TEXT_COLUMN = "text"
TARGET_COLUMN = "target"
SUBJECT_ID_COLUMN = "subject_id"

# Set the random seed
random.seed(42)
set_seed(42)

# Set the style for the plots
sns.set_style("whitegrid")

# Initialize the Faker library
fake = Faker()
Faker.seed(42)

def reformat_icd10(code: str, is_diag: bool) -> str:
    """Put a period in the right place for ICD-10 codes."""
    code = "".join(code.split("."))
    if not is_diag:
        return code
    return code[:3] + "." + code[3:]

def reformat_icd9(code: str, is_diag: bool) -> str:
    """Put a period in the right place for ICD-9 codes."""
    code = "".join(code.split("."))
    if is_diag:
        if code.startswith("E"):
            if len(code) > 4:
                return code[:4] + "." + code[4:]
        else:
            if len(code) > 3:
                return code[:3] + "." + code[3:]
    else:
        if len(code) > 2:
            return code[:2] + "." + code[2:]
    return code

def reformat_icd(code: str, version: int, is_diag: bool) -> str:
    """Format ICD code depending on version."""
    if version == 9:
        return reformat_icd9(code, is_diag)
    elif version == 10:
        return reformat_icd10(code, is_diag)
    else:
        raise ValueError("version must be 9 or 10")

def sort_by_indexes(lst, indexes, reverse=False):
    return [val for (_, val) in sorted(zip(indexes, lst), key=lambda x: x[0], reverse=reverse)]

def reformat_code_dataframe(row: pd.DataFrame, cols: list) -> pd.Series:
    """Takes a dataframe and a column name and returns a series with the column name and a list of codes."""
    out = dict()
    
    # Sort the first column and rearrange the second column accordingly
    sorted_indices = row[cols[0]].argsort()
    out[cols[0]] = sort_by_indexes(row[cols[0]], sorted_indices)
    out[cols[1]] = sort_by_indexes(row[cols[1]], sorted_indices)

    return pd.Series(out)

def parse_codes_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Parse the codes dataframe."""
    df = df.rename(columns={"hadm_id": ID_COLUMN, "subject_id": SUBJECT_ID_COLUMN})
    df = df.dropna(subset=["icd_code"])
    df = df.drop_duplicates(subset=[ID_COLUMN, "icd_code"])
    df = (
        df.groupby([SUBJECT_ID_COLUMN, ID_COLUMN, "icd_version"])
        .apply(partial(reformat_code_dataframe, cols=["icd_code","long_title"]))
        .reset_index()
    )
    return df

def parse_notes_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Parse the notes dataframe."""
    df = df.rename(
        columns={
            "hadm_id": ID_COLUMN,
            "subject_id": SUBJECT_ID_COLUMN,
            "text": TEXT_COLUMN,
        }
    )
    df = df.dropna(subset=[TEXT_COLUMN])
    df = df.drop_duplicates(subset=[ID_COLUMN, TEXT_COLUMN])
    return df

def filter_codes(df: pd.DataFrame, columns: list[str], min_count: int) -> pd.DataFrame:
    """Filter the codes dataframe to only include codes that appear at least min_count times."""
    for col in columns:
        code_counts = Counter([code for codes in df[col] for code in codes])
        codes_to_keep = set(
            code for code, count in code_counts.items() if count >= min_count
        )
        df[col] = df[col].apply(lambda x: [code for code in x if code in codes_to_keep])
        print(f"Number of unique codes in {col} before filtering: {len(code_counts)}")
        print(f"Number of unique codes in {col} after filtering: {len(codes_to_keep)}")
    return df

def main():
    # Load data
    mimic_notes = pd.read_csv("./data/physionet.org/files/mimic-iv-note/2.2/note/discharge.csv.gz", compression='gzip')
    mimic_proc = pd.read_csv("./data/physionet.org/files/mimiciv/2.2/hosp/procedures_icd.csv.gz", compression='gzip')
    mimic_diag = pd.read_csv("./data/physionet.org/files/mimiciv/2.2/hosp/diagnoses_icd.csv.gz", compression='gzip')
    procedures = pd.read_csv("./data/physionet.org/files/mimiciv/2.2/hosp/d_icd_procedures.csv.gz", compression='gzip')
    diagnoses = pd.read_csv("./data/physionet.org/files/mimiciv/2.2/hosp/d_icd_diagnoses.csv.gz", compression='gzip')

    # Merge procedures and diagnoses
    mimic_proc = mimic_proc.merge(procedures, how='inner', on=['icd_code','icd_version'])
    mimic_diag = mimic_diag.merge(diagnoses, how='inner', on=['icd_code','icd_version'])

    # Format ICD codes
    mimic_proc["icd_code"] = mimic_proc.apply(
        lambda row: reformat_icd(code=row["icd_code"], version=row["icd_version"], is_diag=False),
        axis=1,
    )
    mimic_diag["icd_code"] = mimic_diag.apply(
        lambda row: reformat_icd(code=row["icd_code"], version=row["icd_version"], is_diag=True),
        axis=1,
    )

    # Process codes and notes
    mimic_proc = parse_codes_dataframe(mimic_proc)
    mimic_diag = parse_codes_dataframe(mimic_diag)
    mimic_notes = parse_notes_dataframe(mimic_notes)

    # Filter for ICD-10 codes and merge
    mimic_proc_10 = mimic_proc[mimic_proc["icd_version"] == 10]
    mimic_proc_10 = mimic_proc_10.rename(columns={"icd_code": "icd10_proc"})
    mimic_diag_10 = mimic_diag[mimic_diag["icd_version"] == 10]
    mimic_diag_10 = mimic_diag_10.rename(columns={"icd_code": "icd10_diag"})

    # Merge notes with procedures and diagnoses
    mimiciv_10 = mimic_notes.merge(
        mimic_proc_10[[ID_COLUMN, "icd10_proc", "long_title"]], on=ID_COLUMN, how="inner"
    )
    mimiciv_10 = mimiciv_10.merge(
        mimic_diag_10[[ID_COLUMN, "icd10_diag", "long_title"]], on=ID_COLUMN, how="inner"
    )

    # Clean up data
    mimiciv_10 = mimiciv_10.dropna(subset=["icd10_proc", "icd10_diag"], how="all")
    mimiciv_10["icd10_proc"] = mimiciv_10["icd10_proc"].apply(
        lambda x: [] if x is np.nan else x
    )
    mimiciv_10["icd10_diag"] = mimiciv_10["icd10_diag"].apply(
        lambda x: [] if x is np.nan else x
    )

    # Filter codes and create target
    mimiciv_10 = filter_codes(mimiciv_10, ["icd10_proc", "icd10_diag"], MIN_TARGET_COUNT)
    mimiciv_10[TARGET_COLUMN] = mimiciv_10["icd10_proc"] + mimiciv_10["icd10_diag"]
    mimiciv_10["long_title"] = mimiciv_10["long_title_x"] + mimiciv_10["long_title_y"]
    
    # Remove empty targets and reset index
    mimiciv_10 = mimiciv_10[mimiciv_10[TARGET_COLUMN].apply(lambda x: len(x) > 0)]
    mimiciv_10 = mimiciv_10.reset_index(drop=True)

    # Save to disk
    mimiciv_10.to_feather("./data/mimiciv_icd10.feather")

if __name__ == "__main__":
    main() 