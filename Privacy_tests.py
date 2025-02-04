#open generation with canary
import pandas as pd
from argparse import ArgumentParser
import os

parser = ArgumentParser()
parser.add_argument("--file_name", type=str, help="Name of the file to be tested")
args = parser.parse_args()

file_name = args.file_name

def canaryTest(data_path):
    split_path = data_path.replace(".feather","_split.feather")
    file_name = os.path.basename(data_path)
    #print(file_name)
    dataset = pd.read_feather(data_path)
    split_file = pd.read_feather(split_path)
    #gather all the lines in split_file where split is "test"
    test_data = split_file[split_file["split"] == "test"]

    #get _id values into a list
    test_ids = test_data["_id"].tolist()

    #filter f to only keep rows where _id is in test_ids
    dataset = dataset[dataset["_id"].isin(test_ids)]
    #file_name_split is file_name with _split added before the extension
    file_name_no_extension = file_name.split(".")[0]
    file_name_split = f"{file_name_no_extension}_split.feather"
    splitted = file_name_no_extension.split("_")
    #print(splitted)
    privacy_budget = splitted[1]
    generated_dataset_size = splitted[2]
    canary = splitted[3]
    #cleaned = splitted[4]    
    #check how many samples contain the canary
    count = 0
    for i in range(len(dataset)):
        if canary in dataset["text"][i]:
            count+=1
    print(count)

print("Canary test")
canaryTest(file_name)
print("Done")
