#import everything
from transformers import set_seed
from tqdm import tqdm
import random
import json
import time
from sentence_transformers import SentenceTransformer
import ollama
from argparse import ArgumentParser
import pandas as pd
import json
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
import os
from datetime import datetime

#set the seed
set_seed(17)

def get_settings_id(args):
    """Generate a unique identifier for the current settings"""
    if args.nonprivate:
        privacy_setting = "nonprivate"
    else:
        privacy_setting = f"private_eps_{'-'.join(map(str, args.epsilons))}"
    
    return f"{args.model_name}_n{args.generated_dataset_size}_s{args.num_shots}_t{args.temperature}_{privacy_setting}"

def create_output_directory(args):
    """Create and return the output directory path based on current settings and timestamp"""
    # Create base directory if it doesn't exist
    base_dir = "./data/generated"
    os.makedirs(base_dir, exist_ok=True)
    
    # Create settings directory
    settings_id = get_settings_id(args)
    settings_dir = os.path.join(base_dir, settings_id)
    os.makedirs(settings_dir, exist_ok=True)
    
    # Create timestamp directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(settings_dir, timestamp)
    os.makedirs(output_dir, exist_ok=True)
    
    return output_dir

#set the arguments
parser = ArgumentParser()
parser.add_argument("--num_shots",type=int,default=5,help="Number of few-shot examples to use")
parser.add_argument("--num_c",type=int,default=5,help="Number of classes to generate samples for")
#parser.add_argument("--dataset_size",type=int,default=50000,help="Number of samples to generate for each class")
#parser.add_argument("--num_partitions",type=int,default=500,help="Number of partitions to split the training data into")
parser.add_argument("--canary",type=str,default=None,help="Canary string to use")
parser.add_argument("--canary_n",type=int,default=1,help="Number of canary strings to use")
parser.add_argument("--generated_dataset_size",type=int,default=100,help="Number of samples to generate for each class")
parser.add_argument("--nonprivate",action="store_true",help="Use non-private ESA")
parser.add_argument("--long_titles",action="store_true",help="Use long titles (actual disease names)")
parser.add_argument("--epsilons",nargs="+",type=float,default=[1,3,8],help="Epsilons to use for private ESA")
parser.add_argument("--ensemble1",type=int,default=5,help="Number of samples to average for the few-shot examples")
parser.add_argument("--ensemble2",type=int,default=5,help="Number of samples to average for the zero-shot examples")
parser.add_argument("--prompt_index",type=int,default=0,help="Index of the prompt to use")
parser.add_argument("--prompt",type=str,default=None,help="Prompt to use for text generation")
parser.add_argument("--zero_shot_forced",type=int,default=0,help="Number of zero-shot examples to force the canary into")
parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for model generation")
parser.add_argument("--cardio",default=False,action="store_true",help="Use cardio codes")
parser.add_argument("--canary_rate",type=float,default=0,help="Rate of few-shot examples containing the canary")
parser.add_argument("--model_name",type=str,default="llama3.2",help="LLM to use for text generation")
parser.add_argument("--custom_dataset_path",type=str,default=None,help="Custom dataset to use")
args = parser.parse_args()

# Validate custom prompt if provided
if args.prompt is not None:
    prompt = args.prompt.strip()
    if not prompt.endswith("ICD10-CODES="):
        error_message = """
Custom prompt validation failed!

Your prompt must end with exactly 'ICD10-CODES=' (without quotes). This is required because:
1. The script needs to know where to insert the ICD-10 codes
2. The format must be exact (no extra spaces after the '=')
3. The codes will be inserted immediately after the '='

Your prompt ends with: '{}'

To fix this:
1. Check for any trailing spaces or newlines
2. Make sure 'ICD10-CODES=' is the last part of your prompt
3. Verify there are no extra characters after the '='

Example of correct prompt ending:
"... rest of your prompt text here ICD10-CODES="
""".format(prompt[-20:] if len(prompt) > 20 else prompt)
        raise ValueError(error_message)

# Create output directory for this run
output_dir = create_output_directory(args)

NUM_SHOTS = args.num_shots
NUM_C = args.num_c
#DATASET_SIZE = args.dataset_size
GENERATED_DATASET_SIZE = args.generated_dataset_size
NUM_PARTITIONS = GENERATED_DATASET_SIZE*args.ensemble2    #NUM_PARTITIONS needs to be at least GENERATED_DATASET_SIZE*ensemble2
CANARY = args.canary
CANARY_N = args.canary_n

EPSILONS = args.epsilons
#define start runtime
start = time.time()

prompts = ["Generate a clinical discharge summary of a patient who had the conditions and procedures described by the following codes ICD10-CODES= ",
"""Please generate a realistic and concise clinical discharge summary for a patient based on the following ICD-10 codes. Do not include the ICD-10 codes themselves in the report; instead, refer to the medical conditions they represent. Before writing the summary, internally create a logical and medically accurate timeline of the patient's diagnosis, treatment, and progress. Use standard medical abbreviations where appropriate to mirror real clinical documentation. Focus on essential clinical information, avoiding unnecessary explanations or verbosity. Ensure that the report accurately reflects the management of both common and rare diseases as applicable. Requirements:\

Use standard medical abbreviations (e.g., BP for blood pressure, HR for heart rate).
Keep the summary concise and focused on relevant clinical details.
Ensure the sequence of events and timing within the report make medical sense.
Cover a wide range of use cases, including rare diseases when specified.
Do not include any ICD-10 codes in the text of the report.
Format:

Admission Date: [Provide a realistic date]
Discharge Date: [Provide a realistic date]
Discharge Summary:
Reason for Admission: [Briefly state]
History of Present Illness: [Concise description]
Hospital Course: [Key interventions and patient response]
Discharge Plan: [Medications, follow-up, patient instructions]
ICD10-CODES= """,
"""
Please generate a realistic, concise, and professional clinical discharge summary for a patient based on the following ICD-10 codes. Do not include the ICD-10 codes themselves in the report; instead, reference the medical conditions they represent. Before composing the summary, internally develop a logical and medically accurate patient case, including the timeline of symptom onset, diagnosis, interventions, and outcomes. Do not include this internal planning in the final summary.

The discharge summary should:

Use clinical language with standard medical abbreviations (e.g., CHF for congestive heart failure, N/V for nausea and vomiting).
Be succinct, focusing on essential clinical information without unnecessary explanations.
Reflect a coherent and medically plausible sequence of events with appropriate timing.
Represent a wide range of cases, including both common and rare conditions when specified.
Mimic the style and tone of actual clinical documentation used among healthcare professionals.
Format:

Patient Identification:
Name: [Use initials only, e.g., J.D.]
Age/Gender: [e.g., 45-year-old male]
Admission Date: [Realistic date]
Discharge Date: [Realistic date]
Admitting Diagnosis: [Primary reason for admission]
Discharge Diagnoses:
Primary: [State main condition]
Secondary: [List comorbidities or complications]
Hospital Course:
[Summarize key diagnostic findings, treatments, and patient response]
Discharge Instructions:
Medications: [List with dosages]
Follow-Up: [Appointments, referrals]
Activity: [Restrictions or recommendations]
Diet: [Instructions if applicable]
Warnings: [Symptoms that require immediate attention]
Additional Requirements:

Exclude any patient-identifiable information beyond initials.
Do not include the internal case planning or timeline in the summary.
Ensure medical accuracy and plausibility in terms of disease progression and treatment.
Use appropriate medical terminology relevant to the conditions.
ICD10-CODES= """,
"""[ADD HERE YOUR OWN PROMPT]
ICD10-CODES= """
]

test_prompt = f"Generate a clinical discharge summary of {CANARY} a patient who had the conditions and procedures described by the following codes ICD10-CODES= "

reverse_index = {prompt: i for i,prompt in enumerate(prompts)}

#set the seed for reproducibility
set_seed(42)

#load the dataset
dataset = pd.read_feather("./data/mimiciv_icd10.feather") if not args.custom_dataset_path else pd.read_feather(args.custom_dataset_path)
if not args.custom_dataset_path:
    dataset_split = pd.read_feather("./data/mimiciv_icd10_split.feather")
    dataset = dataset[dataset_split["split"]=="test"]
print("dataset length",len(dataset))
DATASET_SIZE = len(dataset)
#reset indices
dataset = dataset.reset_index(drop=True)

#open "./data/cardio_codes.csv" to get the list of cardio codes
if args.cardio:
    cardio_codes = pd.read_csv("./data/cardio_codes.csv")
    cardio_codes_list = cardio_codes.squeeze().tolist()
    #filter df to only keep rows where at least one of the cardio codes is present in the target column
    to_keep = []
    for i in range(len(dataset)):
        if any(code in dataset["target"][i] for code in cardio_codes_list):
            to_keep.append(i)

    dataset = dataset.iloc[to_keep]

    #reset indices
    dataset = dataset.reset_index(drop=True)
    print(len(dataset), "dataset length after removing non-cardio codes")
    indexes = range(len(dataset))

#if there's a canary string, add it to the dataset as many times as specified
if CANARY:
    for i in range(CANARY_N):
        sampled_index = random.choice(indexes)
        #remove sampled_index from the indexes
        indexes = [x for x in indexes if x!=sampled_index]
        dataset['text'][i] = CANARY + " " + dataset['text'][i]
        

#get the indices of the training samples
train_indices = list(range(len(dataset)))
#get the indices of the training samples shuffled
shuffled_indices = random.sample(train_indices, len(train_indices))
#get the shuffled indices in partitions of the same size
partition_size = NUM_SHOTS
#use all partitions available
final_outputs = [] 
zero_shot_samples = []
few_shot_samples = []
model = SentenceTransformer('all-MiniLM-L6-v2')
zero_shots_embeddings = []
few_shots_embeddings = []
labels = []
for j,k in tqdm(enumerate(range(0,NUM_PARTITIONS*partition_size,partition_size))):
    print(j,k)
    print(f"Processing partition {j}")
    print(f"Processing indices {k} to {k+partition_size}")
    
    #take a partition of the shuffled indices of size PARTITION_SIZE
    partition = shuffled_indices[k:k+partition_size]
    partitions = [shuffled_indices[i:i+partition_size] for i in range(0, len(shuffled_indices), partition_size)]
    
    #sample a disease label for the case to generate
    sampled_index = random.randint(0,len(dataset))
    sample_disease = dataset['target'][sampled_index].tolist()
    sample_proc = dataset['icd10_proc'][sampled_index].tolist()
    sample_diag = dataset['icd10_diag'][sampled_index].tolist()
    sample_disease_description = dataset['long_title'][sampled_index].tolist()
    #link the diseases codes to the descriptions
    coupled_diseases = []
    for i in range(len(sample_disease)):
        coupled_diseases.append(f'{sample_disease[i]}:{sample_disease_description[i]}')
    few_shot_prompt = prompts[args.prompt_index] if args.prompt is None else args.prompt
    zero_shot_prompt = prompts[args.prompt_index] if args.prompt is None else args.prompt
    diseases = sample_disease if not args.long_titles else coupled_diseases

    

    #insert the sample_disease in the prompt after the = sign
    few_shot_prompt = few_shot_prompt.replace("ICD10-CODES=",f"ICD10-CODES={diseases}, it is extremely important that you do not include such codes in the report, but you should include the corresponding conditions. Also do not leave anything blank, the reports are anonymized, but you should invent the missing information like names and ages such that they are realistic and coherent with conditions and procedures.\n\n")
    zero_shot_prompt = zero_shot_prompt.replace("ICD10-CODES=",f"ICD10-CODES={diseases}, it is extremely important that you do not include such codes in the report, but you should include the corresponding conditions.\n\n")
    test_prompt_complete = test_prompt.replace("ICD10-CODES=",f"ICD10-CODES={diseases}, it is extremely important that you do not include such codes in the report, but you should include the corresponding conditions.\n\n")
    # Generate NUM_C different zero-shot samples for each case
    current_zero_shot_samples = []
    current_zero_shot_embeddings = []
    for c in range(NUM_C):
        prompt = test_prompt_complete if c < args.zero_shot_forced else zero_shot_prompt
        zero_shot_response = ollama.chat(model=args.model_name, messages=[
        {
            'role': 'user',
            'content': prompt,
        },
        ], options={"temperature": args.temperature, "top_p": 0.95})
        zero_shot_out = zero_shot_response['message']['content']
        zero_shot_output_sample = {
            "input": prompt,
            "prediction": zero_shot_out,
            "label": sample_disease,
            "icd10_diag": sample_diag,
            "icd10_proc": sample_proc,
            "icd10_long_title": sample_disease_description
        }
        zero_shot_output_embedding = model.encode(zero_shot_out)
        current_zero_shot_samples.append(zero_shot_output_sample)
        current_zero_shot_embeddings.append(zero_shot_output_embedding)
    zero_shot_samples.extend(current_zero_shot_samples)
    zero_shots_embeddings.extend(current_zero_shot_embeddings)
    
    few_shots_examples = [dataset["text"][i] for i in partition]
    few_shot_text = "\n".join(few_shots_examples)+"\n"

    # Set the few-shot prompt text
    prompt = few_shot_prompt+few_shot_text
    
    response = ollama.chat(model=args.model_name, messages=[
    {
        'role': 'user',
        'content': prompt,
    },
    ], options={"temperature": args.temperature, "top_p": 0.95})
    out = response['message']['content']
    output_sample = {
        "input": prompt,
        "prediction": CANARY + "\n" + out if i//partition_size < args.canary_n and args.canary else out,
        "label": sample_disease,
        "icd10_diag": sample_diag,
        "icd10_proc": sample_proc,
        "icd10_long_title": sample_disease_description
    }
    few_shot_samples.append(output_sample)
    few_shot_output_embedding = model.encode(out)
    few_shots_embeddings.append(few_shot_output_embedding)

generated_data_path = output_dir
run_id = f"{GENERATED_DATASET_SIZE}_{CANARY}_{CANARY_N}_ensemble1_{args.ensemble1}_ensemble2_{args.ensemble2}_{args.nonprivate}"

# Save embeddings and samples in the base generated directory
zero_shot_samples_path = os.path.join("./data/generated", "zero_shot_samples.json")
few_shot_samples_path = os.path.join("./data/generated", "few_shot_samples.json")
zero_shot_embeddings_path = os.path.join("./data/generated", "zero_shot_embeddings.csv")
few_shot_embeddings_path = os.path.join("./data/generated", "few_shot_embeddings.csv")

#save the zero shot samples to a json file
with open(zero_shot_samples_path,"w") as f:
    json.dump(zero_shot_samples,f,indent=4)    

#save the zero shot embeddings to a csv file
print(f"Length of zero_shots_embeddings before saving: {len(zero_shots_embeddings)}")
pd.DataFrame(zero_shots_embeddings).to_csv(zero_shot_embeddings_path, index=False)

#save generated samples to a json file
with open(few_shot_samples_path,"w") as f:
    json.dump(few_shot_samples,f,indent=4)

#save the few shot embeddings to a csv file
pd.DataFrame(few_shots_embeddings).to_csv(few_shot_embeddings_path, index=False)

path = './data/'

# After reading the CSV
print(f"Number of rows in loaded CSV: {len(pd.read_csv(zero_shot_embeddings_path))}")

with open(zero_shot_samples_path, 'r') as f:
  zero_pred_f = json.load(f)
zero_pred = []
for i in range(len(zero_pred_f)):
  zero_pred.append([zero_pred_f[i]['prediction'], zero_pred_f[i]['label'], zero_pred_f[i]['icd10_diag'], zero_pred_f[i]['icd10_proc'], zero_pred_f[i]['icd10_long_title']])
with open(few_shot_samples_path, 'r') as f:
  few_pred_f = json.load(f)
few_pred = []
for i in range(len(few_pred_f)):
  few_pred.append([few_pred_f[i]['prediction'], few_pred_f[i]['label'], few_pred_f[i]['icd10_diag'], few_pred_f[i]['icd10_proc'], few_pred_f[i]['icd10_long_title']])

if args.nonprivate:
    indexs = []
    zero_shot_embedding  = pd.read_csv(zero_shot_embeddings_path).values
    few_shot_embedding  = pd.read_csv(few_shot_embeddings_path).values
    print(zero_shot_embedding.shape, few_shot_embedding.shape)

    ensemble1 = NUM_C
    ensemble2 = args.ensemble2


    for i in range(GENERATED_DATASET_SIZE):
        # get mean embedding of df2
        curr_emb = few_shot_embedding[i*ensemble2:(i+1)*ensemble2]
        curr_emb_sum = np.sum(curr_emb, axis=0, keepdims=True)
        mean_emb = curr_emb_sum/ensemble1 # (1, 1536) 1 data, 1536 features
        start_idx = i * NUM_C
        end_idx = (i + 1) * NUM_C
        
        dist = cosine_similarity(mean_emb, zero_shot_embedding[start_idx:end_idx])
        indexs.append(np.argmax(dist)+i*ensemble1)
    
    # get predictions from json file
    final_pred = []
    for index in indexs:
        final_pred.append(zero_pred[index])

    df = pd.DataFrame(final_pred, columns=["text","target","icd10_diag","icd10_proc","icd10_long_title"])
    #add column named _id with sequential numbers
    df["_id"] = [i for i in range (len(df))]
    #add column named num_words with the number of words in the text
    df["num_words"] = [len(x.split()) for x in df["text"]]
    #add column named num_targets with the number of targets in the target
    df["num_targets"] = [len(x) for x in df["target"]]
    #generate a split file assignin all the samples to the test set
    split = ["test"]*len(df)
    ids = [df["_id"][i] for i in range(len(df))]
    split_df = pd.DataFrame({"_id":ids,"split":split})

    #add 10 random samples from dataset to the generated dataset for train split and 10 for val split (necessary workaround for evaluation through automatical-medical-coding evaluation)
    for i in range(10):
        index = random.randint(0,len(dataset))
        df = df._append({"text":dataset["text"][index],"target":dataset["target"][index],"icd10_diag":dataset["icd10_diag"][index],"icd10_proc":dataset["icd10_proc"][index],"_id":len(df),"num_words":len(dataset["text"][index].split()),"num_targets":len(dataset["target"][index]),"icd10_long_title":dataset["long_title"][index]},ignore_index=True)
            
        split_df = split_df._append({"_id":len(df)-1,"split":"train"},ignore_index=True)
    for i in range(10):
        index = random.randint(0,len(dataset))
        df = df._append({"text":dataset["text"][index],"target":dataset["target"][index],"icd10_diag":dataset["icd10_diag"][index],"icd10_proc":dataset["icd10_proc"][index],"_id":len(df),"num_words":len(dataset["text"][index].split()),"num_targets":len(dataset["target"][index]),"icd10_long_title":dataset["long_title"][index]},ignore_index=True)
            
        split_df = split_df._append({"_id":len(df)-1,"split":"val"},ignore_index=True)

    # Save the generated dataset and split file in the output directory
    df.to_feather(os.path.join(generated_data_path, "generated_dataset.feather"))
    split_df.to_feather(os.path.join(generated_data_path, "generated_dataset_split.feather"))
    df.to_csv(os.path.join(generated_data_path, "generated_dataset.csv"), index=False)
    print(f"Generated dataset without noise saved to {os.path.join(generated_data_path, 'generated_dataset.feather')}")

else:
    #private ESA

    embedding_few_shot = few_shot_embeddings_path
    embedding_zero_shot = zero_shot_embeddings_path


    zero_shot_embedding  = pd.read_csv(embedding_zero_shot).values
    few_shot_embedding  = pd.read_csv(embedding_few_shot).values



    from prv_accountant.dpsgd import find_noise_multiplier
    size22 = 100
    size11 = 100
    num_step =100
    final = {}
    for eps in EPSILONS:
        sampling_probability = 100*NUM_SHOTS/DATASET_SIZE # 4 is 4-shot prediction, 14732 is the data size
        noise_multiplier = find_noise_multiplier(
                        sampling_probability=sampling_probability,
                        num_steps=num_step,
                        target_epsilon=eps,
                        target_delta=5e-5,
                        eps_error=0.01,
                        mu_max=100)
        print("noise", noise_multiplier)

        
        ensemble1 = NUM_C
        ensemble2 = args.ensemble2
        size1 = size11
        size2 = size22
        rouge1, rouge2, rougeL, rougeLsum = [],[],[],[]
        indexs = []
        for i in range(GENERATED_DATASET_SIZE):
            # get mean embedding of df2
            curr_emb = few_shot_embedding[(i*ensemble2):((i+1)*ensemble2+size1)]
            curr_emb_sum = np.sum(curr_emb, axis=0, keepdims=True)
            curr_emb_sum += np.random.normal(loc=0, scale=noise_multiplier, size=curr_emb_sum.shape)
            mean_emb = curr_emb_sum/ensemble1

            # Use consistent indexing based on NUM_C
            start_idx = i * NUM_C
            end_idx = (i + 1) * NUM_C
            dist = cosine_similarity(mean_emb, zero_shot_embedding[start_idx:end_idx])

            #dist = cosine_similarity(mean_emb, zero_shot_embedding[(i*ensemble2):(i*ensemble2+size2)])
            indexs.append(np.argmax(dist)+i*ensemble2)
        final_pred = []
        for index in indexs:
            final_pred.append(zero_pred[index])
        final[str(eps)] = final_pred

    #turn the three generated datasets into three feather files and generate a split for each which is entirely test set
    
    #import feather

    for epsilon in final.keys():
        df = pd.DataFrame(final[epsilon], columns=["text","target","icd10_diag","icd10_proc","icd10_long_title"])
        #add column named _id with random numbers
        df["_id"] = [i for i in range (len(df))]
        #add column named num_words with the number of words in the text
        df["num_words"] = [len(x.split()) for x in df["text"]]
        #add column named num_targets with the number of targets in the target
        df["num_targets"] = [len(x) for x in df["target"]]
        #generate a split file assignin all the samples to the test set
        split = ["test"]*len(df)
        ids = [df["_id"][i] for i in range(len(df))]

        split_df = pd.DataFrame({"_id":ids,"split":split})
        
        #add 100 random samples from dataset to the generated dataset for train split and 100 for val split (necessary workaround for evaluation through automatical-medical-coding evaluation)
        for i in range(10):
            index = random.randint(0,len(dataset))
            df = df._append({"text":dataset["text"][index],"target":dataset["target"][index],"icd10_diag":dataset["icd10_diag"][index],"icd10_proc":dataset["icd10_proc"][index],"_id":len(df),"num_words":len(dataset["text"][index].split()),"num_targets":len(dataset["target"][index]),"icd10_long_title":dataset["long_title"][index]},ignore_index=True)
            split_df = split_df._append({"_id":len(df)-1,"split":"train"},ignore_index=True)
        for i in range(10):
            index = random.randint(0,len(dataset))
            df = df._append({"text":dataset["text"][index],"target":dataset["target"][index],"icd10_diag":dataset["icd10_diag"][index],"icd10_proc":dataset["icd10_proc"][index],"_id":len(df),"num_words":len(dataset["text"][index].split()),"num_targets":len(dataset["target"][index]),"icd10_long_title":dataset["long_title"][index]},ignore_index=True)
            split_df = split_df._append({"_id":len(df)-1,"split":"val"},ignore_index=True)

        # Save the generated dataset and split file in the output directory with epsilon in the name
        df.to_feather(os.path.join(generated_data_path, f"generated_dataset_eps_{epsilon}.feather"))
        split_df.to_feather(os.path.join(generated_data_path, f"generated_dataset_split_eps_{epsilon}.feather"))
        df.to_csv(os.path.join(generated_data_path, f"generated_dataset_eps_{epsilon}.csv"), index=False)
        print(f"Generated dataset for epsilon {epsilon} saved to {os.path.join(generated_data_path, f'generated_dataset_eps_{epsilon}.feather')}")

#print the total runtime
print("Total runtime: ",(time.time()-start)/60,"minutes")
#print dataset size
print("Dataset size: ",len(dataset))

# Save a settings.json file with all parameters used
settings = vars(args)
settings["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
with open(os.path.join(generated_data_path, "settings.json"), "w") as f:
    json.dump(settings, f, indent=4)
