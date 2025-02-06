# DP-Clinical-ICL

This repository contains code for generating clinical discharge summaries using In-Context Learning (ICL) with differential privacy guarantees. The project uses the MIMIC-IV dataset and various language models through Ollama.


## Installation

1. Create and activate a new conda environment:
```bash
conda create -n dp-clinical python=3.8
conda activate dp-clinical
```

2. Clone the repository:
```bash
git clone https://github.com/yourusername/DP-Clinical-ICL.git
cd DP-Clinical-ICL
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## Dataset Setup

1. Request access to MIMIC-IV dataset at:
   - MIMIC-IV Clinical Database: https://physionet.org/content/mimiciv/2.2/
   - MIMIC-IV Notes: https://www.physionet.org/content/mimic-iv-note/2.2/

2. After getting access, download the necessary files:
   - From MIMIC-IV Clinical: `diagnoses_icd.csv`, `procedures_icd.csv`
   - From MIMIC-IV Notes: `discharge.csv`

3. Navigate to the data directory:
```bash
cd data
```

> **Note**: The download process requires approximately 10GB of disk space and may take a considerable amount of time depending on your internet connection. It's recommended to use `tmux` to prevent the download from being interrupted if your connection drops:
> ```bash
> # Install tmux if not already installed
> sudo apt-get install tmux
> 
> # Create a new tmux session
> tmux new -s mimic_download
> 
> # Now run the download commands inside tmux
> # To detach from the session: press Ctrl+B, then D
> # To reattach to the session later: tmux attach -t mimic_download
> ```

4. Run the commands to download the data:
```bash
wget -r -N -c -np --user [YOUR_USERNAME] --ask-password https://physionet.org/files/mimic-iv-note/2.2/
wget -r -N -c -np --user [YOUR_USERNAME] --ask-password https://physionet.org/files/mimiciv/2.2/
```

5. If everything went well, you should have the following structure:
```
data/
├── generated/
│   └── [Generated datasets will be saved here]
└── physionet.org/
    └── files/
        ├── mimic-iv-note/
        │   └── 2.2/
        │       └── note/
        │           └── discharge.csv.gz
        └── mimiciv/
            └── 2.2/
                └── hosp/
                    ├── diagnoses_icd.csv.gz
                    ├── procedures_icd.csv.gz
                    ├── d_icd_procedures.csv.gz
                    └── d_icd_diagnoses.csv.gz
```


## Data Extraction

Before running the generation script, you need to process the MIMIC-IV dataset to create the required format. The `extract_data_amc.py` script handles this by:
1. Loading and merging the necessary MIMIC-IV files
2. Formatting ICD codes correctly
3. Creating the required data structure with discharge summaries and their associated codes

### Running the Extraction

1. Make sure all MIMIC-IV files are in place as shown in the directory structure above
2. Navigate to the root directory again:
```bash
cd ..
```
3. Run the extraction script:
```bash
python extract_data_amc.py
```

This will create two files in your `data/` directory:
- `mimiciv_icd10.feather`: The main dataset file containing:
  - Discharge summaries (`text`)
  - ICD-10 codes (`target`, `icd10_diag`, `icd10_proc`)
  - Code descriptions (`long_title`)
- `mimiciv_icd10_split.feather`: Train/validation/test split information

### Verifying the Extraction

You can check if the extraction was successful by verifying that both files exist and contain data:
```bash
ls -l data/mimiciv_icd10*.feather
```

The extracted dataset should contain properly formatted records with:
- Diagnostic codes including periods (e.g., "I25.10")
- Procedure codes without periods (e.g., "02HN3DZ")
- Non-empty text fields
- At least one ICD-10 code per record

Only after successful data extraction should you proceed to running `DP_ICL_gen.py`.

## Using Custom Dataset

If you want to use your own dataset instead of MIMIC-IV, you'll need to format it according to the following specifications:

### Required File Format

Your dataset should be saved as a Feather file (`.feather`) with the following columns:

- `_id`: Unique identifier for each record
- `text`: The clinical discharge summary text
- `target`: List of ICD-10 codes associated with the text (in order: first every diagnostic code, then every procedure code)
- `icd10_diag`: List of ICD-10 diagnostic codes
- `icd10_proc`: List of ICD-10 procedure codes
- `long_title`: List of long descriptions for the ICD codes

### Data Requirements

1. The ICD-10 codes should be properly formatted:
   - Diagnostic codes should include a period after the first 3 characters (e.g., "A01.1")
   - Procedure codes should not include periods

2. Each record should have:
   - Non-empty text field
   - At least one ICD-10 code in either diagnostic or procedure codes
   - Corresponding long titles for each code

3. To use your custom dataset:
   - Use the `--custom_dataset_path` parameter to specify a different location:
     ```bash
     python DP_ICL_gen.py --custom_dataset_path /path/to/your/dataset.feather
     ```

### Example Data Format

```python
{
    '_id': 1234,
    'text': 'Patient admitted with chest pain...',
    'target': ['I25.10', 'Z95.5', '02HN3DZ'],
    'icd10_diag': ['I25.10', 'Z95.5'],
    'icd10_proc': ['02HN3DZ'],
    'long_title': ['Atherosclerotic heart disease', 'Presence of coronary stent', 'Insertion of stent into coronary artery']
}
```

This will create processed dataset files in the `data/` directory.

## Data Generation

The main script for generating clinical summaries is `DP_ICL_gen.py`. Here are some key parameters and how to use them:

### Basic Usage

```bash
python DP_ICL_gen.py --model_name llama3.2 --num_shots 5 --generated_dataset_size 100
```

### Important Parameters

- `--model_name`: Specify which Ollama model to use (e.g., `llama2`, `mistral`, `mixtral`)
- `--num_shots`: Number of few-shot examples (default: 5)
- `--generated_dataset_size`: Number of samples to generate (default: 100)
- `--prompt_index`: Index of the prompt template to use (0-2)
- `--temperature`: Temperature for generation (default: 0.7)

### Custom Prompt

To add your own prompt, you can modify the `prompts` list in `DP_ICL_gen.py`. The prompts are defined starting at line 58 of the file:

```python
prompts = [
    "Generate a clinical discharge summary...",  # Prompt 0
    """Please generate a realistic and concise clinical...""",  # Prompt 1
    """Please generate a realistic, concise, and professional...""",  # Prompt 2
    """[ADD HERE YOUR OWN PROMPT]
ICD10-CODES= """  # Prompt 3 (Custom)
]
```
NOTE:
- The prompt should end with `ICD10-CODES= ` so that the script can insert the ICD10 codes in the right place

To use your custom prompt:
1. Open `DP_ICL_gen.py`
2. Find the `prompts` list
3. Replace the text in index 3 at line 111 (`[ADD HERE YOUR OWN PROMPT]`) with your custom prompt
4. Run the script with `--prompt_index 3`

### Example Commands

1. Generate 100 samples using llama3.2 with custom prompt (inserted in the script):
```bash
python DP_ICL_gen.py --model_name llama3.2 --num_shots 5 --generated_dataset_size 100 --prompt_index 3
```

2. Generate samples with higher temperature for more diversity:
```bash
python DP_ICL_gen.py --model_name mistral --temperature 0.9 --num_shots 3 --generated_dataset_size 50
```

3. Non-private generation (without DP):
```bash
python DP_ICL_gen.py --model_name llama2 --nonprivate --num_shots 5 --generated_dataset_size 100
```

## Output

Generated datasets will be saved in the `data/generated/` directory in both Feather and CSV formats. The filename will include the parameters used for generation, making it easy to identify different runs.

## Available Models

You can check available models at https://ollama.com/search. Make sure to have the desired model pulled in Ollama before running the generation script.

To pull a model run:
```bash
ollama pull [MODEL_NAME]
```

## Notes

- The script uses the Sentence Transformers model 'all-MiniLM-L6-v2' for embedding calculations
- For private generation, different epsilon values can be specified using the `--epsilons` parameter
- The `--cardio` flag can be used to filter for cardiology-related codes only