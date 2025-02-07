import streamlit as st
import subprocess
import os
import pandas as pd
from pathlib import Path
import psutil
import time

st.set_page_config(page_title="DP-Clinical-ICL Generator", layout="wide")

def check_system_requirements():
    """Check if system meets the minimum requirements"""
    requirements = {
        "RAM": {"required": 16, "actual": round(psutil.virtual_memory().total/1024**3)},
        "Disk": {"required": 10, "actual": round(psutil.disk_usage('/').free/1024**3)}
    }
    
    # Check GPU using nvidia-smi
    try:
        gpu_info = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.total', '--format=csv,noheader,nounits'], 
                                         universal_newlines=True)
        # Split the output into lines and get the maximum GPU memory
        gpu_memories = [int(x.strip()) for x in gpu_info.strip().split('\n')]
        total_gpu_memory = sum(gpu_memories) / 1024  # Convert MB to GB
        num_gpus = len(gpu_memories)
        
        requirements["GPU"] = {"required": True, "actual": True}
        requirements["GPU Memory"] = {"required": 14, "actual": round(total_gpu_memory, 1)}
        requirements["Number of GPUs"] = {"required": 1, "actual": num_gpus}
    except (subprocess.CalledProcessError, FileNotFoundError):
        requirements["GPU"] = {"required": True, "actual": False}
        requirements["GPU Memory"] = {"required": 14, "actual": 0}
        requirements["Number of GPUs"] = {"required": 1, "actual": 0}
    
    return requirements

def download_mimic_data(username, password):
    """Download MIMIC-IV dataset files"""
    try:
        os.makedirs("data", exist_ok=True)
        os.chdir("data")
        
        # Download commands with credentials
        cmd1 = f"wget -r -N -c -np --user {username} --password {password} https://physionet.org/files/mimic-iv-note/2.2/"
        cmd2 = f"wget -r -N -c -np --user {username} --password {password} https://physionet.org/files/mimiciv/2.2/"
        
        subprocess.run(cmd1.split(), check=True)
        subprocess.run(cmd2.split(), check=True)
        
        os.chdir("..")
        return True
    except subprocess.CalledProcessError:
        return False

def extract_data():
    """Run data extraction script with progress information"""
    try:
        # Create containers for progress
        status_container = st.empty()
        progress_container = st.container()
        
        status_container.info("Starting data extraction process...")
        
        # Check if the script exists
        if not os.path.exists("extract_data_amc.py"):
            status_container.error("❌ extract_data_amc.py not found in the current directory!")
            return False
            
        # Check if input files exist
        required_files = [
            "data/physionet.org/files/mimic-iv-note/2.2/note/discharge.csv.gz",
            "data/physionet.org/files/mimiciv/2.2/hosp/procedures_icd.csv.gz",
            "data/physionet.org/files/mimiciv/2.2/hosp/diagnoses_icd.csv.gz",
            "data/physionet.org/files/mimiciv/2.2/hosp/d_icd_procedures.csv.gz",
            "data/physionet.org/files/mimiciv/2.2/hosp/d_icd_diagnoses.csv.gz"
        ]
        
        for file in required_files:
            if not os.path.exists(file):
                status_container.error(f"❌ Required file not found: {file}")
                return False
        
        status_container.info("📂 All required files found. Starting extraction...")
        
        # Use Popen to capture output in real-time
        process = subprocess.Popen(
            ["python", "-u", "extract_data_amc.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            env={**os.environ, 'PYTHONUNBUFFERED': '1', 'PYTHONWARNINGS': 'ignore'}
        )
        
        # Show output in real-time
        while True:
            output = process.stdout.readline()
            error = process.stderr.readline()
            
            if output == '' and error == '' and process.poll() is not None:
                break
                
            if output:
                with progress_container:
                    if "warning" not in output.lower():  # Skip warning messages
                        st.info(output.strip())
                # Update the status message
                if "Loading" in output:
                    status_container.info("📂 Loading data files...")
                elif "Merging" in output:
                    status_container.info("🔄 Merging datasets...")
                elif "Formatting" in output:
                    status_container.info("✏️ Formatting ICD codes...")
                elif "Filtering" in output:
                    status_container.info("🔍 Filtering records...")
                elif "Saving" in output:
                    status_container.info("💾 Saving processed data...")
                
            if error and "warning" not in error.lower():  # Only show non-warning errors
                with progress_container:
                    st.error(error.strip())
        
        # Get the return code and any error output
        return_code = process.poll()
        error_output = process.stderr.read()
        
        if return_code == 0:
            if os.path.exists("data/mimiciv_icd10.feather"):
                status_container.success("✅ Data extraction completed successfully!")
                return True
            else:
                status_container.error("❌ Extraction process completed but output file not found!")
                return False
        else:
            if "warning" not in error_output.lower():  # Only show non-warning errors
                status_container.error(f"❌ Extraction failed with error code {return_code}: {error_output}")
            return False
            
    except Exception as e:
        st.error(f"❌ Error during extraction: {str(e)}")
        return False

def generate_data(model_name, num_shots, dataset_size, temperature, prompt_index, custom_dataset=None, prompt=None, nonprivate=False, epsilons=None):
    """Run data generation script"""
    try:
        # Create data and generated directories if they don't exist
        os.makedirs("data/generated", exist_ok=True)
        
        cmd = [
            "python", "DP_ICL_gen.py",
            "--model_name", model_name,
            "--num_shots", str(num_shots),
            "--generated_dataset_size", str(dataset_size),
            "--temperature", str(temperature),
        ]
        
        # Handle custom prompt vs prompt index
        if prompt:
            cmd.extend(["--prompt", prompt])
            cmd.extend(["--prompt_index", "0"])  # Use 0 as default when custom prompt is provided
        else:
            cmd.extend(["--prompt_index", str(prompt_index)])
        
        if custom_dataset:
            cmd.extend(["--custom_dataset_path", custom_dataset])
            
        if nonprivate:
            cmd.append("--nonprivate")
            
        if epsilons:
            cmd.extend(["--epsilons"] + [str(eps) for eps in epsilons])
        
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        st.error(f"Generation failed with error: {str(e)}")
        return False

# Streamlit UI
st.title("DP-Clinical-ICL Generator")

# Sidebar for navigation
page = st.sidebar.selectbox(
    "Step",
    ["System Check", "Dataset Download", "Data Extraction", "Data Generation"]
)

if page == "System Check":
    st.header("System Requirements Check")
    
    requirements = check_system_requirements()
    
    for resource, details in requirements.items():
        if details["actual"] >= details["required"]:
            st.success(f"{resource}: {details['actual']} (Required: {details['required']})")
        else:
            st.error(f"{resource}: {details['actual']} (Required: {details['required']})")

elif page == "Dataset Download":
    st.header("MIMIC-IV Dataset Download")
    
    username = st.text_input("PhysioNet Username")
    password = st.text_input("PhysioNet Password", type="password")
    
    if st.button("Download Dataset"):
        if username and password:
            with st.spinner("Downloading dataset..."):
                if download_mimic_data(username, password):
                    st.success("Dataset downloaded successfully!")
                else:
                    st.error("Dataset download failed. Please check your credentials.")
        else:
            st.warning("Please enter your PhysioNet credentials.")

elif page == "Data Extraction":
    st.header("Data Extraction")
    
    st.warning("Note: Data extraction may take a few minutes to start and several more minutes to complete. This is normal as it needs to process large files.")
    
    if st.button("Extract Data"):
        with st.spinner("Extracting data... This may take a few minutes."):
            if extract_data():
                st.success("Data extraction completed!")
            else:
                st.error("Data extraction failed. Please check the logs.")

elif page == "Data Generation":
    st.header("Data Generation")
    
    # Initialize session state for generated files and timestamp if not exists
    if 'generated_files' not in st.session_state:
        st.session_state.generated_files = []
    if 'last_generation_time' not in st.session_state:
        st.session_state.last_generation_time = None
    
    col1, col2 = st.columns(2)
    
    with col1:
        model_name = st.text_input("Model Name", value="llama3.2", help="Enter the name of any model available in Ollama (e.g., llama2, mistral, mixtral, etc.)")
        num_shots = st.number_input("Number of Shots", min_value=1, value=5)
        dataset_size = st.number_input("Dataset Size", min_value=5, value=100)
    
    with col2:
        temperature = st.slider(
            "Temperature",
            min_value=0.1,
            max_value=1.0,
            value=0.7,
            help="Controls the randomness in the model's output. Lower values (0.1-0.3) make the text more focused and deterministic, while higher values (0.7-1.0) make it more creative and diverse. For medical text generation, values between 0.5-0.7 often provide a good balance between accuracy and variation."
        )
        use_custom_prompt = st.checkbox("Use Custom Prompt", value=False)
        custom_dataset = st.file_uploader("Custom Dataset (optional)", type=["feather"])
    
    # Privacy settings
    privacy_option = st.radio(
        "Privacy Setting",
        ["Non-private", "Default epsilons [1,3,8]", "Custom epsilons"],
        help="Choose privacy level for generation"
    )
    
    if privacy_option == "Custom epsilons":
        custom_epsilons = st.text_input(
            "Custom Epsilon Values",
            value="1,3,8",
            help="Enter comma-separated epsilon values (e.g., 1,3,8)"
        )
        epsilons = [float(eps.strip()) for eps in custom_epsilons.split(",") if eps.strip()]
    elif privacy_option == "Default epsilons [1,3,8]":
        epsilons = [1, 3, 8]
    else:  # Non-private
        epsilons = None
    
    # Custom prompt or prompt index based on checkbox
    if use_custom_prompt:
        custom_prompt = st.text_area(
            "Custom Prompt",
            value="""[EXAMPLE PROMPT]
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
            height=600,
            help="Enter your custom prompt. Make sure it ends with 'ICD10-CODES= ' for proper code insertion."
        )
        prompt_index = None  # Not used when custom prompt is provided
    else:
        st.subheader("Choose a Prompt Template")
        
        prompts = {
            0: "Basic prompt - Generates a simple, straightforward discharge summary:\n\n" + 
               "Generate a clinical discharge summary of a patient who had the conditions and procedures described by the following codes ICD10-CODES= ",
            
            1: "Detailed prompt - Focuses on clinical accuracy and standard formatting:\n\n" +
               """Please generate a realistic and concise clinical discharge summary for a patient based on the following ICD-10 codes. Do not include the ICD-10 codes themselves in the report; instead, refer to the medical conditions they represent. Before writing the summary, internally create a logical and medically accurate timeline of the patient's diagnosis, treatment, and progress. Use standard medical abbreviations where appropriate to mirror real clinical documentation. Focus on essential clinical information, avoiding unnecessary explanations or verbosity. Ensure that the report accurately reflects the management of both common and rare diseases as applicable.

Requirements:
- Use standard medical abbreviations (e.g., BP for blood pressure, HR for heart rate)
- Keep the summary concise and focused on relevant clinical details
- Ensure the sequence of events and timing make medical sense
- Cover a wide range of use cases, including rare diseases when specified
- Do not include any ICD-10 codes in the text of the report

Format:
Admission Date: [Date]
Discharge Date: [Date]
Discharge Summary:
Reason for Admission: [Brief]
History of Present Illness: [Concise]
Hospital Course: [Key events]
Discharge Plan: [Instructions]
ICD10-CODES= """,
            
            2: "Professional prompt - Emphasizes medical documentation standards:\n\n" +
               """Please generate a realistic, concise, and professional clinical discharge summary for a patient based on the following ICD-10 codes. Do not include the ICD-10 codes themselves in the report; instead, reference the medical conditions they represent. Before composing the summary, internally develop a logical and medically accurate patient case, including the timeline of symptom onset, diagnosis, interventions, and outcomes.

The discharge summary should:
- Use clinical language with standard medical abbreviations
- Be succinct and focused on essential information
- Reflect a coherent sequence of events
- Mimic actual clinical documentation style

Format:
Patient Identification: [Initials only]
Age/Gender: [e.g., 45M]
Dates: [Admission/Discharge]
Diagnoses: [Primary/Secondary]
Hospital Course: [Key events]
Discharge Plan: [Complete instructions]
ICD10-CODES= """,
            
            3: "Custom prompt template - Use this as a starting point for your own prompt:\n\n" +
               """[ADD YOUR CUSTOM PROMPT HERE]
Remember to:
- Include clear formatting instructions
- Specify medical terminology preferences
- Define documentation standards
- End with ICD10-CODES= """
        }
        
        prompt_index = st.selectbox(
            "Prompt Index",
            options=list(prompts.keys()),
            format_func=lambda x: f"Prompt {x}",
            help="Choose a predefined prompt template"
        )
        
        # Show the selected prompt
        st.text_area(
            "Selected Prompt Preview",
            value=prompts[prompt_index],
            height=400,
            disabled=True
        )
        
        custom_prompt = None
    
    # Add a note about available models
    st.info("You can check available models at https://ollama.com/search. Make sure to pull your desired model first using 'ollama pull MODEL_NAME'")
    
    if st.button("Generate Data"):
        with st.spinner("Generating data..."):
            # Record the start time of generation
            generation_start_time = time.time()
            
            if generate_data(
                model_name, num_shots, dataset_size, temperature, 
                prompt_index, custom_dataset, custom_prompt, 
                nonprivate=(privacy_option == "Non-private"),
                epsilons=epsilons
            ):
                st.success("Data generation completed!")
                
                # Wait a brief moment to ensure files are written
                time.sleep(1)
                
                # Update the list of generated files - only get files from this generation
                st.session_state.last_generation_time = generation_start_time
                current_files = [
                    f for f in Path("data/generated").glob("*.csv")
                    if not any(x in f.name for x in ['embeddings', 'samples'])
                    and os.path.getctime(f) >= generation_start_time
                ]
                st.session_state.generated_files = current_files
            else:
                st.error("Data generation failed. Please check the logs.")
    
    # Display generated files section (always show if files exist)
    if st.session_state.generated_files and st.session_state.last_generation_time:
        st.subheader("Generated Files")
        for file in st.session_state.generated_files:
            if file.exists() and os.path.getctime(file) >= st.session_state.last_generation_time:  # Only show files from last generation
                df = pd.read_csv(file)
                st.write(f"File: {file.name}")
                st.dataframe(df.head())
                
                # Download button for each file
                with open(file, "rb") as f:
                    st.download_button(
                        label=f"Download {file.name}",
                        data=f,
                        file_name=file.name,
                        mime="text/csv"
                    )
    else:
        # Show warning if no files exist
        st.warning("No generated dataset files found. Use the form above to generate new files.") 