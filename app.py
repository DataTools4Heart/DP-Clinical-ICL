import streamlit as st
import subprocess
import os
import pandas as pd
from pathlib import Path

st.set_page_config(page_title="DP-Clinical-ICL Generator", layout="wide")

def check_system_requirements():
    """Check if system meets the minimum requirements"""
    import psutil
    import torch
    
    requirements = {
        "RAM": {"required": 16, "actual": round(psutil.virtual_memory().total/1024**3)},
        "Disk": {"required": 10, "actual": round(psutil.disk_usage('/').free/1024**3)},
        "GPU": {"required": True, "actual": torch.cuda.is_available()},
        "GPU Memory": {"required": 14, "actual": torch.cuda.get_device_properties(0).total_memory/1024**3 if torch.cuda.is_available() else 0}
    }
    return requirements

def setup_environment():
    """Create conda environment and install requirements"""
    try:
        # First try to initialize conda if not already initialized
        try:
            subprocess.run(["conda", "init", "bash"], check=True)
            st.warning("Conda was not initialized. Please restart your terminal or run 'source ~/.bashrc' before proceeding.")
            return False
        except subprocess.CalledProcessError:
            pass  # Conda is already initialized
        
        # Create and activate environment
        subprocess.run(["conda", "create", "-n", "dp-clinical", "python=3.9", "-y"], check=True)
        subprocess.run(["conda", "activate", "dp-clinical"], check=True)
        subprocess.run(["pip", "install", "-r", "requirements.txt"], check=True)
        return True
    except subprocess.CalledProcessError as e:
        st.error(f"Error: {str(e)}")
        return False

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
        # Create a placeholder for progress messages
        progress_placeholder = st.empty()
        
        # Use Popen to capture output in real-time
        process = subprocess.Popen(
            ["python", "extract_data_amc.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            bufsize=1
        )
        
        # Create a progress container
        with st.expander("Extraction Progress", expanded=True):
            # Show output in real-time
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    st.write(output.strip())
                    # Update the main progress message
                    if "Loading" in output:
                        progress_placeholder.info("📂 Loading data files...")
                    elif "Merging" in output:
                        progress_placeholder.info("🔄 Merging datasets...")
                    elif "Formatting" in output:
                        progress_placeholder.info("✏️ Formatting ICD codes...")
                    elif "Filtering" in output:
                        progress_placeholder.info("🔍 Filtering records...")
                    elif "Saving" in output:
                        progress_placeholder.info("💾 Saving processed data...")
        
        # Get the return code
        return_code = process.poll()
        
        if return_code == 0:
            progress_placeholder.success("✅ Data extraction completed successfully!")
            return True
        else:
            error = process.stderr.read()
            progress_placeholder.error(f"❌ Extraction failed: {error}")
            return False
            
    except Exception as e:
        st.error(f"❌ Error during extraction: {str(e)}")
        return False

def generate_data(model_name, num_shots, dataset_size, temperature, prompt_index, custom_dataset=None, prompt=None, nonprivate=False, epsilons=None):
    """Run data generation script"""
    try:
        cmd = [
            "python", "DP_ICL_gen.py",
            "--model_name", model_name,
            "--num_shots", str(num_shots),
            "--generated_dataset_size", str(dataset_size),
            "--temperature", str(temperature),
            "--prompt_index", str(prompt_index)
        ]
        
        if custom_dataset:
            cmd.extend(["--custom_dataset_path", custom_dataset])
            
        if prompt:
            cmd.extend(["--prompt", custom_prompt])
            
        if nonprivate:
            cmd.append("--nonprivate")
            
        if epsilons:
            cmd.extend(["--epsilons"] + [str(eps) for eps in epsilons])
        
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError:
        return False

# Streamlit UI
st.title("DP-Clinical-ICL Generator")

# Sidebar for navigation
page = st.sidebar.selectbox(
    "Step",
    ["System Check", "Environment Setup", "Dataset Download", "Data Extraction", "Data Generation"]
)

if page == "System Check":
    st.header("System Requirements Check")
    
    requirements = check_system_requirements()
    
    for resource, details in requirements.items():
        if details["actual"] >= details["required"]:
            st.success(f"{resource}: {details['actual']} (Required: {details['required']})")
        else:
            st.error(f"{resource}: {details['actual']} (Required: {details['required']})")

elif page == "Environment Setup":
    st.header("Environment Setup")
    
    if st.button("Setup Environment"):
        with st.spinner("Setting up environment..."):
            if setup_environment():
                st.success("Environment setup completed!")
            else:
                st.error("Environment setup failed. Please check the logs.")

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
    
    if st.button("Extract Data"):
        with st.spinner("Extracting data... This may take a few minutes."):
            if extract_data():
                st.success("Data extraction completed!")
            else:
                st.error("Data extraction failed. Please check the logs.")

elif page == "Data Generation":
    st.header("Data Generation")
    
    # Initialize session state for generated files if not exists
    if 'generated_files' not in st.session_state:
        st.session_state.generated_files = []
    
    col1, col2 = st.columns(2)
    
    with col1:
        model_name = st.text_input("Model Name", value="llama3.2", help="Enter the name of any model available in Ollama (e.g., llama2, mistral, mixtral, etc.)")
        num_shots = st.number_input("Number of Shots", min_value=1, value=5)
        dataset_size = st.number_input("Dataset Size", min_value=1, value=100)
    
    with col2:
        temperature = st.slider("Temperature", min_value=0.1, max_value=1.0, value=0.7)
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
            value="Generate a clinical discharge summary...\nICD10-CODES= ",
            height=200,
            help="Enter your custom prompt. Make sure it ends with 'ICD10-CODES= ' for proper code insertion."
        )
        prompt_index = None  # Not used when custom prompt is provided
    else:
        prompt_index = st.number_input("Prompt Index", min_value=0, max_value=3, value=0)
        custom_prompt = None
    
    # Add a note about available models
    st.info("You can check available models at https://ollama.com/search. Make sure to pull your desired model first using 'ollama pull MODEL_NAME'")
    
    if st.button("Generate Data"):
        with st.spinner("Generating data..."):
            if generate_data(
                model_name, num_shots, dataset_size, temperature, 
                prompt_index, custom_dataset, custom_prompt, 
                nonprivate=(privacy_option == "Non-private"),
                epsilons=epsilons
            ):
                st.success("Data generation completed!")
                
                # Update the list of generated files
                generated_files = [
                    f for f in Path("data/generated").glob("*.csv")
                    if not any(x in f.name for x in ['embeddings', 'samples'])
                ]
                st.session_state.generated_files = generated_files
            else:
                st.error("Data generation failed. Please check the logs.")
    
    # Display generated files section (always show if files exist)
    if st.session_state.generated_files:
        st.subheader("Generated Files")
        for file in st.session_state.generated_files:
            if file.exists():  # Check if file still exists
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
                # Remove file from session state if it no longer exists
                st.session_state.generated_files.remove(file)
        
        # Add a clear button
        if st.button("Clear Generated Files List"):
            st.session_state.generated_files = []
            st.experimental_rerun()
    elif page == "Data Generation" and not st.button("Generate Data"):  # Only show warning if no generation is in progress
        st.warning("No generated dataset files found. Click 'Generate Data' to create new files.") 