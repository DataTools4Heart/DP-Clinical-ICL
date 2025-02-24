# DP-Clinical-ICL

A user-friendly web application for generating clinical discharge summaries. This tool helps medical researchers create realistic patient discharge summaries while maintaining privacy. It uses artificial intelligence (specifically, In-Context Learning) and can apply privacy protection to the generated data.

> **Important**: This application is designed to work with MIMIC-IV version 2.2. Other versions may not be compatible.

> **Note for Technical Users**: This README provides user-friendly instructions focused on using the Streamlit web interface. If you want to understand the technical details, command-line usage, custom dataset format, or implementation details, please check [README_OLD.md](README_OLD.md).

## What You Need Before Starting

1. **Computer Requirements**:
   - **Graphics Card (GPU)**:
     - An NVIDIA GPU with at least 14GB of VRAM is practically required
     - While the application can run on CPU, generation would take many hours or even days
   - At least 16GB of RAM (memory)
   - At least 10GB of free disk space
   - Any operating system (Windows, Mac, or Linux)

2. **Software Requirements**:
   - Python 3.9 or newer (if you don't have it, visit [Python's download page](https://www.python.org/downloads/))
   - Conda (download from [Anaconda's website](https://www.anaconda.com/download))
   - Ollama (will be installed automatically by our script)

3. **Access Requirements**:
   - MIMIC-IV version 2.2 dataset access credentials
     - Visit [PhysioNet MIMIC-IV 2.2](https://physionet.org/content/mimiciv/2.2/)
     - Create an account and complete the required training (see more in the [Step 2: Acquire MIMIC access](#step-2-acquire-mimic-access) section)

## Installation Guide

### Step 1: Opening a Terminal

#### On Windows:
1. Press the Windows key + R
2. Type "cmd" and press Enter
   - Or search for "Command Prompt" in the Start menu

#### On Mac:
1. Press Command + Space
2. Type "Terminal" and press Enter
   - Or find Terminal in Applications > Utilities

#### On Linux:
1. Press Ctrl + Alt + T
   - Or search for "Terminal" in your applications menu

### Step 2: Installing the Application

1. First, download the code:
   ```bash
   git clone https://github.com/DataTools4Heart/DP-Clinical-ICL.git
   ```
   - If this doesn't work, you may need to [install Git](https://git-scm.com/downloads) first

2. Move into the downloaded folder:
   ```bash
   cd DP-Clinical-ICL
   ```

3. Make the setup script executable and run it:
   ```bash
   chmod +x run_app.sh
   ./run_app.sh
   ```

The setup script will automatically:
- Set up a new environment for the application
- Install all required software
- Install Ollama (the AI model manager)
- Download the default AI model (llama3.2)
- Start the application in your web browser

If everything works correctly, your default web browser should open automatically with the application running.

## Using the Application

The application works like a step-by-step wizard, with four main steps shown in the left sidebar:

> **Note About Step Progression**: 
> - You can skip any step that was already completed in a previous session
> - The app will detect existing files and configurations
> - For example:
>   - If you've already downloaded the dataset, you can skip directly to extraction
>   - If you've already extracted the data, you can go straight to generation
>   - System checks can be skipped if you've verified your setup before
> - Just click the desired step in the left sidebar to navigate

### Step 1: System Check

This first page checks if your computer meets all requirements:
- Shows how much memory (RAM) you have
- Checks your available disk space
- Verifies GPU compatibility and memory
  - This is crucial as generation speed depends heavily on GPU availability
  - A compatible GPU reduces generation time from hours to minutes
  - The app will warn you if no GPU is found or if GPU memory is insufficient
  - While you can proceed without a GPU, it's not recommended for practical use

If any requirements aren't met, you'll see clear error messages explaining what's missing. Pay special attention to GPU-related messages, as they will significantly impact the usability of the application.

> You can skip this step in future sessions if your system configuration hasn't changed.

### Step 2: Acquire MIMIC access

Mimic is a credentialed dataset. You need to acquire access to it before you can download the dataset. For those who already have access, you can skip this step.
The training is a speciment research training, it is nothing technical or specific to the application and chances are trained clinicians are already able to complete the test.

1. Go to [PhysioNet](https://physionet.org/)
2. Create an account or login to your existing account
3. Go to [PhysioNet MIMIC-IV 2.2](https://physionet.org/content/mimiciv/2.2/)
4. Scroll down to the bottom of the page, here you will see this:

![MIMIC-IV 2.2](https://github.com/DataTools4Heart/DP-Clinical-ICL/blob/main/Images/Screenshot%202025-02-24%20alle%2016.16.36.png)

5. Click on "CITI Data or Specimens Only Research"
6. Complete the training following the instructions
7. Once the training is completed, you will be able to download the dataset
### Step 3: Dataset Download

Here you'll download the medical records database:
1. Enter your PhysioNet username and password
2. Click the "Download Dataset" button
3. Wait for the download to complete (about 8GB of data)
   - This might take from some minutes to some hours depending on your internet speed
   - The app will show download progress
   - It's safe to leave this running in the background

> If you've already downloaded the dataset in a previous session, you can skip this step.

### Step 4: Data Extraction

This step prepares the downloaded data:
1. Click the "Extract Data" button
2. Wait while the app processes the files
   - This typically takes 15-30 minutes
   - You'll see progress messages as it works
   - It's normal if it seems slow at first
   - Don't close the browser window during this step

> If you've already extracted the data and the files exist, you can skip directly to Data Generation.

### Step 5: Data Generation

This is where you create new discharge summaries. You have several options to control how they're generated:

> **Generation Time Estimates**: 
> The application will show you an estimated completion time based on your settings. For example:
> - With 100 samples and 5-shot setting:
>   - Expect ~5 hours on an NVIDIA RTX 3090
>   - Each sample takes about 1-2 minutes to generate
>   - The number of shots affects generation time
>   - More samples = longer total time
> - These estimates assume GPU availability
> - Times will be longer on less powerful GPUs
> - CPU-only generation is not recommended (could take days)

> **Important Note About Generation Times**: 
> - With a compatible GPU (14GB+ VRAM): Expect about 1-2 minutes per summary
> - Without a GPU: Generation could take 30+ minutes per summary
> - For bulk generation (e.g., 100 summaries), the difference is hours vs days
> - We strongly recommend using a computer with a compatible GPU

> This is typically the only step you'll need to repeat in subsequent sessions, as it creates new summaries each time.

#### Basic Options:

1. **AI Model** (default is "llama3.2"):
   - Think of this like choosing which expert writes your summaries
   - Stick with the default unless you have a specific reason to change it
   - Other options include "llama2", "mistral", or "mixtral"

2. **Number of Examples** (default is 5):
   - How many real examples the AI should look at
   - More examples = better quality but slower generation
   - Recommended: start with 5 and adjust if needed

3. **Number of Summaries** (default is 100):
   - How many new summaries you want to create
   - Start small (like 10) for testing
   - Larger numbers take longer to generate

#### Advanced Options:

1. **Temperature** (default is 0.7):
   - Controls how creative the AI can be
   - Lower (0.1-0.3): Very consistent, repetitive outputs
   - Medium (0.5-0.7): Good balance for medical text
   - Higher (0.7-1.0): More varied but potentially less accurate

2. **Privacy Protection**:
   - "Non-private": No privacy protection
   - "Default epsilons": Standard privacy protection
   - "Custom epsilons": Advanced privacy settings (consult with privacy experts)

3. **Custom Instructions**:
   - You can write your own instructions for the AI
   - Use the large text box to enter specific requirements
   - Must end with "ICD10-CODES= "
   - The default instructions work well for most cases

## Working with Generated Files

- All generated files are saved in a folder called "data/generated"
- Each file name includes information about how it was generated
- You can download files directly from the web interface
- Files remain available until you clear them using the "Clear Generated Files List" button

## Using Your Own Dataset

Instead of using MIMIC-IV, you can use your own clinical dataset. Here's what you need to know:

### Dataset Requirements

Your dataset must be in a specific format (Feather file, `.feather`) with these columns:
1. `_id`: A unique number for each record
2. `text`: The actual discharge summary
3. `target`: All the ICD-10 codes for this record
4. `icd10_diag`: Just the diagnostic codes
5. `icd10_proc`: Just the procedure codes
6. `long_title`: Descriptions of what each code means

### Important Format Rules

1. The ICD-10 codes must be written correctly:
   - Diagnostic codes need a period after the first 3 characters (example: "A01.1")
   - Procedure codes should not have periods (example: "02HN3DZ")

2. Each record in your dataset must have:
   - Text content (not empty)
   - At least one ICD-10 code
   - Descriptions for all codes

### Example Record

Here's what a single record in your dataset should look like:
```
Record ID: 1234
Text: "Patient admitted with chest pain..."
Diagnostic Codes: ["I25.10", "Z95.5"]
Procedure Codes: ["02HN3DZ"]
Code Descriptions: ["Atherosclerotic heart disease", "Presence of coronary stent", "Insertion of stent into coronary artery"]
```

## Common Problems and Solutions

1. **"The application seems frozen"**:
   - This is normal during data extraction and generation
   - Look for progress messages at the bottom of the page
   - Don't close the browser window

2. **"Out of memory" errors**:
   - Try generating fewer summaries at once
   - Close other large applications
   - Use a smaller AI model

3. **"Files not found" errors**:
   - Make sure the dataset download completed successfully
   - Try the download step again

4. **"Model not found" errors**:
   - Wait a few minutes after starting the application
   - The model might still be downloading

## Getting Help

If you encounter problems:
1. Check the error messages in the application
2. Look through the Troubleshooting section above
3. Contact your institution's IT support
4. Contact the maintainer directly:
   - Michele Miranda
   - Email: michele.miranda@translated.net or miranda@di.uniroma1.it
5. [Create an issue](https://github.com/yourusername/DP-Clinical-ICL/issues) on our GitHub page

## Citation

If you use this tool in your research, please cite:
[Add citation information] 