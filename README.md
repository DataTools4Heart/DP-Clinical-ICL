# DP-Clinical-ICL

A user-friendly web application for generating clinical discharge summaries. This tool helps medical researchers create realistic patient discharge summaries while maintaining privacy. It uses artificial intelligence (specifically, In-Context Learning) and can apply privacy protection to the generated data.

> **Note for Technical Users**: This README provides user-friendly instructions focused on using the Streamlit web interface. If you want to understand the technical details, command-line usage, custom dataset format, or implementation details, please check [README_OLD.md](README_OLD.md).

## What You Need Before Starting

1. **Computer Requirements**:
   - At least 16GB of RAM (memory)
   - At least 10GB of free disk space
   - An NVIDIA graphics card (GPU) is recommended but not required
   - Any operating system (Windows, Mac, or Linux)

2. **Software Requirements**:
   - Python 3.9 or newer (if you don't have it, visit [Python's download page](https://www.python.org/downloads/))
   - Conda (download from [Anaconda's website](https://www.anaconda.com/download))
   - Ollama (will be installed automatically by our script)

3. **Access Requirements**:
   - MIMIC-IV dataset access credentials
     - Visit [PhysioNet](https://physionet.org/content/mimiciv/2.2/)
     - Create an account and complete the required training
     - Remember your username and password; you'll need them later

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
   git clone https://github.com/yourusername/DP-Clinical-ICL.git
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

### Step 1: System Check

This first page checks if your computer meets all requirements:
- Shows how much memory (RAM) you have
- Checks your available disk space
- Looks for a compatible graphics card
- Tells you if your system is ready to proceed

If any requirements aren't met, you'll see clear error messages explaining what's missing.

### Step 2: Dataset Download

Here you'll download the medical records database:
1. Enter your PhysioNet username and password
2. Click the "Download Dataset" button
3. Wait for the download to complete (about 8GB of data)
   - This might take 30-60 minutes depending on your internet speed
   - The app will show download progress
   - It's safe to leave this running in the background

### Step 3: Data Extraction

This step prepares the downloaded data:
1. Click the "Extract Data" button
2. Wait while the app processes the files
   - This typically takes 15-30 minutes
   - You'll see progress messages as it works
   - It's normal if it seems slow at first
   - Don't close the browser window during this step

### Step 4: Data Generation

This is where you create new discharge summaries. You have several options to control how they're generated:

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
4. [Create an issue](https://github.com/yourusername/DP-Clinical-ICL/issues) on our GitHub page

## Citation

If you use this tool in your research, please cite:
[Add citation information] 