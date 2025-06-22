@echo off
setlocal enabledelayedexpansion

REM Fixed setup script for direnv environments in LangChain notebooks repository (Windows version)
REM This script sets up all virtual environments and installs requirements on Windows

echo Setting up direnv environments for LangChain notebooks repository (Windows)
echo ========================================================================

REM Check if direnv is installed
where direnv >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] direnv is not installed. Please install direnv first
    echo    - Option 1: Install with Scoop (recommended)
    echo      scoop install direnv
    echo    - Option 2: Manual installation
    echo      1. Download from: https://github.com/direnv/direnv/releases
    echo      2. Extract direnv.exe to a folder in your PATH
    echo    - Option 3: Use Windows Package Manager (winget)
    echo      winget install direnv.direnv
    echo    - After installation, add direnv hook to your shell
    echo      PowerShell: Add-Content $profile 'Invoke-Expression "$$(direnv hook pwsh)"'
    echo      Command Prompt: Use PowerShell or Git Bash for better direnv support
    pause
    exit /b 1
)

echo [OK] direnv is installed

REM Check if Python is installed
where python >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in PATH
    echo    - Install Python from: https://python.org
    echo    - Make sure to add Python to PATH during installation
    pause
    exit /b 1
)

echo [OK] Python is installed

REM Function to setup environment (using subroutine)
goto :main

:setup_env
set env_name=%~1
set env_path=%~2
set requirements_file=%~3

echo.
echo [SETUP] Setting up %env_name% environment
echo    Path: %env_path%
echo    Requirements: %requirements_file%

REM Change to the environment directory
pushd "%env_path%"

REM Allow direnv to create virtual environment
echo    Allowing direnv and creating virtual environment
direnv allow

REM Give direnv time to create the virtual environment
timeout /t 5 /nobreak >nul

REM Check if virtual environment was created
set venv_dir=
if exist "venv_root" set venv_dir=venv_root
if exist "venv_langchain" set venv_dir=venv_langchain
if exist "venv_deepeval" set venv_dir=venv_deepeval
if exist "venv_cotton" set venv_dir=venv_cotton
if exist "venv_ai_papers" set venv_dir=venv_ai_papers
if exist "venv_shared" set venv_dir=venv_shared

if defined venv_dir (
    if exist "%venv_dir%" (
        echo    [OK] Virtual environment created: %venv_dir%
        
        REM Activate the virtual environment manually
        call "%venv_dir%\Scripts\activate.bat"
        
        REM Install requirements if file exists
        if exist "%requirements_file%" (
            echo    Installing requirements
            pip install -r "%requirements_file%"
        ) else (
            echo    [WARNING] Requirements file not found: %requirements_file%
        )
    )
) else (
    echo    [WARNING] Virtual environment not found, skipping package installation
)

echo    [OK] %env_name% environment setup complete
popd
goto :eof

:main
REM Setup root environment
call :setup_env "Root" "." "requirements.txt"

REM Setup LangChain environment
call :setup_env "LangChain" "langchain" "requirements.txt"

REM Setup DeepEval environment
call :setup_env "DeepEval" "deepeval\deepeval_claude_created" "requirements.txt"

REM Setup COTTON environment
call :setup_env "COTTON" "Paper\1.3.3" "requirements.txt"

REM Setup AI-Papers environment
call :setup_env "AI-Papers" "AI-Papers" "requirements.txt"

REM Setup ETL environment
echo.
echo [SETUP] Setting up ETL environment
pushd ETL
direnv allow
timeout /t 5 /nobreak >nul
if exist "venv_shared" (
    call venv_shared\Scripts\activate.bat
    pip install -r ..\requirements_shared_graph.txt
)
popd

REM Setup LangFuse environment
echo.
echo [SETUP] Setting up LangFuse environment
pushd langfuse
direnv allow
timeout /t 3 /nobreak >nul
popd

REM Setup LangGraph environment
echo.
echo [SETUP] Setting up LangGraph environment
pushd langgraph
direnv allow
timeout /t 3 /nobreak >nul
popd

echo    [OK] Shared graph environment setup complete

echo.
echo [SUCCESS] All environments have been set up successfully!

REM Register Jupyter kernels (if the script exists)
echo.
echo [JUPYTER] Registering Jupyter kernels
if exist "register_kernels.sh" (
    echo    Note: register_kernels.sh is a bash script. 
    echo    Please run it manually in WSL, Git Bash, or PowerShell with bash support.
    echo    Alternatively, create a register_kernels.bat for native Windows support.
) else (
    echo    register_kernels.sh not found, skipping kernel registration
)

REM Create Jupyter configurations
echo.
echo [CONFIG] Creating Jupyter configurations
if exist "create_notebook_config.py" (
    python create_notebook_config.py
) else (
    echo    create_notebook_config.py not found, skipping configuration creation
)

echo.
echo [SUMMARY] Summary of environments
echo    1. Root (\) - Basic LangChain RAG setup - Root (LangChain RAG)
echo    2. LangChain (\langchain) - Comprehensive LangChain learning - LangChain (Complete)
echo    3. DeepEval (\deepeval\deepeval_claude_created) - Model evaluation - DeepEval (Evaluation)
echo    4. COTTON (\Paper\1.3.3) - ML/PyTorch for code generation - COTTON (ML/PyTorch)
echo    5. AI-Papers (\AI-Papers) - PDF processing utilities - AI-Papers (PDF Utils)
echo    6. Shared (\ETL, \langfuse, \langgraph) - Graph-based applications - Graph (LangGraph/LangFuse)
echo.
echo [INFO] Usage
echo    - Navigate to any directory to automatically activate its environment
echo    - Use 'direnv allow' to permit new .envrc files
echo    - Use 'direnv deny' to disable direnv for a directory
echo    - Jupyter notebooks will automatically suggest the correct kernel for each directory
echo.
echo [JUPYTER] Jupyter Integration
echo    - Each directory has a registered Jupyter kernel (if register_kernels ran successfully)
echo    - New notebooks will show kernel suggestions based on location
echo    - Use Kernel Change Kernel menu to switch if needed
echo.
echo [CONFIG] To customize environments
echo    - Edit the .envrc files in each directory
echo    - Modify requirements.txt files as needed
echo    - Run 'direnv reload' after making changes
echo    - Re-run register_kernels script after adding new dependencies to kernels
echo.
echo [WINDOWS] Windows-specific notes
echo    - Use Command Prompt or PowerShell to run this script
echo    - For best direnv experience, consider using PowerShell with direnv hook
echo    - Some features may work better in WSL (Windows Subsystem for Linux)

pause