<#
Prepares the isolated Python environment used to package the NVIDIA/CUDA
edition of APVD. Run from PowerShell in this project folder:

    .\prepare_cuda_release.ps1

It does not require users of APVD to install the CUDA Toolkit. The resulting
application still needs a working NVIDIA driver on the user's computer.
#>

[CmdletBinding()]
param(
    [string]$PythonCommand = "python"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvPath = Join-Path $ProjectRoot ".venv-apvd-cuda"
$VenvPython = Join-Path $VenvPath "Scripts\python.exe"

if (-not (Test-Path $VenvPython)) {
    Write-Host "Creating the APVD CUDA release environment..."
    & $PythonCommand -m venv $VenvPath
    if ($LASTEXITCODE -ne 0) {
        throw "Could not create the virtual environment with '$PythonCommand'. Install Python 3.9-3.12, then run this script again."
    }
}

Write-Host "Installing APVD's non-PyTorch dependencies..."
& $VenvPython -m pip install --upgrade pip
& $VenvPython -m pip install opencv-python Pillow numpy PyYAML mss psutil

Write-Host "Installing the CUDA-enabled PyTorch build..."
& $VenvPython -m pip install --upgrade --force-reinstall `
    torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 `
    --index-url https://download.pytorch.org/whl/cu121

Write-Host "Verifying CUDA access..."
& $VenvPython -c "import sys, torch; print('PyTorch:', torch.__version__); print('Built for CUDA:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NOT DETECTED'); sys.exit(0 if torch.cuda.is_available() else 1)"
if ($LASTEXITCODE -ne 0) {
    throw "CUDA verification failed. Do not package APVD from this environment. Confirm that this computer has a working NVIDIA driver, then run the script again."
}

Write-Host ""
Write-Host "CUDA environment is ready. Package APVD using:"
Write-Host "  $VenvPython app.py"
Write-Host "or use this same Python executable with your packaging command."
