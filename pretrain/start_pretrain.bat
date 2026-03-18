@echo off
REM Kronos Pretraining Script for Windows
REM Kronos 预训练脚本（Windows 版）

echo ============================================================
echo Kronos Pretraining / Kronos 预训练
echo ============================================================
echo.

REM Check Python installation
where python >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Python not found. Please install Python 3.10+
    echo 错误：未找到 Python，请安装 Python 3.10+
    pause
    exit /b 1
)

python --version
echo.

REM Activate virtual environment if exists
if exist ".venv\Scripts\activate.bat" (
    echo Activating virtual environment...
    call .venv\Scripts\activate.bat
    echo.
) else if exist "..\..\.venv\Scripts\activate.bat" (
    echo Activating virtual environment from parent directory...
    call ..\..\..venv\Scripts\activate.bat
    echo.
)

REM Check PyTorch installation
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')" 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: PyTorch not installed. Please install PyTorch >= 2.0.0
    echo 错误：未安装 PyTorch，请安装 PyTorch >= 2.0.0
    echo.
    echo Install with: pip install torch torchvision torchaudio
    pause
    exit /b 1
)
echo.

REM Parse command line arguments
set CONFIG=config.yaml
set DISTRIBUTED=false
set NPROC=1

:parse_args
if "%~1"=="" goto :end_parse_args
if /i "%~1"=="--config" (
    set CONFIG=%~2
    shift
    shift
    goto :parse_args
)
if /i "%~1"=="--distributed" (
    set DISTRIBUTED=true
    shift
    goto :parse_args
)
if /i "%~1"=="--nproc" (
    set NPROC=%~2
    shift
    shift
    goto :parse_args
)
if /i "%~1"=="--help" (
    echo Usage: start_pretrain.bat [options]
    echo 用法：start_pretrain.bat [选项]
    echo.
    echo Options:
    echo   --config FILE    Configuration file (default: config.yaml)
    echo   --distributed    Enable distributed training
    echo   --nproc N        Number of GPUs (default: 1)
    echo   --help           Show this help message
    echo.
    exit /b 0
)
:end_parse_args

echo ============================================================
echo Configuration / 配置
echo ============================================================
echo Config file: %CONFIG%
echo Distributed: %DISTRIBUTED%
echo Processes: %NPROC%
echo.

REM Check if config file exists
if not exist "%CONFIG%" (
    echo ERROR: Configuration file not found: %CONFIG%
    echo 错误：配置文件不存在：%CONFIG%
    pause
    exit /b 1
)

REM Run pretraining
echo ============================================================
echo Starting pretraining... / 开始预训练...
echo ============================================================
echo.

if "%DISTRIBUTED%"=="true" (
    echo Running distributed training with %NPROC% GPUs...
    torchrun --nproc_per_node=%NPROC% pretrain_kronos.py --config %CONFIG%
) else (
    echo Running single GPU/CPU training...
    python pretrain_kronos.py --config %CONFIG%
)

set EXIT_CODE=%ERRORLEVEL%

echo.
echo ============================================================
if %EXIT_CODE% EQU 0 (
    echo Pretraining completed successfully! / 预训练成功完成！
) else (
    echo Pretraining failed with exit code %EXIT_CODE% / 预训练失败，退出码：%EXIT_CODE%
)
echo ============================================================
echo.

pause
exit /b %EXIT_CODE%
