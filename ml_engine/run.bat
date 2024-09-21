@echo off
setlocal

if "%1"=="d" (
    echo "Debug Build"
	pip install -e .
) else if "%1"=="r" (
    echo "Release Build"
	python -m build
) else if "%1"=="docs" (
    echo "Generating Docs"
  sphinx-apidoc -o .\sphinx_docs\source\ .\tiny_gpt\
  .\sphinx_docs\make.bat html
  .\sphinx_docs\copy_files.bat
	REM .\sphinx_docs\make.bat latexpdf
) else (
    echo Invalid argument. Please provide either "d" or "r" or "docs".
)



