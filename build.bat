@echo off
rem Build on Windows from a normal terminal: finds Visual Studio's compiler, then runs nvcc.
for /f "usebackq delims=" %%i in (`"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" -latest -property installationPath`) do set VSDIR=%%i
call "%VSDIR%\VC\Auxiliary\Build\vcvars64.bat" >nul || exit /b 1
nvcc -O3 laplace.cu -o laplace.exe -lcublas
