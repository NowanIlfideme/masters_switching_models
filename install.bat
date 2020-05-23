:: This will generally work. 
:: May need to separate out the env for Windows users.

:: Windows installation order for Theano is wonky... may not work!
call conda env create -f env.yml -n rs-model
call activate rs-model

:: Remove this batch file - with it, performance is worse!
del %CONDA_PREFIX%\etc\conda\activate.d\vs2015_compiler_vars.bat

:: Install my package
python setup.py develop

:: Install Jupyter kernel
python -m ipykernel install --name rs-model

:: Install pre-commit hooks
:: NOTE: Make sure you have `%CONDA_PREFIX%\DLLs\sqlite3.dll`
:: If not, copy from your base conda installation
pre-commit install
pre-commit run --all-files

:: Re-activate
call conda deactivate
call activate rs-model

:: Check installation of pymc3
python -m regime_switching.utils.pymc.test_installation