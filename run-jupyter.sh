#!/bin/bash

. activate

conda activate rs-model

# Tokenless access - don't use for public-facing servers!
# See: https://github.com/jupyter/notebook/issues/2254#issuecomment-321189274
jupyter notebook --generate-config
echo "c.NotebookApp.token = ''" > /root/.jupyter/jupyter_notebook_config.py

jupyter notebook --ip='0.0.0.0' --port=8888 --allow-root
