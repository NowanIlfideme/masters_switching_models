FROM continuumio/miniconda3:4.7.12

LABEL description="Container for regime_switching repo & LaTeX"
LABEL author="Anatoly Makarevich"

# Does the following:
# 2: Fixes g++ warning of Theano
# 3: Installation of LaTeX
# 4: dos2unix to change line endings for scripts
RUN apt-get update -y \
    && apt-get install -y \
    g++ \
    biber \
    latexmk \
    make \
    texlive-full \
    dos2unix \
    && apt-get clean

# Set workdir to be /mnt
WORKDIR /mnt

# Install base requirements (can take long!)
COPY env.yml /mnt/env.yml
RUN conda env create -f /mnt/env.yml -n rs-model

# Add nbextensions
RUN /bin/bash -c ". activate rs-model && jupyter contrib nbextension install --sys-prefix --symlink"
# Append to automatically activate rs-model env for users
RUN echo "conda activate rs-model" >> ~/.bashrc

# Add and install our package (can mount "over" it)
COPY . /mnt
RUN /bin/bash -c ". activate rs-model && python setup.py develop"

# This allows `docker run -it <container>`
# CMD ["/bin/bash"]

# This runs the jupyter notebook server
RUN dos2unix /mnt/run-jupyter.sh
RUN chmod +x /mnt/run-jupyter.sh
CMD ["/mnt/run-jupyter.sh"]
