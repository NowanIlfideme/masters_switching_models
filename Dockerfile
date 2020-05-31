FROM continuumio/miniconda3:4.7.12

# Fixes g++ warning of Theano
RUN apt-get update -y && \
    apt-get install g++ -y && \
    apt-get install dos2unix -y && \
    apt-get clean

# Set workdir to be /mnt
WORKDIR /mnt

# Install base requirements
COPY env.yml /mnt/env.yml
RUN conda env create -f /mnt/env.yml -n rs-model

# Add nbextensions
RUN /bin/bash -c ". activate rs-model && jupyter contrib nbextension install --sys-prefix --symlink"
# Append to automatically activate rs-model env for users
RUN echo "conda activate rs-model" >> ~/.bashrc

# Add and install our package
COPY . /mnt
RUN /bin/bash -c ". activate rs-model && python setup.py develop"

# This allows `docker run -it <container>`
# CMD ["/bin/bash"]

# This runs the jupyter notebook server
RUN dos2unix /mnt/run-jupyter.sh
RUN chmod +x /mnt/run-jupyter.sh
CMD ["/mnt/run-jupyter.sh"]
