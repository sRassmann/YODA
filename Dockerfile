FROM projectmonai/monai:1.3.0
RUN pip3 install torch==2.2.0 torchvision==0.17.0 xformers --index-url https://download.pytorch.org/whl/cu118

# Set your working directory
WORKDIR /workspace

# Install additional system dependencies and nice-to-have tools
RUN apt-get update && apt-get install -y git  && apt-get install -y tree

RUN git clone https://github.com/Project-MONAI/GenerativeModels.git
WORKDIR /workspace/GenerativeModels
RUN python setup.py install

# Install additional dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Change the working directory diffusion
WORKDIR /workspace/diffusion

ARG UID
ARG GID
ARG USER

RUN groupadd -g $GID -o $USER
RUN useradd -m -u $UID -g $GID -o -s /bin/bash $USER
USER $USER

# Expose any necessary ports
EXPOSE 8080

# Specify the command to run when the container starts
CMD ["bash"]



