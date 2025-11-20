FROM --platform=linux/amd64 pytorch/pytorch:2.5.0-cuda11.8-cudnn9-runtime
# Using a 'large' base container to show-case how to load pytorch and use the GPU (when enabled)

# Ensures that Python output to stdout/stderr is not buffered: prevents missing information when terminating
ENV PYTHONUNBUFFERED=1
ENV PYTHONWARNINGS="ignore"

# Install git so we can clone the nnunet repository
# RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

RUN groupadd -r user && useradd -m --no-log-init -r -g user user
RUN mkdir -p /opt/algorithm
RUN chown -R user /opt/algorithm
ENV PATH="/home/user/.local/bin:${PATH}"
USER user

COPY --chown=user:user requirements.txt /opt/app/

# You can add any Python dependencies to requirements.txt
RUN python -m pip install \
    --user \
    --no-cache-dir \
    -i https://pypi.tuna.tsinghua.edu.cn/simple \
    --no-color \
    --requirement /opt/app/requirements.txt


### Clone nnUNet
# Configure Git, clone the repository without checking out, then checkout the specific commit
# RUN git config --global advice.detachedHead false && \
# git clone https://github.com/MIC-DKFZ/nnUNet.git /opt/algorithm/nnunet/ 

COPY --chown=user:user ./nnUNet/ /opt/algorithm/nnunet/

### Install local MRSegmentator
COPY --chown=user:user ./MRSegmentator/ /opt/algorithm/MRSegmentator/

# Install both local packages and additional dependencies
RUN pip3 install --user \
        -e /opt/algorithm/nnunet \
        -e /opt/algorithm/MRSegmentator \
        graphviz \
        onnx \
        SimpleITK && \
    rm -rf ~/.cache/pip

COPY --chown=user:user ./nnUNet_results/ /opt/algorithm/nnunet/nnUNet_results/

WORKDIR /opt/app

COPY --chown=user:user inference.py /opt/app/
COPY --chown=user:user data_utils.py /opt/app/

### Set environment variable defaults
ENV nnUNet_raw="/opt/algorithm/nnunet/nnUNet_raw" \
    nnUNet_preprocessed="/opt/algorithm/nnunet/nnUNet_preprocessed" \
    nnUNet_results="/opt/algorithm/nnunet/nnUNet_results"

ENTRYPOINT ["python", "inference.py"]
