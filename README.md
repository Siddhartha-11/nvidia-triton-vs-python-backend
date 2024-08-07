# NVIDIA Triton vs Python Backend

This repository contains the requirements for deploying an NLLB model in NVIDIA Triton Inference Server and a standalone Python-based server for deep learning model inference performance and scalability.

**Triton container version used:** `nvcr.io/nvidia/tritonserver:24.01-py`

## Quick Steps to Serve the Model
```bash
### Step 1: Clone the Repository
git clone https://github.com/Siddhartha-11/nvidia-triton-vs-python-backend.git
cd nvidia-triton-vs-python-backend

### Step 2: Launch Triton from the NGC Triton Container and Deploy the Model
docker run --rm --gpus=all -p8000:8000 -p8001:8001 -p8002:8002 \
  -v$(pwd)/Modelrepo/nllb:/models/nllb \
  --name triton-client nvcr.io/nvidia/tritonserver:24.01-py3 \
  tritonserver --model-repository=/models

### Step 3: Deploy the Model in Standard Server
python3 model.py

### Step 4: Sending Inference request
##NVIDIA Triton Server
cd python script file
##To enter a text and get the returned translated text:
python3 entertext2.py
##To send a text file containing 11 sentences for translation:
python3 triton_translation.py

##Standard Server
cd ServerNormal
##To enter a text and get the returned translated text:
python3 client.py
##To send a text file containing 11 sentences for translation:
python3 client2.py
```

To set up Docker on your system, follow the official Docker installation guide for your operating system : [Install Docker](https://www.docker.com/)

To know more about NVIDIA Triton Inference server, follow the link to official NVIDIA Triton Inference Server guide : [NVIDIA Triton](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/index.html)

List of all the Triton Inference Server Container Image : [Containers](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tritonserver)

Model used : [NLLB](https://huggingface.co/facebook/nllb-200-distilled-600M)
