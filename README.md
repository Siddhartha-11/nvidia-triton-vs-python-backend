# NVIDIA Triton vs Python Backend

This repository contains a comparative analysis of NVIDIA Triton Inference Server and a standalone Python-based server for deep learning model inference performance and scalability.

**Triton container version used:** `nvcr.io/nvidia/tritonserver:24.01-py`

## Quick Steps to Serve the Model
```bash
### Step 1: Clone the Repository
git clone https://github.com/Siddhartha-11/nvidia-triton-vs-python-backend.git
cd nvidia-triton-vs-python-backend

### Step 2: Launch Triton from the NGC Triton Container and Deploy the Model
docker run --rm -p8000:8000 -p8001:8001 -p8002:8002 \
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
cs ServerNormal
##To enter a text and get the returned translated text:
python3 client.py
##To send a text file containing 11 sentences for translation:
python3 client2.py

