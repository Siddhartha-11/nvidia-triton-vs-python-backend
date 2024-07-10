# nvidia-triton-vs-python-backend
comparative analysis of NVIDIA triton inference server and standalone server for deep learning model's inference performance and scalability.

Triton container version used: nvcr.io/nvidia/tritonserver:24.01-py

Quick steps to serve the model:-
# Step 1: Clone the repository
git clone https://github.com/Siddhartha-11/nvidia-triton-vs-python-backend.git

# Step 2: Launch triton from the NGC Triton container and deploy the model
docker run --rm -p8000:8000 -p8001:8001 -p8002:8002 -v$(pwd)/Modelrepo/nllb:/models/nllb --name triton-client nvcr.io/nvidia/tritonserver:24.01-py3 tritonserver --model-repository=/models
 [ #To launch a interactive shell session in a separare console:
    docker exec -ti triton-client bash ]

# Step 3: Deploy the model in standard server
python3 model python.py

# Step 4: Sending an Inference Request
   1. NVIDIA Triton Server
    #enter a text and get the returned translated text: 
        In a separate console:
          python3 entertext2.py
    #sending a txt file containing 11 sentences for translation:
        In a separate console:
          python3 triton_translation.py
   2. Standard Server (Python-based backend)
    #enter a text and get returned the translated text
        python3 client.py
    #send a text file that contains 11 sentences
        python3 client2.py




