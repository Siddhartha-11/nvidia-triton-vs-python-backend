import socket
import json
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class NLLBInference:
    def __init__(self, model_name="facebook/nllb-200-distilled-600M"):
        # Load the tokenizer and model from Hugging Face
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        # Print available language codes for debugging
        self.lang_code_to_id = self.tokenizer.lang_code_to_id
        print("Available language codes:", self.lang_code_to_id)

    def translate(self, text, source_lang, target_lang):
        # Ensure the target language code is valid
        if target_lang not in self.lang_code_to_id:
            raise ValueError(f"Target language code '{target_lang}' is not supported.")

        # Set the source language for the tokenizer
        self.tokenizer.src_lang = source_lang

        # Tokenize the input text
        inputs = self.tokenizer(text, return_tensors="pt", padding=True)
        # Generate the translation
        translated_tokens = self.model.generate(
            **inputs,
            forced_bos_token_id=self.lang_code_to_id[target_lang]
        )
        # Decode the tokens to get the translated text
        return self.tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]

def handle_client(client_socket):
    try:
        # Receive data from the client
        data = client_socket.recv(1024).decode()
        # Parse the JSON data
        request = json.loads(data)
        
        text = request['text']
        source_lang = request['source_lang']
        target_lang = request['target_lang']

        # Perform model inference
        translated_text = inference.translate(text, source_lang, target_lang)

        # Send the response back to the client
        response = json.dumps({'translated_text': translated_text})
        client_socket.send(response.encode())
    except Exception as e:
        # Handle any exceptions and send an error response
        response = json.dumps({'error': str(e)})
        client_socket.send(response.encode())
    finally:
        # Close the client connection
        client_socket.close()

if __name__ == "__main__":
    # Initialize the NLLB inference class
    inference = NLLBInference()

    # Create a TCP socket
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Bind the socket to an IP address and port 8000
    server_socket.bind(('127.0.0.1', 8003))
    # Start listening for incoming connections
    server_socket.listen(5)

    print("Server listening on port 8003")

    while True:
        # Accept a new client connection
        client_socket, addr = server_socket.accept()
        print(f"Connection from {addr}")
        # Handle the client connection
        handle_client(client_socket)

