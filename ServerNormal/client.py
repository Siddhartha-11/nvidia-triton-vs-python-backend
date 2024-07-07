import socket
import json
import time
import psutil
import os

def send_request(text, source_lang, target_lang):
    # Create the request dictionary
    request = {
        "text": text,
        "source_lang": source_lang,
        "target_lang": target_lang
    }
    # Convert the request to JSON format
    request_data = json.dumps(request)

    # Create a TCP socket
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Connect to the server on port 8000
    client_socket.connect(('127.0.0.1', 8003))

    # Send the request data to the server
    client_socket.send(request_data.encode())

    # Receive the response data from the server
    response_data = client_socket.recv(1024).decode()
    # Parse the JSON response
    response = json.loads(response_data)

    # Close the socket connection
    client_socket.close()
    return response

def measure_metrics(text, source_lang, target_lang):
    # Measure initial memory usage
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / (1024 * 1024)  # Convert to MB

    # Measure start time
    start_time = time.time()

    # Send the translation request to the server
    response = send_request(text, source_lang, target_lang)

    # Measure end time
    end_time = time.time()

    # Measure final memory usage
    final_memory = process.memory_info().rss / (1024 * 1024)  # Convert to MB

    # Calculate latency and memory used
    latency = end_time - start_time
    memory_used = final_memory - initial_memory

    # Calculate throughput (characters per second)
    throughput = len(text) / latency

    return response, latency, memory_used, throughput

# Main function to get user input and send the request
if __name__ == "__main__":
    # Get the text and language details from the user
    text = input("Enter the text to be translated: ")
    source_lang = input("Enter the source language code (e.g., 'eng_Latn' for English): ")
    target_lang = input("Enter the target language code (e.g., 'asm_Beng' for Assamese): ")

    # Measure metrics and get the response
    response, latency, memory_used, throughput = measure_metrics(text, source_lang, target_lang)

    # Print the translated text and metrics
    if 'translated_text' in response:
        print("Translated text:", response['translated_text'])
    else:
        print("Error:", response.get('error', 'Unknown error occurred'))

    print(f"Latency: {latency:.2f} seconds")
    print(f"Memory used: {memory_used:.2f} MB")
    print(f"Throughput: {throughput:.2f} characters per second")

