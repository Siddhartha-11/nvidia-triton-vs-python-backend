import socket
import json
import time
import psutil
import os

BATCH_SIZE = 5  # Number of sentences to send in each batch
TIMEOUT_SECONDS = 10  # Timeout in seconds for batch processing

def send_batch_request(batch_requests):
    request_data = json.dumps(batch_requests)

    try:
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect(('127.0.0.1', 8003))

        client_socket.send(request_data.encode())
        client_socket.settimeout(TIMEOUT_SECONDS)
        response_data = client_socket.recv(4096).decode()

        client_socket.close()

        try:
            responses = json.loads(response_data)
        except json.JSONDecodeError as e:
            # Handle the case where the response is not JSON
            print(f"Error decoding JSON: {e}")
            responses = []
    except ConnectionRefusedError:
        print("Connection to translation service refused.")
        responses = []
    except socket.timeout:
        print(f"Socket timed out after {TIMEOUT_SECONDS} seconds.")
        responses = []
    except Exception as ex:
        print(f"Exception occurred: {ex}")
        responses = []

    return responses

def measure_metrics(batch_requests, results, index):
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / (1024 * 1024)

    start_time = time.time()

    responses = send_batch_request(batch_requests)

    end_time = time.time()

    final_memory = process.memory_info().rss / (1024 * 1024)

    latency = end_time - start_time
    memory_used = final_memory - initial_memory
    throughput = len(batch_requests) / latency

    for i, response in enumerate(responses):
        if isinstance(response, str):
            response = {'translated_text': response}  # Handle case where response is a string
        original_text = batch_requests[i]['text']
        translated_text = response.get('translated_text', 'Translation not available')
        results[index + i] = (response, latency, memory_used, throughput, original_text)

def process_file(file_path, source_lang, target_lang):
    with open(file_path, 'r') as file:
        sentences = file.readlines()

    results = [None] * len(sentences)
    batch_requests = []
    total_sentences = len(sentences)

    for i, sentence in enumerate(sentences):
        batch_requests.append({
            'text': sentence.strip(),
            'source_lang': source_lang,
            'target_lang': target_lang
        })

        if len(batch_requests) == BATCH_SIZE or i == total_sentences - 1:
            print(f"Processing {len(batch_requests)} sentences with dynamic batching...")
            measure_metrics(batch_requests, results, i - len(batch_requests) + 1)
            batch_requests = []

    total_latency = 0
    processed_count = 0  # Track how many sentences were actually processed
    batch_index = 0

    while batch_index < total_sentences:
        batch_size = min(BATCH_SIZE, total_sentences - batch_index)
        print(f"Processing {batch_size} sentences with dynamic batching...")

        for result_index in range(batch_index, batch_index + batch_size):
            result = results[result_index]
            if result is not None:  # Check if result is not None
                response, latency, memory_used, throughput, original_text = result
                translated_text = response.get('translated_text', 'Translation not available')
                print(f"Original: {original_text}")
                print(f"Translated: {translated_text}")
                print(f"Latency: {latency * 1000:.2f} ms")
                print("---")
                total_latency += latency
                processed_count += 1
            else:
                print("Failed to process a sentence.")

        batch_index += BATCH_SIZE

    if processed_count > 0:
        avg_latency = (total_latency / processed_count) * 1000
        throughput = processed_count / total_latency
        print(f"Processed {processed_count} sentences in {total_latency:.2f} seconds.")
        print(f"Throughput: {throughput:.2f} sentences/second")
        print(f"Average Latency: {avg_latency:.2f} milliseconds")
    else:
        print("No sentences were processed successfully.")

if __name__ == "__main__":
    file_path = 'sentences.txt'  # Replace with your file path
    
    source_lang = input("Enter the source language code (e.g., 'eng_Latn' for English): ")
    target_lang = input("Enter the target language code (e.g., 'asm_Beng' for Assamese): ")

    print("Starting dynamic batching...")
    process_file(file_path, source_lang, target_lang)

