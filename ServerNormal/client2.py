import socket
import json
import time
import psutil
import os
import threading

def send_request(text, source_lang, target_lang):
    request = {
        "text": text,
        "source_lang": source_lang,
        "target_lang": target_lang
    }
    request_data = json.dumps(request)

    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect(('127.0.0.1', 8003))

    client_socket.send(request_data.encode())
    response_data = client_socket.recv(4096).decode()
    response = json.loads(response_data)

    client_socket.close()
    return response

def measure_metrics(text, source_lang, target_lang, results, index):
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / (1024 * 1024)

    start_time = time.time()

    response = send_request(text, source_lang, target_lang)

    end_time = time.time()

    final_memory = process.memory_info().rss / (1024 * 1024)

    latency = end_time - start_time
    memory_used = final_memory - initial_memory
    throughput = len(text) / latency

    results[index] = (response, latency, memory_used, throughput, text)

def process_file(file_path, source_lang, target_lang):
    with open(file_path, 'r') as file:
        sentences = file.readlines()

    results = [None] * len(sentences)
    threads = []

    for i, sentence in enumerate(sentences):
        thread = threading.Thread(target=measure_metrics, args=(sentence.strip(), source_lang, target_lang, results, i))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    total_latency = 0
    for result in results:
        response, latency, memory_used, throughput, original_text = result
        translated_text = response.get('translated_text', 'Translation not available')
        print(f"Original: {original_text}")
        print(f"Translated: {translated_text}")
        print(f"Latency: {latency * 1000:.2f} ms")
        print("---")
        total_latency += latency

    avg_latency = (total_latency / len(sentences)) * 1000
    throughput = len(sentences) / total_latency
    print(f"Processed {len(sentences)} sentences in {total_latency:.2f} seconds.")
    print(f"Throughput: {throughput:.2f} sentences/second")
    print(f"Average Latency: {avg_latency:.2f} milliseconds")

if __name__ == "__main__":
    file_path = 'sentences.txt'  # Replace with your file path
    
    source_lang = input("Enter the source language code (e.g., 'eng_Latn' for English): ")
    target_lang = input("Enter the target language code (e.g., 'asm_Beng' for Assamese): ")

    process_file(file_path, source_lang, target_lang)

