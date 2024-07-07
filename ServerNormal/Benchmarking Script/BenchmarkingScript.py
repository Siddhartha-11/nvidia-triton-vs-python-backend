import asyncio
import time
import socket
import json
import psutil
import os
import tritonclient.grpc.aio
from tritonclient.utils import np_to_triton_dtype
from transformers import AutoTokenizer
import numpy as np

TRITON_SERVER_URL = "127.0.0.1:8001"
STANDARD_SERVER_URL = "127.0.0.1"
MODEL_NAME = "nllb"
SENTENCES_FILE = "sentences.txt"


# Benchmarking function for Triton Inference Server
async def benchmark_triton(texts, source_lang, target_lang):
    client = tritonclient.grpc.aio.InferenceServerClient(TRITON_SERVER_URL)
    tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M", src_lang=source_lang)
    
    latencies = []
    memory_usage = []

    for sentence in texts:
        # Tokenize sentence
        encoded = tokenizer(sentence, return_tensors="np", padding=True, truncation=True, max_length=128)

        input_ids = encoded['input_ids'].astype(np.int32)
        attention_mask = encoded['attention_mask'].astype(np.int32)
        token_type_ids = encoded.get('token_type_ids', None)  # Assuming the model might not need this
        
        # Create Triton inputs
        inputs = [
            tritonclient.grpc.aio.InferInput("INPUT_IDS", input_ids.shape, np_to_triton_dtype(input_ids.dtype)),
            tritonclient.grpc.aio.InferInput("ATTENTION_MASK", attention_mask.shape, np_to_triton_dtype(attention_mask.dtype)),
        ]
        
        inputs[0].set_data_from_numpy(input_ids)
        inputs[1].set_data_from_numpy(attention_mask)
        
        if token_type_ids is not None:
            inputs.append(tritonclient.grpc.aio.InferInput("TOKEN_TYPE_IDS", token_type_ids.shape, np_to_triton_dtype(token_type_ids.dtype)))
            inputs[-1].set_data_from_numpy(token_type_ids)
        
        outputs = [tritonclient.grpc.aio.InferRequestedOutput("OUTPUT_IDS")]

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / (1024 * 1024)
        start_time = time.time()

        res = await client.infer(model_name=MODEL_NAME, inputs=inputs, outputs=outputs)
        out_tokens = res.as_numpy("OUTPUT_IDS")

        latency = time.time() - start_time
        final_memory = process.memory_info().rss / (1024 * 1024)
        memory_used = final_memory - initial_memory

        latencies.append(latency)
        memory_usage.append(memory_used)

    average_latency = sum(latencies) / len(latencies)
    average_memory_used = sum(memory_usage) / len(memory_usage)
    throughput = len(latencies) / sum(latencies)

    return {
        "average_latency": average_latency,
        "average_memory_used": average_memory_used,
        "throughput": throughput
    }


# Standard server benchmarking function
def send_request(text, source_lang, target_lang):
    request = {
        "text": text,
        "source_lang": source_lang,
        "target_lang": target_lang
    }
    request_data = json.dumps(request)

    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((STANDARD_SERVER_URL, 8003))

    client_socket.send(request_data.encode())
    response_data = client_socket.recv(4096).decode()
    response = json.loads(response_data)

    client_socket.close()
    return response


def benchmark_standard(texts, source_lang, target_lang):
    latencies = []
    memory_usage = []

    for sentence in texts:
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / (1024 * 1024)
        start_time = time.time()

        response = send_request(sentence, source_lang, target_lang)

        latency = time.time() - start_time
        final_memory = process.memory_info().rss / (1024 * 1024)
        memory_used = final_memory - initial_memory

        latencies.append(latency)
        memory_usage.append(memory_used)

    average_latency = sum(latencies) / len(latencies)
    average_memory_used = sum(memory_usage) / len(memory_usage)
    throughput = len(latencies) / sum(latencies)

    return {
        "average_latency": average_latency,
        "average_memory_used": average_memory_used,
        "throughput": throughput
    }


# Comparative analysis function
def comparative_analysis(source_lang, target_lang):
    with open(SENTENCES_FILE, 'r') as file:
        texts = [line.strip() for line in file if line.strip()]

    print("Benchmarking Triton server...")
    triton_metrics = asyncio.run(benchmark_triton(texts, source_lang, target_lang))
    print("Triton server metrics:", triton_metrics)

    print("Benchmarking standard server...")
    standard_metrics = benchmark_standard(texts, source_lang, target_lang)
    print("Standard server metrics:", standard_metrics)

    print("\nComparative Analysis:")
    print(f"{'Metric':<25}{'Triton Server':<15}{'Standard Server':<15}")
    print(f"{'Average Latency (ms)':<25}{triton_metrics['average_latency'] * 1000:<15.2f}{standard_metrics['average_latency'] * 1000:<15.2f}")
    print(f"{'Average Memory Used (MB)':<25}{triton_metrics['average_memory_used']:<15.2f}{standard_metrics['average_memory_used']:<15.2f}")
    print(f"{'Throughput (sentences/sec)':<25}{triton_metrics['throughput']:<15.2f}{standard_metrics['throughput']:<15.2f}")


if __name__ == "__main__":
    source_lang = input("Enter the source language code (e.g., 'eng_Latn' for English): ")
    target_lang = input("Enter the target language code (e.g., 'asm_Beng' for Assamese): ")

    comparative_analysis(source_lang, target_lang)

