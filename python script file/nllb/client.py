import asyncio
import time
import tritonclient.grpc.aio
from tritonclient.utils import np_to_triton_dtype
from transformers import AutoTokenizer
import numpy as np
import psutil
import os

async def main(filename="sentences.txt", model_name="nllb", server_address="localhost:8001"):
    start_time = time.time()
    sentence_count = 0
    total_latency = 0
    total_memory_used = 0

    client = tritonclient.grpc.aio.InferenceServerClient(server_address)
    tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")

    try:
        with open(filename, "r") as file:
            sentences = [line.strip() for line in file if line.strip()]
        print(f"Read {len(sentences)} sentences from the file.")

        batch_size = 8
        padded_sentences = []

        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i + batch_size]
            inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="np")
            padded_sentences.append(inputs)

        async def translate_batch(input_ids, attention_mask):
            inputs = [
                tritonclient.grpc.aio.InferInput("INPUT_IDS", input_ids.shape, np_to_triton_dtype(input_ids.dtype)),
                tritonclient.grpc.aio.InferInput("ATTENTION_MASK", attention_mask.shape, np_to_triton_dtype(attention_mask.dtype)),
            ]
            inputs[0].set_data_from_numpy(input_ids)
            inputs[1].set_data_from_numpy(attention_mask)
            outputs = [tritonclient.grpc.aio.InferRequestedOutput("OUTPUT_IDS")]

            try:
                res = await client.infer(model_name=model_name, inputs=inputs, outputs=outputs)
                return res.as_numpy("OUTPUT_IDS")
            except tritonclient.utils.InferenceServerException as e:
                print(f"Inference error: {e}")
                return None

        for batch in padded_sentences:
            input_ids = batch["input_ids"].astype(np.int32)
            attention_mask = batch["attention_mask"].astype(np.int32)

            start_translate = time.time()
            out_tokens = await translate_batch(input_ids, attention_mask)

            if out_tokens is None:
                continue

            end_translate = time.time()
            latency = (end_translate - start_translate) * 1000  # milliseconds
            total_latency += latency
            sentences_translated = tokenizer.batch_decode(out_tokens, skip_special_tokens=True)

            print(f"Batch size: {len(input_ids)}")  # Print the batch size being processed
            for original, translated in zip(input_ids, sentences_translated):
                sentence_count += 1
                print(f"Original: {tokenizer.decode(original, skip_special_tokens=True)}")
                print(f"Translated: {translated}")
                print(f"Latency: {latency:.2f} ms")

                # Measure memory usage after translation of each sentence
                process = psutil.Process(os.getpid())
                memory_used = process.memory_info().rss / (1024 * 1024)  # in MB
                print(f"Memory used after translation: {memory_used:.2f} MB")
                total_memory_used += memory_used

                print("---")

        end_time = time.time()
        total_time = end_time - start_time

        if sentence_count > 0:
            throughput = sentence_count / total_time
            avg_latency = total_latency / sentence_count
            avg_memory_used = total_memory_used / sentence_count
            print(f"\nProcessed {sentence_count} sentences in {total_time:.2f} seconds.")
            print(f"Throughput: {throughput:.2f} sentences/second")
            print(f"Average Latency: {avg_latency:.2f} milliseconds")
            print(f"Average Memory Used: {avg_memory_used:.2f} MB")
        else:
            print("No sentences found in the file.")

    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
    except Exception as e:
        print(f"Error encountered: {e}")

if __name__ == "__main__":
    asyncio.run(main("sentences.txt"))

