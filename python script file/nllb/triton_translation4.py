import asyncio
import time
import psutil
import tritonclient.grpc.aio
from tritonclient.utils import np_to_triton_dtype
from transformers import AutoTokenizer
import numpy as np

async def main(filename="sentences.txt", model_name="nllb", server_address="127.0.0.1:8001"):
    """
    Translates sentences from a file and displays performance information, including memory usage.

    Args:
        filename: Path to the file containing sentences.
        model_name: Name of the Triton Inference Server model (default: "nllb").
        server_address: Address of the Triton Inference Server (default: "127.0.0.1:8001").

    Returns:
        None
    """
    start_time = time.time()
    sentence_count = 0
    total_latency = 0
    total_memory_used = 0

    client = tritonclient.grpc.aio.InferenceServerClient(server_address)
    tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M", src_lang="en")

    with open(filename, "r") as file:
        for line in file:
            sentence = line.strip()
            if sentence:
                # Tokenize the sentence
                start_translate = time.time()
                inputs = tokenizer(sentence, return_tensors="np", return_attention_mask=True)

                input_ids = inputs.input_ids.astype(np.int32)
                attention_mask = inputs.attention_mask.astype(np.int32)

                # Create Triton inputs
                inputs = [
                    tritonclient.grpc.aio.InferInput("INPUT_IDS", input_ids.shape, np_to_triton_dtype(input_ids.dtype)),
                    tritonclient.grpc.aio.InferInput("ATTENTION_MASK", attention_mask.shape, np_to_triton_dtype(attention_mask.dtype)),
                ]
                inputs[0].set_data_from_numpy(input_ids)
                inputs[1].set_data_from_numpy(attention_mask)

                outputs = [tritonclient.grpc.aio.InferRequestedOutput("OUTPUT_IDS")]

                # Measure memory before inference
                process = psutil.Process()
                mem_before = process.memory_info().rss

                # Perform inference
                res = await client.infer(model_name=model_name, inputs=inputs, outputs=outputs)
                out_tokens = res.as_numpy("OUTPUT_IDS")

                # Measure memory after inference
                mem_after = process.memory_info().rss
                memory_used = (mem_after - mem_before) / (1024 ** 2)  # Convert to MB
                total_memory_used += memory_used

                # Decode translated tokens and calculate latency
                end_translate = time.time()
                latency = (end_translate - start_translate) * 1000  # milliseconds
                total_latency += latency
                translated_text = tokenizer.batch_decode(out_tokens)

                # Print translated sentence and memory usage
                print(f"Original: {sentence}")
                print(f"Translated: {translated_text[0]}")
                print(f"Latency: {latency:.2f} ms")
                print(f"Memory Used: {memory_used:.2f} MB")
                print("---")

                sentence_count += 1

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

if __name__ == "__main__":
    asyncio.run(main())

