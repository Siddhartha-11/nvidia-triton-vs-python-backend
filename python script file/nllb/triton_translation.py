import tritonclient.grpc.aio
from tritonclient.utils import np_to_triton_dtype
from transformers import AutoTokenizer
import numpy as np

async def main(filename="sentences.txt", model_name="nllb", server_address="localhost:8001"):
    """
    Translates sentences from a file and displays performance information.

    Args:
        filename: Path to the file containing sentences.
        model_name: Name of the Triton Inference Server model (default: "nllb").
        server_address: Address of the Triton Inference Server (default: "localhost:8001").

    Returns:
        None
    """
    start_time = time.time()
    sentence_count = 0
    total_latency = 0

    try:
        client = tritonclient.grpc.aio.InferenceServerClient(server_address)
    except Exception as e:
        print(f"Error connecting to Triton server at {server_address}: {e}")
        return

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

                # Perform inference
                try:
                    res = await client.infer(model_name=model_name, inputs=inputs, outputs=outputs)
                    out_tokens = res.as_numpy("OUTPUT_IDS")

                    # Decode translated tokens and calculate latency
                    end_translate = time.time()
                    latency = (end_translate - start_translate) * 1000  # milliseconds
                    total_latency += latency
                    translated_text = tokenizer.batch_decode(out_tokens)

                    # Print translated sentence
                    print(f"Original: {sentence}")
                    print(f"Translated: {translated_text[0]}")
                    print(f"Latency: {latency:.2f} ms")
                    print("---")

                    sentence_count += 1
                except Exception as e:
                    print(f"Error performing inference for sentence: {sentence}")
                    print(f"Error message: {e}")

    end_time = time.time()
    total_time = end_time - start_time

    if sentence_count > 0:
        throughput = sentence_count / total_time
        avg_latency = total_latency / sentence_count
        print(f"\nProcessed {sentence_count} sentences in {total_time:.2f} seconds.")
        print(f"Throughput: {throughput:.2f} sentences/second")
        print(f"Average Latency: {avg_latency:.2f} milliseconds")
    else:
        print("No sentences found in the file.")

if __name__ == "__main__":
    asyncio.run(main())

