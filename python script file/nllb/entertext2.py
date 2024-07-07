import asyncio
import tritonclient.grpc.aio
from tritonclient.utils import np_to_triton_dtype
from transformers import AutoTokenizer
import numpy as np

async def main():
    MODEL_NAME = "nllb"
    client = tritonclient.grpc.aio.InferenceServerClient("127.0.0.1:8001")
    
    en_text = input("Enter text to be translated: ")
    tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M", src_lang="en")
    
    inputs = tokenizer(en_text, return_attention_mask=True, return_tensors="np")
    input_ids = inputs.input_ids.astype(np.int32)
    attention_mask = inputs.attention_mask.astype(np.int32)
    
    print(f"Tokenized input_ids: {input_ids}")
    print(f"Attention mask: {attention_mask}")
    
    infer_inputs = [
        tritonclient.grpc.aio.InferInput("INPUT_IDS", input_ids.shape, np_to_triton_dtype(input_ids.dtype)),
        tritonclient.grpc.aio.InferInput("ATTENTION_MASK", attention_mask.shape, np_to_triton_dtype(attention_mask.dtype))
    ]
    infer_inputs[0].set_data_from_numpy(input_ids)
    infer_inputs[1].set_data_from_numpy(attention_mask)
    
    outputs = [tritonclient.grpc.aio.InferRequestedOutput("OUTPUT_IDS")]
    
    res = await client.infer(model_name=MODEL_NAME, inputs=infer_inputs, outputs=outputs)
    out_tokens = res.as_numpy("OUTPUT_IDS")
    
    print(f"Returned tokens: {out_tokens}")
    translated_text = tokenizer.batch_decode(out_tokens, skip_special_tokens=True)
    print(f"Translated text: {translated_text}")

if __name__ == "__main__":
    asyncio.run(main())

