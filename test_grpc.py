import asyncio
import json
import sys
import numpy as np
import tritonclient.grpc.aio as grpcclient

class VLLMClient:
    def __init__(self, url="localhost:9002", model_name="vllm_model"):
        self.url = url
        self.model_name = model_name
        self.results = []

    def get_client(self):
        try:
            return grpcclient.InferenceServerClient(url=self.url, verbose=False)
        except Exception as e:
            print(f"Failed to create client: {e}")
            sys.exit(1)

    async def request_iterator(self, prompt, streaming=False):
        """Generator tạo request cho streaming"""
        # Text input
        text_input = grpcclient.InferInput("text_input", [1], "BYTES")
        text_input.set_data_from_numpy(
            np.array([prompt.encode("utf-8")], dtype=np.object_)
        )

        # Stream flag
        stream_input = grpcclient.InferInput("stream", [1], "BOOL")
        stream_input.set_data_from_numpy(np.array([streaming], dtype=bool))

        # Sampling parameters
        sampling_params = {
            "temperature": 0.7,
            "top_p": 0.95,
            "max_tokens": 100
        }
        sampling_input = grpcclient.InferInput("sampling_parameters", [1], "BYTES")
        sampling_input.set_data_from_numpy(
            np.array([json.dumps(sampling_params).encode("utf-8")], dtype=np.object_)
        )

        # Output
        output = grpcclient.InferRequestedOutput("text_output")

        # Yield request
        yield {
            "model_name": self.model_name,
            "inputs": [text_input, stream_input, sampling_input],
            "outputs": [output],
            "request_id": "1"
        }

    async def infer_streaming(self, prompt):
        """Inference với streaming"""
        client = self.get_client()
        self.results = []
        
        print(f"Prompt: {prompt}")
        print("\nGenerating (streaming):")
        
        try:
            # Sử dụng stream_infer với request iterator
            response_iterator = client.stream_infer(
                inputs_iterator=self.request_iterator(prompt, streaming=True)
            )
            
            # Đọc responses từ stream
            async for response in response_iterator:
                result, error = response
                if error:
                    print(f"Error: {error}")
                else:
                    output = result.as_numpy("text_output")
                    text = output[0].decode("utf-8")
                    print(text, end='', flush=True)
                    self.results.append(text)
            
            print("\n\nDone!")
            return True
            
        except Exception as e:
            print(f"Error during inference: {e}")
            return False

    async def infer_non_streaming(self, prompt):
        """Inference không streaming (nhưng vẫn phải dùng stream_infer)"""
        client = self.get_client()
        self.results = []
        
        print(f"Prompt: {prompt}")
        print("\nGenerating (non-streaming)...\n")
        
        try:
            # Vẫn phải dùng stream_infer cho decoupled models
            response_iterator = client.stream_infer(
                inputs_iterator=self.request_iterator(prompt, streaming=False)
            )
            
            async for response in response_iterator:
                result, error = response
                if error:
                    print(f"Error: {error}")
                else:
                    output = result.as_numpy("text_output")
                    text = output[0].decode("utf-8")
                    self.results.append(text)
            
            print("Output:", ''.join(self.results))
            print("\nDone!")
            return True
            
        except Exception as e:
            print(f"Error during inference: {e}")
            return False

async def main():
    client = VLLMClient()
    
    # Test non-streaming
    prompt = "Once upon a time, in a land far away"
    await client.infer_non_streaming(prompt)
    
    print("\n" + "="*50 + "\n")
    
    # Test streaming
    await client.infer_streaming(prompt)

if __name__ == "__main__":
    asyncio.run(main())
