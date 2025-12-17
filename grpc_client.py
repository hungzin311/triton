import argparse
import tritonclient.grpc.aio as grpcclient
import numpy as np 
import asyncio
import json 
import sys  
from tritonclient.utils import *

class LLMClient: 
    def __int__(self, flags: argparse.Namespace):
        self._flags = flags 
        self._results_dict = {}
    
    def get_triton_client(self): 
        try:
            triton_client = grpcclient.InferenceServerClient(
                url = self._flags.url, 
                verbose = self._flags.verbose
            )
        except Exception as e:
            print("Error: ", e)
            sys.exit(1)
        
        return triton_client
    
    async def async_request_iterator(self, prompts, sampling_parameters, exclude_input_output): 
        try: 
            for iter in range(self._flags.iterations):
                for i, prompt in enumerate(prompts): 
                    prompt_id = self. _flags.offset + len(prompts) * iter + i
                    self._results_dict[prompt_id] = []
                    yield self.create_request(
                        prompt, 
                        self._flags.streaming_mode, 
                        prompt_id, 
                        sampling_parameters, 
                        exclude_input_output
                    )

        except Exception as e:
            print(f"Caught an error in the request iterator: {e}")

    async def stream_infer(self, prompts, sampling_parameter, exclude_input_in_output): 
        try:
            triton_client = self.get_triton_client()
            response_iterator = triton_client.stream_infer(
                inputs_generator = self.async_request_iterator(prompts, sampling_parameter, exclude_input_in_output), 
                stream_timeout = self._flags.stream_timeout
            )

            async for response in response_iterator:
                yield response

        except Exception as e:
            print(e)
            sys.exit(1)

    async def process_stream(self, prompts, sampling_parameter, exclude_input_in_output): 
        self._results_dict = {}
        success = True 

        async for response in self.stream_infer(prompts, sampling_parameter, exclude_input_in_output):
            result, error = response 
            if error: 
                print(f"Encountered an error: {error}")
                success = False 
            else: 
                output = result.as_numpy("text_output")
                for i in output: 
                    self._results_dict[result.get_response().id].append(i)
        
        return success
    
    async def run(self): 
        sampling_parameters = {
            "temperature": "0.1",
            "top_p": "0.95",
            "max_tokens": "100",
        }
        exclude_input_in_output = self._flags.exclude_inputs_in_outputs
        if self._flags.lora_name is not None:
            sampling_parameters["lora_name"] = self._flags.lora_name
        
        prompts = [self._flags.prompt]
        success = await self.process_stream(prompts, sampling_parameters, exclude_input_in_output)
        if success:
            print("Inference completed successfully")
            for prompt_id, output in self._results_dict.items():
                print(f"Prompt ID: {prompt_id}")
                print(f"Output: {output}")
        else:
            print("Inference failed")
    
    def run_async(self): 
        asyncio.run(self.run())
    
    def create_request(self, prompt, stream, request_id, sampling_parameters, exclude_input_output, send_parameters_as_tensor=True):
        inputs= []
        prompt_data = np.array([prompt.encode('utf-8')], dtype = np.object_)

        try: 
            inputs.append(grpcclient.InferInput('text_input', [1], "BYTES"))
            inputs[-1].set_data_from_numpy(prompt_data)
        except Exception as e:
            print(f"Error adding text_input: {e}")
            sys.exit(1)
        
        stream_data = np.array([stream], dtype = np.bool_)
        inputs.append(grpcclient.InferInput('stream', [1], "BOOL"))
        inputs[-1].set_data_from_numpy(stream_data)

        if send_parameters_as_tensor:
            sampling_parameters_data = np.array([sampling_parameters.encode('utf-8')], dtype = np.object_)
            inputs.append(grpcclient.InferInput('sampling_parameters', [1], "BYTES"))
            inputs[-1].set_data_from_numpy(sampling_parameters_data)
        
        inputs.append(grpcclient.InferRequestedOutput('exclude_input_in_output', [1], "BOOL"))
        inputs[-1].set_data_from_numpy(np.array([exclude_input_output], dtype = bool))

        outputs = []
        outputs.append(grpcclient.InferRequestedOutput('text_output'))

        return{ 
            "model_name": self._flags.model,
            "inputs": inputs, 
            "outputs": outputs, 
            "request_id": str(request_id), 
            "parameters": sampling_parameters,
        }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, required=False, default="vllm_model", help="Model name")
    parser.add_argument("-v", "--verbose", type=bool, required=False, default=False, help="Verbose")
    parser.add_argument("-u", "--url", type=str, required=False, default="localhost:9001", help="URL")
    parser.add_argument("-t", "--stream-timeout", type=float, required=False, default=None, help="Stream timeout")
    parser.add_argument("-p", "--prompt", type=str, required=False, default=None, help="Prompt")
    parser.add_argument("--iterations", type=int, required=False, default=1, help="Iterations")
    parser.add_argument("-s", "--streaming-mode", action = "store_true", required=False, default=False, help="Streaming mode")
    parser.add_argument("--exclude-inputs-in-outputs", action = "store_true", required=False, default=False, help="Exclude inputs in outputs")
    parser.add_argument("-l", "--lora-name", type=str, required=False, default=None, help="Lora name")
    parser.add_argument("--offset", type=int, required=False, default=0, help="Offset")

    FLAGS = parser.parse_args()
    client = LLMClient(FLAGS)