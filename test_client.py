import tritonclient.http as httpclient
import numpy as np
import sys

if __name__ == "__main__":
    try:
        # Kết nối tới Triton Server
        triton_client = httpclient.InferenceServerClient(
            url="localhost:9001", 
            verbose=True
        )
        
        # Kiểm tra server có hoạt động không
        if not triton_client.is_server_live():
            print("Triton server is not live")
            sys.exit(1)
            
        print("Triton server is live!")
        
        model_repo = triton_client.get_model_repository_index()
        
        model_name = "vllm_model"  # Thay bằng tên model của bạn
        input_text ="Hello, what is triton?"
        input_data = np.array([input_text], dtype=object)
        
        # Tạo InferInput
        inputs = []
        inputs.append(httpclient.InferInput("INPUT_NAME", input_data.shape, "BYTES"))
        inputs[0].set_data_from_numpy(input_data)
        
        outputs = []
        outputs.append(httpclient.InferRequestedOutput("OUTPUT_NAME"))
        
        # Gọi inference
        response = triton_client.infer(
            model_name=model_name,
            inputs=inputs,
            outputs=outputs
        )
        
        output_data = response.as_numpy("OUTPUT_NAME")
        print(f"Output: {output_data}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)


def streaming_inference_example():
    try:
        triton_client = httpclient.InferenceServerClient(
            url="localhost:9001", 
            verbose=True
        )
        
        model_name = "your_model_name"
        input_text = "Hello world"
        
        # Chuẩn bị input
        input_data = np.array([input_text], dtype=object)
        inputs = []
        inputs.append(httpclient.InferInput("INPUT_NAME", input_data.shape, "BYTES"))
        inputs[0].set_data_from_numpy(input_data)
        
        outputs = []
        outputs.append(httpclient.InferRequestedOutput("OUTPUT_NAME"))
        
        # Stream inference
        triton_client.start_stream()
        
        triton_client.async_stream_infer(
            model_name=model_name,
            inputs=inputs,
            outputs=outputs
        )
        
        # Nhận kết quả streaming
        result = triton_client.get_async_stream_results()
        print(f"Stream result: {result}")
        
        triton_client.stop_stream()
        
    except Exception as e:
        print(f"Streaming error: {str(e)}")


# Gọi hàm streaming nếu cần
# streaming_inference_example()
