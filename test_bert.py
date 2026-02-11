"""
Test client for BERT model on Triton Inference Server
Supports both HTTP and gRPC protocols
"""

import argparse
import numpy as np
import sys

# HTTP Client
def test_http(url: str, texts: list):
    """Test BERT model using HTTP protocol"""
    import tritonclient.http as httpclient
    
    try:
        client = httpclient.InferenceServerClient(url=url, verbose=False)
        
        if not client.is_server_live():
            print("Server is not live!")
            return None
        
        # Check if model is ready
        if not client.is_model_ready("bert_model"):
            print("BERT model is not ready!")
            return None
        
        print(f"Sending {len(texts)} text(s) for inference...")
        
        # Prepare input
        text_data = np.array(texts, dtype=object)
        
        inputs = [
            httpclient.InferInput("text_input", text_data.shape, "BYTES")
        ]
        inputs[0].set_data_from_numpy(text_data)
        
        outputs = [
            httpclient.InferRequestedOutput("embeddings"),
            httpclient.InferRequestedOutput("pooler_output")
        ]
        
        # Inference
        response = client.infer(
            model_name="bert_model",
            inputs=inputs,
            outputs=outputs
        )
        
        embeddings = response.as_numpy("embeddings")
        pooler_output = response.as_numpy("pooler_output")
        
        print(f"\n✅ Inference successful!")
        print(f"   Embeddings shape: {embeddings.shape}")
        print(f"   Pooler output shape: {pooler_output.shape}")
        
        return embeddings, pooler_output
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None


# gRPC Client  
def test_grpc(url: str, texts: list):
    """Test BERT model using gRPC protocol"""
    import tritonclient.grpc as grpcclient
    
    try:
        client = grpcclient.InferenceServerClient(url=url, verbose=False)
        
        if not client.is_server_live():
            print("Server is not live!")
            return None
            
        if not client.is_model_ready("bert_model"):
            print("BERT model is not ready!")
            return None
        
        print(f"Sending {len(texts)} text(s) for inference via gRPC...")
        
        # Prepare input
        text_data = np.array(texts, dtype=object)
        
        inputs = [
            grpcclient.InferInput("text_input", text_data.shape, "BYTES")
        ]
        inputs[0].set_data_from_numpy(text_data)
        
        outputs = [
            grpcclient.InferRequestedOutput("embeddings"),
            grpcclient.InferRequestedOutput("pooler_output")
        ]
        
        # Inference
        response = client.infer(
            model_name="bert_model",
            inputs=inputs,
            outputs=outputs
        )
        
        embeddings = response.as_numpy("embeddings")
        pooler_output = response.as_numpy("pooler_output")
        
        print(f"\n✅ Inference successful!")
        print(f"   Embeddings shape: {embeddings.shape}")
        print(f"   Pooler output shape: {pooler_output.shape}")
        
        return embeddings, pooler_output
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None


def compute_similarity(embeddings1, embeddings2):
    """Compute cosine similarity between two embeddings"""
    # Use pooler output (CLS token)
    vec1 = embeddings1.flatten()
    vec2 = embeddings2.flatten()
    
    similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    return similarity


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test BERT model on Triton")
    parser.add_argument("--protocol", type=str, default="http", choices=["http", "grpc"],
                        help="Protocol to use (http or grpc)")
    parser.add_argument("--http-url", type=str, default="localhost:8000",
                        help="HTTP endpoint URL")
    parser.add_argument("--grpc-url", type=str, default="localhost:8001", 
                        help="gRPC endpoint URL")
    parser.add_argument("--text", type=str, nargs="+", 
                        default=["Hello, how are you?", "I am fine, thank you!"],
                        help="Text(s) to process")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("BERT Model Test Client")
    print("=" * 60)
    print(f"Protocol: {args.protocol.upper()}")
    print(f"Texts: {args.text}")
    print("=" * 60)
    
    if args.protocol == "http":
        result = test_http(args.http_url, args.text)
    else:
        result = test_grpc(args.grpc_url, args.text)
    
    if result and len(args.text) >= 2:
        embeddings, pooler_output = result
        print("\n" + "=" * 60)
        print("Similarity Analysis")
        print("=" * 60)
        
        # Compute similarity between first two texts
        sim = compute_similarity(pooler_output[0], pooler_output[1])
        print(f"Cosine similarity between text 1 and text 2: {sim:.4f}")

