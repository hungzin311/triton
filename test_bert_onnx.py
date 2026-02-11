"""
Test client for BERT ONNX model on Triton Inference Server
Supports both HTTP and gRPC protocols
"""

import argparse
import numpy as np
import sys
from transformers import AutoTokenizer
from pathlib import Path


def load_tokenizer(tokenizer_path: str = "model_repository/bert_onnx/tokenizer"):
    """Load tokenizer for preprocessing"""
    if Path(tokenizer_path).exists():
        return AutoTokenizer.from_pretrained(tokenizer_path)
    else:
        print(f"⚠️  Tokenizer not found at {tokenizer_path}, using bert-base-uncased")
        return AutoTokenizer.from_pretrained("bert-base-uncased")


# HTTP Client
def test_http(url: str, texts: list, tokenizer, max_length: int = 512):
    """Test BERT ONNX model using HTTP protocol"""
    import tritonclient.http as httpclient
    
    try:
        client = httpclient.InferenceServerClient(url=url, verbose=False)
        
        if not client.is_server_live():
            print("❌ Server is not live!")
            return None
        
        # Check if model is ready
        if not client.is_model_ready("bert_onnx"):
            print("❌ BERT ONNX model is not ready!")
            return None
        
        print(f"📝 Processing {len(texts)} text(s)...")
        
        # Tokenize texts
        encoded = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="np"
        )
        
        input_ids = encoded["input_ids"].astype(np.int64)
        attention_mask = encoded["attention_mask"].astype(np.int64)
        
        print(f"   Input IDs shape: {input_ids.shape}")
        print(f"   Attention mask shape: {attention_mask.shape}")
        
        # Prepare inputs
        inputs = [
            httpclient.InferInput("input_ids", input_ids.shape, "INT64"),
            httpclient.InferInput("attention_mask", attention_mask.shape, "INT64")
        ]
        inputs[0].set_data_from_numpy(input_ids)
        inputs[1].set_data_from_numpy(attention_mask)
        
        # Prepare outputs
        outputs = [
            httpclient.InferRequestedOutput("last_hidden_state"),
            httpclient.InferRequestedOutput("pooler_output")
        ]
        
        # Inference
        print("🚀 Running inference...")
        response = client.infer(
            model_name="bert_onnx",
            inputs=inputs,
            outputs=outputs
        )
        
        last_hidden_state = response.as_numpy("last_hidden_state")
        pooler_output = response.as_numpy("pooler_output")
        
        print(f"\n✅ Inference successful!")
        print(f"   Last hidden state shape: {last_hidden_state.shape}")
        print(f"   Pooler output shape: {pooler_output.shape}")
        
        return last_hidden_state, pooler_output
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None


# gRPC Client  
def test_grpc(url: str, texts: list, tokenizer, max_length: int = 512):
    """Test BERT ONNX model using gRPC protocol"""
    import tritonclient.grpc as grpcclient
    
    try:
        client = grpcclient.InferenceServerClient(url=url, verbose=False)
        
        if not client.is_server_live():
            print("❌ Server is not live!")
            return None
            
        if not client.is_model_ready("bert_onnx"):
            print("❌ BERT ONNX model is not ready!")
            return None
        
        print(f"📝 Processing {len(texts)} text(s) via gRPC...")
        
        # Tokenize texts
        encoded = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="np"
        )
        
        input_ids = encoded["input_ids"].astype(np.int64)
        attention_mask = encoded["attention_mask"].astype(np.int64)
        
        print(f"   Input IDs shape: {input_ids.shape}")
        print(f"   Attention mask shape: {attention_mask.shape}")
        
        # Prepare inputs
        inputs = [
            grpcclient.InferInput("input_ids", input_ids.shape, "INT64"),
            grpcclient.InferInput("attention_mask", attention_mask.shape, "INT64")
        ]
        inputs[0].set_data_from_numpy(input_ids)
        inputs[1].set_data_from_numpy(attention_mask)
        
        # Prepare outputs
        outputs = [
            grpcclient.InferRequestedOutput("last_hidden_state"),
            grpcclient.InferRequestedOutput("pooler_output")
        ]
        
        # Inference
        print("🚀 Running inference...")
        response = client.infer(
            model_name="bert_onnx",
            inputs=inputs,
            outputs=outputs
        )
        
        last_hidden_state = response.as_numpy("last_hidden_state")
        pooler_output = response.as_numpy("pooler_output")
        
        print(f"\n✅ Inference successful!")
        print(f"   Last hidden state shape: {last_hidden_state.shape}")
        print(f"   Pooler output shape: {pooler_output.shape}")
        
        return last_hidden_state, pooler_output
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def compute_similarity(vec1, vec2):
    """Compute cosine similarity between two vectors"""
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def benchmark(url: str, texts: list, tokenizer, protocol: str = "http", iterations: int = 10):
    """Benchmark inference performance"""
    import time
    
    print(f"\n{'='*60}")
    print(f"🔥 Performance Benchmark ({iterations} iterations)")
    print(f"{'='*60}")
    
    latencies = []
    
    for i in range(iterations):
        start_time = time.time()
        
        if protocol == "http":
            result = test_http(url, texts, tokenizer)
        else:
            result = test_grpc(url, texts, tokenizer)
        
        if result is None:
            print(f"❌ Benchmark failed at iteration {i+1}")
            return
        
        latency = (time.time() - start_time) * 1000  # Convert to ms
        latencies.append(latency)
        
        if (i + 1) % 10 == 0:
            print(f"   Completed {i+1}/{iterations} iterations")
    
    latencies = np.array(latencies)
    
    print(f"\n📊 Benchmark Results:")
    print(f"   Mean latency: {latencies.mean():.2f} ms")
    print(f"   Median latency: {np.median(latencies):.2f} ms")
    print(f"   Min latency: {latencies.min():.2f} ms")
    print(f"   Max latency: {latencies.max():.2f} ms")
    print(f"   Std deviation: {latencies.std():.2f} ms")
    print(f"   Throughput: {1000 / latencies.mean():.2f} requests/sec")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test BERT ONNX model on Triton")
    parser.add_argument("--protocol", type=str, default="http", choices=["http", "grpc"],
                        help="Protocol to use (http or grpc)")
    parser.add_argument("--http-url", type=str, default="localhost:8000",
                        help="HTTP endpoint URL")
    parser.add_argument("--grpc-url", type=str, default="localhost:8001", 
                        help="gRPC endpoint URL")
    parser.add_argument("--text", type=str, nargs="+", 
                        default=["Hello, how are you?", "I am fine, thank you!"],
                        help="Text(s) to process")
    parser.add_argument("--tokenizer-path", type=str, 
                        default="model_repository/bert_onnx/tokenizer",
                        help="Path to tokenizer directory")
    parser.add_argument("--max-length", type=int, default=512,
                        help="Maximum sequence length")
    parser.add_argument("--benchmark", action="store_true",
                        help="Run performance benchmark")
    parser.add_argument("--benchmark-iterations", type=int, default=100,
                        help="Number of benchmark iterations")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🤖 BERT ONNX Model Test Client")
    print("=" * 60)
    print(f"Protocol: {args.protocol.upper()}")
    print(f"Texts: {args.text}")
    print("=" * 60)
    
    # Load tokenizer
    print("\n📥 Loading tokenizer...")
    tokenizer = load_tokenizer(args.tokenizer_path)
    print("✅ Tokenizer loaded!")
    
    url = args.http_url if args.protocol == "http" else args.grpc_url
    
    if args.benchmark:
        benchmark(url, args.text, tokenizer, args.protocol, args.benchmark_iterations)
    else:
        if args.protocol == "http":
            result = test_http(url, args.text, tokenizer, args.max_length)
        else:
            result = test_grpc(url, args.text, tokenizer, args.max_length)
        
        if result and len(args.text) >= 2:
            _, pooler_output = result
            print("\n" + "=" * 60)
            print("🔍 Similarity Analysis")
            print("=" * 60)
            
            # Compute similarity between first two texts
            sim = compute_similarity(pooler_output[0], pooler_output[1])
            print(f"Text 1: {args.text[0]}")
            print(f"Text 2: {args.text[1]}")
            print(f"Cosine similarity: {sim:.4f}")

