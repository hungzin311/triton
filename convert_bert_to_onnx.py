"""
Script to convert BERT model to ONNX format for Triton Inference Server
"""

import torch
import argparse
from transformers import AutoTokenizer, AutoModel
from pathlib import Path
import onnx
import onnxruntime as ort


def convert_bert_to_onnx(
    model_name: str = "bert-base-uncased",
    output_dir: str = "model_repository/bert_onnx/1",
    max_seq_length: int = 512,
    opset_version: int = 14
):
    """
    Convert BERT model to ONNX format
    
    Args:
        model_name: HuggingFace model name
        output_dir: Directory to save ONNX model
        max_seq_length: Maximum sequence length
        opset_version: ONNX opset version
    """
    
    print(f"🔄 Converting {model_name} to ONNX format...")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load model and tokenizer
    print("📥 Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    
    # Create dummy input
    print("🔧 Creating dummy inputs...")
    dummy_text = "This is a sample text for ONNX conversion"
    inputs = tokenizer(
        dummy_text,
        return_tensors="pt",
        max_length=max_seq_length,
        padding="max_length",
        truncation=True
    )
    
    # Define input/output names
    input_names = ["input_ids", "attention_mask"]
    output_names = ["last_hidden_state", "pooler_output"]
    
    # Dynamic axes for variable batch size and sequence length
    dynamic_axes = {
        "input_ids": {0: "batch_size", 1: "sequence_length"},
        "attention_mask": {0: "batch_size", 1: "sequence_length"},
        "last_hidden_state": {0: "batch_size", 1: "sequence_length"},
        "pooler_output": {0: "batch_size"}
    }
    
    # Export to ONNX
    onnx_path = output_path / "model.onnx"
    print(f"💾 Exporting to {onnx_path}...")
    
    torch.onnx.export(
        model,
        args=(inputs["input_ids"], inputs["attention_mask"]),
        f=str(onnx_path),
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=opset_version,
        do_constant_folding=True,
        export_params=True,
    )
    
    print("✅ ONNX model exported successfully!")
    
    # Verify the model
    print("🔍 Verifying ONNX model...")
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    print("✅ ONNX model is valid!")
    
    # Test with ONNX Runtime
    print("🧪 Testing with ONNX Runtime...")
    ort_session = ort.InferenceSession(str(onnx_path))
    
    # Run inference
    ort_inputs = {
        "input_ids": inputs["input_ids"].numpy(),
        "attention_mask": inputs["attention_mask"].numpy()
    }
    ort_outputs = ort_session.run(None, ort_inputs)
    
    print(f"   Last hidden state shape: {ort_outputs[0].shape}")
    print(f"   Pooler output shape: {ort_outputs[1].shape}")
    
    # Save tokenizer
    tokenizer_path = output_path.parent / "tokenizer"
    tokenizer_path.mkdir(exist_ok=True)
    tokenizer.save_pretrained(str(tokenizer_path))
    print(f"💾 Tokenizer saved to {tokenizer_path}")
    
    # Print model info
    print("\n" + "="*60)
    print("📊 Model Information")
    print("="*60)
    print(f"Model: {model_name}")
    print(f"ONNX file: {onnx_path}")
    print(f"ONNX opset version: {opset_version}")
    print(f"Max sequence length: {max_seq_length}")
    print(f"Hidden size: {ort_outputs[1].shape[-1]}")
    print("="*60)
    
    return str(onnx_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert BERT to ONNX")
    parser.add_argument(
        "--model-name",
        type=str,
        default="bert-base-uncased",
        help="HuggingFace model name (default: bert-base-uncased)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="model_repository/bert_onnx/1",
        help="Output directory for ONNX model"
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=512,
        help="Maximum sequence length"
    )
    parser.add_argument(
        "--opset-version",
        type=int,
        default=14,
        help="ONNX opset version"
    )
    
    args = parser.parse_args()
    
    convert_bert_to_onnx(
        model_name=args.model_name,
        output_dir=args.output_dir,
        max_seq_length=args.max_seq_length,
        opset_version=args.opset_version
    )
    
    print("\n✨ Conversion complete! You can now use the model with Triton Inference Server.")
    print("   Run: docker run --gpus all --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 \\")
    print("        -v ${PWD}/model_repository:/models \\")
    print("        nvcr.io/nvidia/tritonserver:24.01-py3 \\")
    print("        tritonserver --model-repository=/models")

