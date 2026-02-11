# BERT ONNX Model cho Triton Inference Server

Hướng dẫn deploy BERT model với ONNX backend trên Triton Inference Server.

## 🎯 Lợi ích của ONNX

So với Python backend, ONNX mang lại:
- ⚡ **Performance cao hơn**: ONNX Runtime được tối ưu hóa cho inference
- 💾 **Sử dụng ít tài nguyên hơn**: Không cần Python interpreter overhead
- 🔧 **Tính tương thích**: Có thể chạy trên nhiều framework khác nhau
- 📦 **Deployment đơn giản hơn**: Không cần quản lý Python dependencies

## 📁 Cấu trúc thư mục

```
model_repository/
├── bert_onnx/
│   ├── config.pbtxt           # Cấu hình Triton
│   ├── tokenizer/             # Tokenizer (tạo sau khi convert)
│   │   ├── config.json
│   │   ├── tokenizer.json
│   │   └── ...
│   └── 1/
│       └── model.onnx         # ONNX model (tạo sau khi convert)
```

## 🚀 Hướng dẫn sử dụng

### 1. Cài đặt dependencies

```bash
pip install -r requirements_onnx.txt
```

### 2. Convert BERT sang ONNX

```bash
# Convert BERT base model (mặc định)
python convert_bert_to_onnx.py

# Hoặc convert model khác
python convert_bert_to_onnx.py --model-name bert-base-multilingual-cased

# Với các tùy chọn khác
python convert_bert_to_onnx.py \
    --model-name vinai/phobert-base \
    --output-dir model_repository/bert_onnx/1 \
    --max-seq-length 256 \
    --opset-version 14
```

Các models phổ biến:
- `bert-base-uncased` - BERT tiếng Anh
- `bert-base-multilingual-cased` - BERT đa ngôn ngữ
- `vinai/phobert-base` - PhoBERT cho tiếng Việt
- `microsoft/deberta-v3-base` - DeBERTa

### 3. Chạy Triton Server

```bash
docker run --gpus all --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v ${PWD}/model_repository:/models \
  nvcr.io/nvidia/tritonserver:24.01-py3 \
  tritonserver --model-repository=/models
```

### 4. Test model

```bash
# Test với HTTP
python test_bert_onnx.py \
    --protocol http \
    --text "Hello world" "How are you?"

# Test với gRPC
python test_bert_onnx.py \
    --protocol grpc \
    --text "Hello world" "How are you?"

# Chạy benchmark
python test_bert_onnx.py \
    --protocol http \
    --benchmark \
    --benchmark-iterations 100 \
    --text "This is a test sentence"
```

## ⚙️ Cấu hình

### config.pbtxt

File cấu hình chính cho Triton Server:

```protobuf
name: "bert_onnx"
backend: "onnxruntime"
max_batch_size: 32

# Dynamic batching tự động gom các request lại
dynamic_batching {
  preferred_batch_size: [4, 8, 16]
  max_queue_delay_microseconds: 100
}

# Tối ưu hóa GPU
optimization {
  execution_accelerators {
    gpu_execution_accelerator : [ {
      name : "cuda"
    }]
  }
}
```

### Tùy chỉnh batch size

Để thay đổi batch size tối đa, sửa `max_batch_size` trong `config.pbtxt`:

```protobuf
max_batch_size: 64  # Tăng lên 64
```

### Tối ưu cho CPU

Nếu chạy trên CPU, sửa `instance_group`:

```protobuf
instance_group [
  {
    count: 1
    kind: KIND_CPU
  }
]
```

## 📊 So sánh Python vs ONNX Backend

| Tiêu chí | Python Backend | ONNX Backend |
|----------|----------------|--------------|
| Performance | Chậm hơn | ⚡ Nhanh hơn 2-3x |
| Memory | Cao hơn | 💾 Thấp hơn |
| Deployment | Phức tạp | 📦 Đơn giản |
| Flexibility | Cao (code Python) | Thấp (fixed graph) |
| Debugging | Dễ | Khó hơn |

## 🔧 Troubleshooting

### Lỗi "Model not ready"

Kiểm tra Triton server logs:
```bash
docker logs <container_id>
```

### Lỗi ONNX conversion

Thử giảm opset version:
```bash
python convert_bert_to_onnx.py --opset-version 12
```

### Out of memory

Giảm max_batch_size trong config.pbtxt hoặc max_seq_length:
```bash
python convert_bert_to_onnx.py --max-seq-length 128
```

## 📝 Input/Output

### Input
- `input_ids`: INT64, shape [batch_size, seq_length]
- `attention_mask`: INT64, shape [batch_size, seq_length]

### Output
- `last_hidden_state`: FP32, shape [batch_size, seq_length, 768]
- `pooler_output`: FP32, shape [batch_size, 768]

## 💡 Tips

1. **Dynamic batching**: Bật để tự động gom các requests lại, tăng throughput
2. **Preferred batch sizes**: Đặt theo workload của bạn
3. **Max queue delay**: Điều chỉnh trade-off giữa latency và throughput
4. **Model warmup**: Chạy vài requests đầu để warmup model

## 🔗 Tài liệu tham khảo

- [ONNX Runtime](https://onnxruntime.ai/)
- [Triton Inference Server](https://github.com/triton-inference-server/server)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers)

