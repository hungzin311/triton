import json
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    """
    BERT Model for Triton Inference Server (Python Backend)
    
    Outputs:
        - embeddings: Last hidden state (batch_size, seq_len, hidden_size)
        - pooler_output: Pooled output for classification (batch_size, hidden_size)
    """

    def initialize(self, args):
        """
        Args:
            args: Dictionary containing model configuration
        """
        self.model_config = model_config = json.loads(args["model_config"])
        
        # Get model parameters from config.pbtxt
        model_name = "bert-base-uncased"
        max_length = 512
        
        for key, value in model_config.get("parameters", {}).items():
            if key == "model_name":
                model_name = value["string_value"]
            elif key == "max_length":
                max_length = int(value["string_value"])
        
        self.max_length = max_length
        
        # Get device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load tokenizer and model
        self.logger = pb_utils.Logger
        self.logger.log_info(f"Loading BERT model: {model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        self.logger.log_info(f"BERT model loaded successfully on {self.device}")

    def execute(self, requests):
        """
        Process inference requests.
        
        Args:
            requests: List of pb_utils.InferenceRequest
            
        Returns:
            List of pb_utils.InferenceResponse
        """
        responses = []
        
        for request in requests:
            try:
                # Get input tensor
                text_input = pb_utils.get_input_tensor_by_name(request, "text_input")
                
                # Decode text inputs
                texts = []
                for text_bytes in text_input.as_numpy():
                    if isinstance(text_bytes, bytes):
                        texts.append(text_bytes.decode("utf-8"))
                    else:
                        texts.append(str(text_bytes))
                
                # Tokenize
                encoded = self.tokenizer(
                    texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt"
                )
                
                # Move to device
                input_ids = encoded["input_ids"].to(self.device)
                attention_mask = encoded["attention_mask"].to(self.device)
                
                # Forward pass
                with torch.no_grad():
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )
                
                # Get embeddings
                last_hidden_state = outputs.last_hidden_state.cpu().numpy()
                pooler_output = outputs.pooler_output.cpu().numpy()
                
                # Create output tensors
                embeddings_tensor = pb_utils.Tensor(
                    "embeddings",
                    last_hidden_state.astype(np.float32)
                )
                pooler_tensor = pb_utils.Tensor(
                    "pooler_output", 
                    pooler_output.astype(np.float32)
                )
                
                # Create response
                response = pb_utils.InferenceResponse(
                    output_tensors=[embeddings_tensor, pooler_tensor]
                )
                responses.append(response)
                
            except Exception as e:
                self.logger.log_error(f"Error processing request: {str(e)}")
                error = pb_utils.TritonError(f"Error: {str(e)}")
                responses.append(pb_utils.InferenceResponse(error=error))
        
        return responses

    def finalize(self):
        self.logger.log_info("Cleaning up BERT model...")
        del self.model
        del self.tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.logger.log_info("BERT model cleaned up successfully")

