import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import os

class CLIPZeroShot:
    def __init__(self, model_name_or_path, device=None):
        """
        Args:
            model_name_or_path: 本地路径或 HuggingFace 模型名
            device: 'cuda' or 'cpu'
        """
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"Loading CLIP model from {model_name_or_path} to {self.device}...")
        self.model = CLIPModel.from_pretrained(model_name_or_path).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name_or_path)
        self.model.eval()

    def predict(self, images, candidate_labels):
        """
        Args:
            images: List of PIL Images or single PIL Image
            candidate_labels: List of strings (e.g., ["positive", "neutral", "negative"])
        
        Returns:
            preds: List of predicted indices (0, 1, 2...) matching candidate_labels order
            probs: Tensor of shape (batch, num_labels)
        """
        if not isinstance(images, list):
            images = [images]
            
        # 构造 Prompts
        # 简单的 "positive", "negative" 可能有歧义，加上 sentiment 上下文
        text_prompts = [f"a photo of a {label} sentiment" for label in candidate_labels]
        
        inputs = self.processor(
            text=text_prompts,
            images=images, 
            return_tensors="pt", 
            padding=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            # logits_per_image: (batch_size, num_labels)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)
            preds = probs.argmax(dim=1)
            
        return preds.cpu().numpy(), probs.cpu()
