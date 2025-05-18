import os
from pathlib import Path
import torch
import threading

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from huggingface_hub import snapshot_download
import logging
from transformers import AutoTokenizer, AutoModel

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

parent_dir = Path.cwd().parent


try:
    if os.path.exists("../huggingfaceModel"):
        logger.info("embedding model already exist")
        pass
    else:
        local_dir = "../huggingfaceModel"
        logger.info(f"ready to download model to path:{local_dir}")
        snapshot_download(
            repo_id="DMetaSoul/Dmeta-embedding-zh",
            local_dir=local_dir,
            proxies={"https": "http://localhost:7890"},
            max_workers=8,
        )

        logger.info(f"Model is downloading to {local_dir}")
except Exception as e:
    logging.error(e)


_embedding_model = None
_embedding_model_lock = threading.Lock()


class EmbeddingModel:
    def __init__(self, default_path="../huggingfaceModel"):
        self.default_path = default_path
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(default_path)
            self.model = AutoModel.from_pretrained(default_path)
        except Exception as e:
            logger.error(f"加载模型失败: {e}")
            raise e  # 或者根据需求设为None等处理方式

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    def cls_pooling(self, model_output):
        return model_output[0][:, 0]

    def inference(self, text):
        try:
            self.model.eval()
            with torch.no_grad():
                inputs = self.tokenizer(
                    text, padding=True, truncation=True, return_tensors="pt"
                )
                model_output = self.model(**inputs)
                embeds = self.cls_pooling(model_output)
                embeds = torch.nn.functional.normalize(embeds, p=2, dim=1).numpy()
                return embeds
        except Exception as e:
            logger.error(f"推理失败: {e}")
            return None


def get_embedding_model():
    """延迟加载嵌入模型"""
    global _embedding_model
    if _embedding_model is None:
        with _embedding_model_lock:
            if _embedding_model is None:
                try:
                    _embedding_model = EmbeddingModel(
                        default_path="../huggingfaceModel"
                    )
                    logger.info("Embedding模型加载成功")
                except Exception as e:
                    logger.error(f"加载Embedding模型失败: {str(e)}")
                    _embedding_model = None
    return _embedding_model
