import os, sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

# 添加到 sys.path
sys.path.append(parent_dir)
import gc
import torch
import re
from typing import List
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from uuid import uuid4
import logging

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


faiss_vector_store = None


def lazy_load_faiss(embedding_model):
    global faiss_vector_store
    if faiss_vector_store is None:
        if faiss_vector_store is None:
            try:
                faiss_vector_store = FAISS.load_local(
                    "./faiss_index",
                    embedding_model,
                    allow_dangerous_deserialization=True,
                )
                logger.info("Faiss索引加载成功")
            except Exception as e:
                logger.error(f"Faiss索引失败: {str(e)}")
                faiss_vector_store = None
    return faiss_vector_store
