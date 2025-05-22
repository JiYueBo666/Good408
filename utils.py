import os, sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
# 添加到 sys.path
sys.path.append(parent_dir)
import numpy as np
from typing import List
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from uuid import uuid4
from FileProcess.Embedding import get_embedding_model, get_rerank_model
import logging
from FileProcess.Process import BM25Manager


logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
logging.getLogger("openai").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.WARNING)
faiss_vector_store = None
bm25_store = None


import time


def timer(func):
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed_time = time.perf_counter() - start_time
        print(f"函数 {func.__name__} 执行耗时：{elapsed_time:.9f} 秒")
        return result

    return wrapper


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


def lazy_load_bm25():
    global bm25_store
    if bm25_store is None:
        if bm25_store is None:
            try:
                bm25_store = BM25Manager.load_index()
                logger.info("BM25索引加载成功")
            except Exception as e:
                logger.error(f"BM25索引失败: {str(e)}")
                bm25_store = None
    return bm25_store


@timer
def search_query(query: str):
    embedding_model = get_embedding_model()
    bm25_store = lazy_load_bm25()
    faiss_store = lazy_load_faiss(embedding_model)

    semantic_results_docs = []
    semantic_results_metadatas = []
    semantic_results_ids = []
    if faiss_store is not None:
        try:
            semantic_results = faiss_store.similarity_search_with_score(query, k=10)
            semantic_results_docs = [
                result[0].page_content for result in semantic_results
            ]
            semantic_results_metadatas = [
                result[0].metadata for result in semantic_results
            ]
            semantic_results_ids = [result[0].id for result in semantic_results]

        except Exception as e:
            logger.info(f"faiss search wrong: {e}")

    if bm25_store is not None:
        try:
            bm25_result = bm25_store.search(query, top_k=10)
        except Exception as e:
            logger.info(f"bm25 search wrong: {e}")

    prepared_semantic_results_for_hybrid = {
        "ids": [semantic_results_ids],
        "documents": [semantic_results_docs],
        "metadatas": [semantic_results_metadatas],
    }

    hybrid_results = hybrid_merge(prepared_semantic_results_for_hybrid, bm25_result)

    doc_ids_current = []
    docs_current = []

    if hybrid_results:
        for doc_id, result_data in hybrid_results[:10]:
            doc_ids_current.append(doc_id)
            docs_current.append(result_data["content"])

    if doc_ids_current:
        rerank_results = reranked_results(query, docs_current, doc_ids_current)

    return rerank_results


@timer
def reranked_results(query, docs, doc_ids, method=None, top_k=5):
    """
    对结果进行重排序

    参数：
        query:查询字符串
        docs:文档内容列表
        doc_ids:文档id列表
        method:重排序方法
    """

    # 暂时支持一种方法
    method = "cross_encoder"
    if method == "cross_encoder":
        return rerank_with_cross_encoder(query, docs, doc_ids, top_k)


def rerank_with_cross_encoder(query, docs, doc_ids, top_k=5):
    if not docs:
        return []
    encoder = get_rerank_model()
    if encoder is None:
        logger.warning("rerank encoder is None, skip rerank step")
        return [
            (doc_id, {"content": doc, "score": 1.0 - idx / len(docs)})
            for idx, (doc_id, doc) in enumerate(zip(doc_ids, docs))
        ]

    cross_input = [[query, doc] for doc in docs]

    try:
        scores = encoder.predict(cross_input)
        results = [
            (
                doc_id,
                {
                    "content": doc,
                    "score": float(score),
                },
            )
            for doc_id, doc, score in zip(doc_ids, docs, scores)
        ]

        results = sorted(results, key=lambda x: x[1]["score"], reverse=True)

        logger.info(f"rerank results 过程结束")
        return results[:top_k]
    except Exception as e:
        logger.error(f"rerank with cross_encoder failed: {e}")
        return [
            (doc_id, {"content": doc, "score": 1.0 - idx / len(docs)})
            for idx, (doc_id, doc) in enumerate(zip(doc_ids, docs))
        ]


@timer
def hybrid_merge(semantic_results, bm25_results, alpha=0.7):
    """
    合并语义搜索和BM25检索结果
    """
    merged_dict = {}

    if (
        semantic_results
        and isinstance(semantic_results.get("documents"), list)
        and len(semantic_results["documents"]) > 0
        and isinstance(semantic_results.get("metadatas"), list)
        and len(semantic_results["metadatas"]) > 0
        and isinstance(semantic_results.get("ids"), list)
        and len(semantic_results["ids"]) > 0
        and isinstance(semantic_results["documents"][0], list)
        and isinstance(semantic_results["metadatas"][0], list)
        and isinstance(semantic_results["ids"][0], list)
        and len(semantic_results["documents"][0])
        == len(semantic_results["metadatas"][0])
        == len(semantic_results["ids"][0])
    ):

        num_results = len(semantic_results["documents"][0])
        # Assuming semantic_results are already ordered by relevance (higher is better)
        # A simple rank-based score, can be replaced if actual scores/distances are available and preferred
        for i, (doc_id, doc, meta) in enumerate(
            zip(
                semantic_results["ids"][0],
                semantic_results["documents"][0],
                semantic_results["metadatas"][0],
            )
        ):
            score = 1.0 - (
                i / max(1, num_results)
            )  # Higher rank (smaller i) gets higher score
            merged_dict[doc_id] = {
                "score": alpha * score,
                "content": doc,
            }
    # 处理bm25
    if not bm25_results:
        return sorted(merged_dict.items(), key=lambda x: x[1]["score"], reverse=True)

    valid_bm25_score = [
        r["score"] for r in bm25_results if isinstance(r, dict) and "score" in r
    ]

    max_bm25_score = max(valid_bm25_score) if valid_bm25_score else 1.0

    for result in bm25_results:
        if not (
            isinstance(result, dict)
            and "id" in result
            and "score" in result
            and "content" in result
        ):
            logging.warning(f"Skipping invalid BM25 result item: {result}")
            continue
        doc_id = result["id"]
        normalized_score = result["score"] / max_bm25_score if max_bm25_score > 0 else 0

        if doc_id in merged_dict:
            merged_dict[doc_id]["score"] += (1 - alpha) * normalized_score
        else:
            merged_dict[doc_id] = {
                "score": (1 - alpha) * normalized_score,
                "content": result["content"],
            }
    merged_results = sorted(
        merged_dict.items(), key=lambda x: x[1]["score"], reverse=True
    )

    return merged_results
