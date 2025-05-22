import os
import gc
import pickle
import numpy as np
import torch
import re
from typing import List
from pathlib import Path
from langchain.schema import Document
from magic_pdf.data.data_reader_writer import FileBasedDataWriter, FileBasedDataReader
from magic_pdf.data.dataset import PymuDocDataset
from magic_pdf.model.doc_analyze_by_custom_model import doc_analyze
from magic_pdf.config.enums import SupportedPdfParseMethod
from langchain_community.document_loaders import UnstructuredMarkdownLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from uuid import uuid4
from FileProcess.Embedding import get_embedding_model
import logging
from rank_bm25 import BM25Okapi
import jieba

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def process_single_pdf(
    pdf_file_name, local_image_dir="output/md", local_md_dir="output"
):
    try:
        # 获取文件名，不包含后缀
        name_without_suff = pdf_file_name.split(".")[0]
        print(f"Processing: {name_without_suff}")

        # 创建输出目录
        os.makedirs(local_image_dir, exist_ok=True)

        image_writer = FileBasedDataWriter(local_image_dir)
        md_writer = FileBasedDataWriter(local_md_dir)

        # 读取PDF内容
        reader = FileBasedDataReader("")
        pdf_bytes = reader.read(pdf_file_name)
        print(f"PDF size: {len(pdf_bytes)} bytes")

        # 创建数据集实例并进行分类
        ds = PymuDocDataset(pdf_bytes)

        # 清理不需要的变量
        del pdf_bytes
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        pipe_result = None
        try:
            if ds.classify() == SupportedPdfParseMethod.OCR:
                # 进行OCR处理
                infer_result = ds.apply(doc_analyze, ocr=True)
                pipe_result = infer_result.pipe_ocr_mode(image_writer)

            file_name = pdf_file_name.split("\\")[-1][:-4]

            if pipe_result:
                # 导出处理结果
                pipe_result.dump_md(md_writer, f"{file_name}.md", local_md_dir)
        except Exception as e:
            print(f"Error processing PDF content: {str(e)}")
            return False

        # 清理内存
        del ds
        del pipe_result
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return True

    except Exception as e:
        print(f"Error processing file {pdf_file_name}: {str(e)}")
        return False


def split_big2small(pdf_path, output_path="./MetaData/SmallData"):
    """
    pdf太大会爆存，切分成小块多次处理
    """
    import fitz

    doc = fitz.open(pdf_path)

    # 获取目录
    toc = doc.get_toc()
    print("目录：")
    for level, title, page in toc:
        print(f"{'-' * (level-1)} {title} starts at page {page}")

    # 遍历目录，按照顶级目录处理PDF
    prev_page = 0
    part_number = 1
    for i, (_, _, page) in enumerate(toc):
        if i == len(doc) - 1:  # 如果是最后一个项目，则切到最后一页
            end_page = doc.page_count
        else:
            end_page = toc[i + i][2] - 1

        new_doc = fitz.open()
        new_doc.insert_pdf(doc, from_page=prev_page - 1, to_page=end_page - 1)

        name = pdf_path.split("\\")[-1][:-4]

        output_filename = f"{output_path}/{name}{part_number}.pdf"
        if not os.path.exists(output_filename):
            os.makedirs(output_filename)
        new_doc.save(output_filename)
        new_doc.close()

        print(f"output to folder : {output_filename}")


def main():
    from pathlib import Path

    # 获取当前文件所在目录的父目录
    current_file_path = Path(__file__)
    parent_dir = current_file_path.parent.parent

    # 构造MetaData/SmallData路径
    data_folder = parent_dir / "MetaData" / "SmallData"
    # 确保目录存在
    if data_folder.exists() and data_folder.is_dir():
        # 获取文件夹中所有文件的路径，并添加到wait_list列表中
        wait_list = [
            str(file_path) for file_path in data_folder.iterdir() if file_path.is_file()
        ]
    else:
        print(f"The directory {data_folder} does not exist or is not a directory.")
        wait_list = []

    print(f"Files to process: {wait_list}")
    for pdf_file in wait_list:
        # split_big2small(pdf_file)

        if "参考文献" in pdf_file:
            continue

        print(f"\n{'='*50}")

        print(f"Starting to process: {pdf_file}")

        success = process_single_pdf(pdf_file)

        if not success:
            print(f"Failed to process: {pdf_file}")

        # 每个文件处理完后清理内存
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class FileProcessManager:
    def __init__(self) -> None:
        self.filter_words = ["竞成", "习题精编", "真题演练", "答案与解析"]
        self.recursive_splitter = RecursiveCharacterTextSplitter(
            separators=[
                "\n## ",
                "\n# ",
                "\n### ",
                "\n#### ",
                "# ",
                "## ",
                "### ",
                "#### ",
                "\n\n",
                "\n",
                " ",
                "",
            ],
            chunk_size=500,
            chunk_overlap=50,
        )

    def remove_filter_words(self, content):
        pattern = "|".join(map(re.escape, self.filter_words))
        return re.sub(pattern, "", content)

    def pipeline(self, path):
        # loader = UnstructuredMarkdownLoader(path, autodetect_encoding=True)
        loader = TextLoader(
            file_path=path,
            # mode="elements",
            # autodetect_encoding=True,
            encoding="utf-8",
        )
        data = loader.load()
        logger.info(f"file load done, docs count : {len(data)}")
        for i, _ in enumerate(data):
            data[i].page_content = self.remove_practice(data[i].page_content)
            data[i].page_content = self.remove_preface(data[i].page_content)
            data[i].page_content = self.remove_filter_words(data[i].page_content)

        split_docs = self.split_docs(data)

        return split_docs

    def remove_preface(self, content):
        # 去除前言
        pattern = r"前言.*?目录"
        # 将匹配到的部分替换为空字符串，从而移除这部分内容
        result, _ = re.subn(pattern, "", content, flags=re.DOTALL)
        return result

    def remove_practice(self, text):
        pattern = re.compile(
            r"(# \d+(?:\.\d+)+习题精编.*?)(?=\n# \d+(?:\.\d+)+(?!.*(?:真题演练|答案与解析)).*$)",
            re.DOTALL | re.MULTILINE,
        )

        match = pattern.search(text)
        if match:
            text = text.replace(match.group(0).strip(), "")
        return text

    def split_docs(self, doc: List[Document]):
        try:
            split_docs = self.recursive_splitter.split_documents(doc)
            # for d in split_docs:
            #     print(f"分割后：{d}")
        except Exception as e:
            logger.error(f"递归切分错误:{e}")
        return split_docs


class BM25Manager:
    def __init__(self):
        self.bm25_index = None
        self.doc_mapping = {}  # 映射BM25索引到文档ID
        self.tokenizer_corpus = []
        self.raw_corpus = []

    def build_index(self, documents: List[Document], doc_ids):
        """
        构建BM25索引
        """

        self.raw_corpus = documents
        self.doc_mapping = {i: doc_id for i, doc_id in enumerate(doc_ids)}

        # 分词
        for doc in documents:
            tokens = list(jieba.cut(doc.page_content))
            self.tokenizer_corpus.append(tokens)
        self.bm25_index = BM25Okapi(self.tokenizer_corpus)
        return True

    def search(self, query, top_k=5):
        if not self.bm25_index:
            return []

        tokenized_query = list(jieba.cut(query))
        bm25_scores = self.bm25_index.get_scores(tokenized_query)

        top_indices = np.argsort(bm25_scores)[-top_k:][::-1]

        results = []
        for idx in top_indices:
            if bm25_scores[idx] > 0:  # 只返回有相关性的结果
                results.append(
                    {
                        "id": self.doc_mapping[idx],
                        "score": float(bm25_scores[idx]),
                        "content": self.raw_corpus[idx].page_content,
                    }
                )

        return results

    def save_index(self, file_path: str = "./bm25_index"):
        """
        将BM25索引及关联数据保存到指定路径。
        :param file_path: 文件保存路径。
        """
        data_to_save = {
            "bm25_index": self.bm25_index,
            "doc_mapping": self.doc_mapping,
            "tokenizer_corpus": self.tokenizer_corpus,
            "raw_corpus": self.raw_corpus,
        }

        with open(file_path, "wb") as f:
            pickle.dump(data_to_save, f)

        print(f"索引已保存到 {file_path}")

    def clear(self):
        """清空索引"""
        self.bm25_index = None
        self.doc_mapping = {}
        self.tokenized_corpus = []
        self.raw_corpus = []

    @staticmethod
    def load_index(file_path: str = "./bm25_index"):
        """
        从指定路径加载BM25索引及相关数据。
        :param file_path: 文件路径。
        :return: 返回一个包含所有必要数据的BM25Manager实例。
        """
        with open(file_path, "rb") as f:
            data_loaded = pickle.load(f)

        bm25_manager = BM25Manager()
        bm25_manager.bm25_index = data_loaded["bm25_index"]
        bm25_manager.doc_mapping = data_loaded["doc_mapping"]
        bm25_manager.tokenizer_corpus = data_loaded["tokenizer_corpus"]
        bm25_manager.raw_corpus = data_loaded["raw_corpus"]

        print(f"索引已从 {file_path} 加载")
        return bm25_manager


faiss_contents_map, faiss_id_order_for_index = {}, []


if __name__ == "__main__":

    # 创建path_list存储所有markdown文件路径
    # path_list = []

    # # 获取当前文件所在目录的父目录
    # current_file_path = Path(__file__)
    # parent_dir = current_file_path.parent.parent

    # # 构造output路径
    # output_folder = parent_dir / "output"

    # # 确保目录存在
    # if output_folder.exists() and output_folder.is_dir():
    #     # 获取文件夹中所有markdown文件的路径，并添加到path_list列表中
    #     path_list = [
    #         str(file_path)
    #         for file_path in output_folder.iterdir()
    #         if file_path.is_file() and file_path.suffix.lower() == ".md"
    #     ]
    # else:
    #     print(f"输出目录 {output_folder} 不存在或不是一个目录。")

    # # 导入tqdm库
    # from tqdm import tqdm

    # manager = FileProcessManager()
    # docs = []

    # print(f"找到{len(path_list)}个Markdown文件，开始处理...")

    # # 使用tqdm包装path_list，显示处理进度
    # for path in tqdm(path_list, desc="处理Markdown文件", unit="文件"):
    #     doc_current = manager.pipeline(path)
    #     docs.extend(doc_current)

    # print(f"共处理了{len(docs)}个文档片段")

    embedding_model = get_embedding_model()

    # index = faiss.IndexFlatL2(len(embedding_model.embed_query("测试")))

    # vector_store = FAISS(
    #     embedding_function=embedding_model,
    #     index=index,
    #     docstore=InMemoryDocstore(),
    #     index_to_docstore_id={},
    # )
    # uuids = [str(uuid4()) for _ in range(len(docs))]
    # try:
    #     faiss_id_order_for_index = uuids
    #     faiss_contents_map = {uuids[i]: document for i, document in enumerate(docs)}
    #     documents = [
    #         faiss_contents_map.get(uuid, None) for uuid in faiss_id_order_for_index
    #     ]

    #     valid_docs_with_ids = [
    #         (doc_id, doc) for doc_id, doc in zip(faiss_id_order_for_index, documents)
    #     ]
    #     final_doc_ids = [item[0] for item in valid_docs_with_ids]
    #     final_documents = [item[1] for item in valid_docs_with_ids]
    #     bm25_manager = BM25Manager()
    #     bm25_manager.build_index(final_documents, final_doc_ids)
    #     bm25_manager.save_index()
    # except Exception as e:
    #     logger.error(f"BM25索引构建失败:{e}")
    # logger.info("BM25索引构建完成")

    # # 使用tqdm显示向量存储添加文档的进度
    # print("开始构建FAISS索引...")

    # # 批量处理文档以提高效率
    # batch_size = 100  # 可以根据实际情况调整批量大小
    # total_batches = (len(docs) + batch_size - 1) // batch_size

    # for i in tqdm(
    #     range(0, len(docs), batch_size),
    #     desc="构建FAISS索引",
    #     total=total_batches,
    #     unit="批次",
    # ):
    #     batch_docs = docs[i : i + batch_size]
    #     batch_uuids = uuids[i : i + batch_size]
    #     vector_store.add_documents(documents=batch_docs, ids=batch_uuids)

    # print("FAISS索引构建完成，正在保存...")
    # vector_store.save_local("./faiss_index")
    # print("索引已保存到 ./faiss_index")

    new_vector_store = FAISS.load_local(
        "./faiss_index", embedding_model, allow_dangerous_deserialization=True
    )

    r = new_vector_store.similarity_search_with_score("CPU的构成", k=5)
    # print(r)
    s = [result[0].id for result in r]
    print(s)
