from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from pydantic import BaseModel, Field
import logging
import json, sys
import os
from typing import List, AnyStr

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

# 添加到 sys.path
sys.path.append(parent_dir)
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


from utils import lazy_load_faiss
from FileProcess.Embedding import get_embedding_model


class TOPIC(BaseModel):
    topic: str = Field(description="题目所属科目")
    key_words: List = Field(description="题目考察知识点关键字")
    ger_query: List = Field(description="生成的查询")


class TopicUnderstand:
    def __init__(self, openai_api_key=None, base_url=None, model_name="gpt-4o") -> None:
        self.api_key = openai_api_key
        self.base_url = base_url
        self.model_name = model_name
        self.init_instance()

    def init_instance(self):
        if self.api_key and self.base_url:
            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        else:
            logger.info("no api key and base url pass to OCR part,use .env file")
            self.api_key = os.getenv("OPENAI_API_KEY")
            self.base_url = os.getenv("OPENAI_API_BASE")
            self.llm = ChatOpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                model_name=self.model_name,
                streaming=True,
            )

    def analysis(self, content: str, struct: dict):
        """
        分析题目
        """
        pass

    def faiss_search(self, querys: List[str]):

        search_result = []

        embedding_model = get_embedding_model()
        faiss_search_engine = lazy_load_faiss(embedding_model=embedding_model)

        for i, query in enumerate(querys):
            result = faiss_search_engine.similarity_search(query, k=5)
            print(result)
            print()

        return faiss_search_engine


class StructuredAnalysis:
    def __init__(self, openai_api_key=None, base_url=None, model_name="gpt-4o") -> None:
        self.api_key = openai_api_key
        self.base_url = base_url
        self.model_name = model_name
        self.parser = JsonOutputParser(pydantic_object=TOPIC)
        self.init_instance()

    def init_instance(self):
        if self.api_key and self.base_url:
            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        else:
            logger.info("no api key and base url pass to OCR part,use .env file")
            self.api_key = os.getenv("OPENAI_API_KEY")
            self.base_url = os.getenv("OPENAI_API_BASE")
            self.llm = ChatOpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                model_name=self.model_name,
                streaming=False,
            )

    def analysis(self, topic: str):
        """
        结构化解析题干信息
        """
        prompt = ChatPromptTemplate.from_template(
            """
        Please perform a structured analysis of the given question stem and return the results in the specified JSON format. The analysis objectives include the following aspects:

        1. Determine the subject category:
        Select the most appropriate subject from these four options:

        ["Computer Organization", "Data Structures", "Operating Systems", "Computer Networks"]

        2. Identify the core exam topics covered by the question:
        Based on the 408 Computer Science Foundation Comprehensive Exam syllabus, extract the key knowledge points examined in the question stem (be as specific as possible) and present them in list format (recommended not to exceed 5 keywords).

        3. Generate semantic matching query text:
        Create search queries for semantic similarity retrieval by combining the question content with relevant knowledge points.

        The query text should consist of explanations or definitional descriptions of keywords, adhering to textbook-style content.
        The purpose is to enable semantic matching with educational materials, thus avoiding vague expressions and emphasizing accuracy and standardized terminology.

        4. Output format requirements:
        Return results in the following JSON format without any additional fields or explanations:

                {{
        "subject": "Subject Category",
        "key_words": ["Knowledge Point 1", "Knowledge Point 2", "..."],
        "ger_query": ["Explanation of Knowledge Point 1", "Explanation of Knowledge Point 2", "..."]
        }}

        5.Alwayse answer question in chinese (including the json content)!

        User input:
        {topic}
        """
        )

        chain = prompt | self.llm | self.parser
        try:
            result = chain.invoke({"topic": topic})

            logger.info(
                f"""结构化解析完成
                        科目:{result.get("subject")},
                        关键词:{result.get("key_words")},
                        查询：{result.get("ger_query")}
                        """
            )
            return result
        except Exception as e:
            logger.error(f"题目结构解析错误: {e}")
            return {"error": 1}


def main():
    model = StructuredAnalysis()
    result = model.analysis(
        """1.下列关于客户/服务器（C/S）模型的描述，正确的是（）。  

A.客户端需要提前知道服务器的地址，服务器不需要提前知道客户端的地址  
B.所有程序在进程通信中的客户端与服务器端的地位保持不变  
C.客户端之间可以直接通信  
D.服务器面向用户，客户端面向任务   """
    )

    model2 = TopicUnderstand()
    model2.faiss_search(result.get("ger_query"))


main()
