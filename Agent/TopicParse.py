from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import logging
import os
logger=logging.getLogger(__name__)

class TopicUnderstand:
    def __init__(self,openai_api_key,base_url,model_name='gpt-4o') -> None:
        self.api_key=openai_api_key
        self.base_url=base_url
        self.model_name=model_name
        self.init_instance()

    def init_instance(self):
        if self.api_key and self.base_url:
            self.client=OpenAI(api_key=self.api_key,base_url=self.base_url)
        else:
            logger.info("no api key and base url pass to OCR part,use .env file")
            self.api_key=os.getenv('OPENAI_API_KEY')
            self.base_url=os.getenv('OPENAI_API_BASE')
            self.llm=ChatOpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                model_name=self.model_name,
                streaming=True,
            )
            
class StructuredAnalysis:
    def __init__(self,openai_api_key,base_url,model_name='gpt-3.5-turbo') -> None:
        self.api_key=openai_api_key
        self.base_url=base_url
        self.model_name=model_name
        self.output_parser=StrOutputParser()
        self.init_instance()

    def init_instance(self):
        if self.api_key and self.base_url:
            self.client=OpenAI(api_key=self.api_key,base_url=self.base_url)
        else:
            logger.info("no api key and base url pass to OCR part,use .env file")
            self.api_key=os.getenv('OPENAI_API_KEY')
            self.base_url=os.getenv('OPENAI_API_BASE')
            self.llm=ChatOpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                model_name=self.model_name,
                streaming=False
            )
    def analysis(self,topic:str):
        '''
            结构化解析题干信息
        '''
        prompt=ChatPromptTemplate.from_template(
            '''
            请你根据以下题干，结构化分析题目，分析目标包括:
            确定所属科目,从[计算机组成原理，数据结构，操作系统，计算机网络]中选择。
            以json格式返回数据，例如: {{"subject": "计算机组成原理"}}
            ______________________________________________________
            下面是输入的题目：{topic}
            ______________________________________________________
            '''
        )


        message=prompt.invoke({"topic":topic})
        response=self.llm.invoke(message)
        response=self.output_parser.invoke(response)
        
        try:
            response=json.loads(response)
            subject=response.get('subject')
        except Exception as e:
            logger.error(e)
            