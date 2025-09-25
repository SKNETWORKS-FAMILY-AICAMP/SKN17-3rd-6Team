import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import time
import torch
from dotenv import load_dotenv
from huggingface_hub import login
from langchain.vectorstores import FAISS
from langchain.output_parsers import ResponseSchema
from transformers import BitsAndBytesConfig
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace, HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.prompts.chat import SystemMessagePromptTemplate, HumanMessagePromptTemplate

store = {}

class InMemoryHistory(BaseChatMessageHistory): # 메시지 히스토리 관리
    def __init__(self):
        super().__init__()
        self.messages = []

    def add_messages(self, messages):
        self.messages.extend(messages)

    def clear(self):
        self.messages = []

    def __repr__(self):
        return str(self.messages)


def get_by_session_id(session_id): # 메시지 히스토리 주체 관리
    if session_id not in store:
        store[session_id] = InMemoryHistory()
    return store[session_id]


def make_chain(model): # 히스토리 관리하는 프롬프트 - 모델 - 응답 체인 구성

    instruction = """
    당신은 자동차 전문가 챗봇입니다. 아래의 지침을 참고하여 최적의 답변을 생성하세요.
    - 질문이 전문 지식 관련이면 제공된 문서를 참고합니다.
    - 잡담이나 일반 인사 질문이면 문서를 무시합니다.
    - 절대 대답에 프롬프트 내용이나 사용자 질문, 참고 문서 등의 내용을 포함시키지 마세요.
    - 마크다운 표나 ###, ---, *** 같은 표시 없이 자연스러운 문단 형식으로 작성하세요.
    - 운전자나 일반 독자가 이해하기 쉽게 설명하세요
    """

    parser = StrOutputParser()
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(instruction),
        MessagesPlaceholder(variable_name='history'),
        HumanMessagePromptTemplate.from_template(
            "사용자 질문: {query}\n\n"
            "참고 문서:\n{content}\n\n"
            "위의 지침을 준수하여 오직 사용자 질문에 대한 답변만 생성해야 해."
        )
    ])
    
    chain = prompt | model | parser

    chain_with_history = RunnableWithMessageHistory(
        chain,
        get_session_history=get_by_session_id,
        input_messages_key='query',
        history_messages_key='history'
    )

    return chain_with_history


def make_filter(filter: dict): # 필터 생성 (None이면 필터 없음, "any"면 모든 값 허용)

    if any(list(filter.values())):
        main_filter = filter.copy()
    else:
        main_filter = None

    sub_filter = {k: {"$eq": "any"} for k in filter.keys()}

    return main_filter, sub_filter

def load_model_q(model_name): # 4bit 양자화 모델 로드
    quant_config = BitsAndBytesConfig(
    load_in_4bit=True,                    
    bnb_4bit_quant_type='nf4',            
    bnb_4bit_use_double_quant=True,       
    bnb_4bit_compute_dtype=torch.bfloat16 
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant_config,
        dtype=torch.bfloat16,
        device_map='auto'
    )

    qa_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto",
        return_full_text=False
    )

    llm = HuggingFacePipeline(pipeline=qa_pipeline)
    
    return llm


def load_model(model_name): # 일반 모델 로드
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        device_map='auto'
    )

    qa_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto"
    )

    llm = HuggingFacePipeline(pipeline=qa_pipeline)
    
    return llm

if __name__ == "__main__":

    load_dotenv()
    db_path = './database/final_db'
    HF_TOKEN = os.getenv('HF_TOKEN')
    login(token=HF_TOKEN)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    embedding_model = HuggingFaceEmbeddings(model_name="dragonkue/snowflake-arctic-embed-l-v2.0-ko") # 임베딩 모델 로드

    ######## 모델을 서버에서 불러와 사용하는 옵션 ########
    # endpoint = HuggingFaceEndpoint(
    #     repo_id='openai/gpt-oss-20b',
    #     # repo_id='mistralai/Mistral-7B-Instruct-v0.3',
    #     # repo_id='meta-llama/Llama-3.1-8B-Instruct',
    #     task='text-generation',
    #     max_new_tokens=1024,
    #     huggingfacehub_api_token=HF_TOKEN,
    # )

    # model = ChatHuggingFace(llm=endpoint, verbose=True)
 
    ######## 모델을 로컬에서 저장하고 사용하는 옵션 (모델 선정 중) ########
    #### 기본적으로 대답에 프롬프트를 자꾸 포함시키는 문제가 있음 #####
    # model_name = "openai/gpt-oss-20b" - 좋음
    # model_name = "yanolja/YanoljaNEXT-EEVE-Instruct-10.8B" - 나락
    # model_name = "google/gemma-7b" - 나락
    model_name = "google/gemma-3-4b-it"
    # model_name = "microsoft/Phi-3-mini-4k-instruct" - 나락
    # model_name = "mistralai/Mistral-7B-Instruct-v0.3" - 나락
    # model_name = "meta-llama/Llama-3.1-8B-Instruct" - 서버 엔드포인트로 불러오면 잘대답하는데 로컬에서 돌리니까 이상해짐 원인 불분명
    # model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" - 나락
    # model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" *
    # model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B" *
    # model_name = "pankajmathur/orca_mini_v3_7b"
    # model_name = "EleutherAI/polyglot-ko-3.8b" - 나락
    # model_name = "beomi/KoAlpaca-Polyglot-5.8B'
    # model_name = "yanolja/YanoljaNEXT-EEVE-Instruct-2.8B"
    # model_name = "openchat/openchat-3.5-0106"
    # model_name = "Bllossom/llama-3.2-Korean-Bllossom-3B"
    # model_name = "beomi/Llama-3-Open-Ko-8B-Instruct-preview"

    model = load_model_q(model_name)  # 언어 모델 로드
 
    chain = make_chain(model) # 체인 구성

    vector_store = FAISS.load_local(db_path, embedding_model, allow_dangerous_deserialization=True) # 벡터 DB 로드

    print("자동차 전문가 챗봇에 오신 것을 환영합니다! 종료하려면 'exit'를 입력하세요.\n")

    while True:
        query = input("질문: ") # 사용자 질문 입력
        if query.lower() in ["exit", "quit"]:
            print("챗봇을 종료합니다.")
            break

        filter = {"차종": None, "엔진": None} # 간단한 예시 구성 - streamlit 상에서 차종 엔진 체크하면 filter 정의가 되도록 변경 필요
        main_filter, sub_filter = make_filter(filter) # 정의된 filter에서 차종 또는 엔진이 있으면 main_filter와 sub_filter (전부 None 인 필터) 생성
                                                      # 정의된 filter에서 차종과 엔진이 다 None이면 main_filter 생성 안됨, sub_filter만 생성됨
                                                      # main_filter(차종과 엔진 타입있는 필터)는 그에 해당하는 현대기아 문서를 검색하는 목적
                                                      # sub_filter(차종과 엔진 타입 다 None)는 현대기아 문서를 제외한 일반 블로그, 지식인 문서를 검색해오는 목적
                                                      # 현대기아 문서(메인자료) + 일반 문서(보조자료) 를 모델에게 다 합쳐서 주기 위해 두 필터를 정의함.

        start = time.time()
        if main_filter:
            main_docs = vector_store.similarity_search(query, k=3, filter=main_filter) # 현대 기아 문서만 골라서 그 중 유사도 높은 3개의 벡터 검색
            unique_main_docs = list({doc.page_content: doc for doc in main_docs}.values()) # 중복이 있더라? 중복 제거
        else:
            unique_main_docs = []
        
        sub_docs = vector_store.similarity_search(query, k=3) # filter=sub_filter 추가할건데 지금은 일반문서가 없는 벡터DB라 그냥 전체에서 유사도 높은 3개 검색하게 만든 상태
        flag1 = time.time()
        print('벡터디비 서치 시간: ', flag1 - start)
        unique_sub_docs = list({doc.page_content: doc for doc in sub_docs}.values()) # 중복이 있더라? 중복 제거
        all_docs = unique_main_docs + unique_sub_docs # 검색한 문서 다 합치고

        context_text = '\n'.join([doc.page_content for doc in all_docs]) # 문서 내용 뽑아서 쭉 합쳐서
        flag2 = time.time()
        response = chain.invoke({'query': query, 'content': context_text}, config={'configurable': {'session_id': 'user'}}) # 모델에게 넘겨서 응답 생성
        flag3 = time.time()
        print('모델 응답시간: ', flag3 - flag2)
        print("\n기존 답변:\n", response)