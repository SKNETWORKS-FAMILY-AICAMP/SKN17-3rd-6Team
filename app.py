import os
import streamlit as st
import json
import torch
from datetime import datetime
from dotenv import load_dotenv
from langchain.llms import Ollama
from langchain.vectorstores import FAISS
from langchain.output_parsers import ResponseSchema
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.prompts.chat import SystemMessagePromptTemplate, HumanMessagePromptTemplate
from collections import defaultdict
from langchain_huggingface import HuggingFaceEmbeddings

# 환경 변수 로드
load_dotenv()

# 전역 변수
VECTOR_STORE = None
CHAT_MODEL = None
CHAIN = None
CAR_DATA = None

# 페이지 설정
st.set_page_config(
    page_title="🔧 AutoFix 자동차 진단센터",
    page_icon="🔧",
    layout="wide"
)

# CSS 스타일링
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    
    .header-container {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        border: 3px solid #ffd700;
        box-shadow: 0 8px 25px rgba(0,0,0,0.3);
    }
    
    .chat-message {
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 15px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    
    .user-message {
        background: linear-gradient(135deg, #007bff, #0056b3);
        color: white;
        margin-left: 20%;
        border-bottom-right-radius: 5px;
    }
    
    .bot-message {
        background: linear-gradient(135deg, #f8f9fa, #e9ecef);
        color: #333;
        margin-right: 20%;
        border-left: 4px solid #ffd700;
        border-bottom-left-radius: 5px;
    }
    
    .diagnostic-card {
        background: linear-gradient(45deg, #28a745, #20c997);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 5px solid #ffd700;
        box-shadow: 0 4px 15px rgba(40, 167, 69, 0.3);
    }
    
    .car-info-box {
        background: linear-gradient(135deg, #17a2b8, #138496);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
    }
    
    .tool-icons {
        font-size: 2rem;
        margin: 1rem 0;
    }
    
    .loading-spinner {
        text-align: center;
        padding: 2rem;
        color: #007bff;
    }
</style>
""", unsafe_allow_html=True)


# 메시지 히스토리 관리 클래스
store = {}

class InMemoryHistory(BaseChatMessageHistory):
    def __init__(self):
        super().__init__()
        self.messages = []

    def add_messages(self, messages):
        self.messages.extend(messages)

    def clear(self):
        self.messages = []

    def __repr__(self):
        return str(self.messages)

def get_by_session_id(session_id):
    if session_id not in store:
        store[session_id] = InMemoryHistory()
    return store[session_id]


# RAG 시스템 초기화
def initialize_rag_system():
    db_path = './database/final_db'
    device = "cuda" if torch.cuda.is_available() else "cpu"
        
    # 임베딩 모델 로드
    embedding_model = HuggingFaceEmbeddings(
        model_name="dragonkue/snowflake-arctic-embed-l-v2.0-ko",
        model_kwargs={"device": "cpu"}
    )
        
    vector_store = FAISS.load_local(
        db_path, 
        embedding_model, 
        allow_dangerous_deserialization=True
    )

    model_name = 'EEVE-Korean-10.8B:latest'
    chat_model = Ollama(model=model_name)

    # 체인 생성
    chain = make_chain(chat_model)

    return vector_store, chain
    


# 차량 데이터 추출
def extract_car_data_from_db(vector_store):
    car_data = defaultdict(lambda: defaultdict(lambda: {"engines": set()}))
    
    try:
        all_docs = vector_store.similarity_search("", k=3000)
        
        for doc in all_docs:
            metadata = dict(doc.metadata) if doc.metadata else {}

            model_type = metadata.get('type', '').strip()
            model = metadata.get('car_type', '').strip()
            engine = metadata.get('engine_type', '').strip()
            
            if model_type and model:
                if engine:
                    car_data[model_type][model]["engines"].add(engine)
        
        # Set을 리스트로 변환
        for model_type in car_data:
            for model in car_data[model_type]:
                car_data[model_type][model]["engines"] = sorted(list(car_data[model_type][model]["engines"]))
                
        return dict(car_data)

    except Exception as e:
        st.error(f"차량 데이터 추출 중 오류: {str(e)}")
        return {}


# 체인 생성
def make_chain(model):
    instruction = """
    당신은 자동차 전문가 챗봇입니다. 아래의 지침을 참고하여 최적의 답변을 생성하세요.
    - 질문이 전문 지식 관련이면 제공된 문서를 참고하여 답변합니다.
    - 잡담이나 일반 인사 질문이면 문서를 무시하고 질문만 답변합니다.
    - 답변은 반드시 한국어로만 하며, 영어를 절대로 사용하지 마세요.
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


# 필터 생성
def make_filter(filter: dict):
    if any(list(filter.values())):
        main_filter = filter.copy()
    else:
        main_filter = None

    sub_filter = {k: {"$eq": "any"} for k in filter.keys()}

    return main_filter, sub_filter


# 문서 검색
def search_documents(query, car_info, vector_store, k=5):
    main_filter, sub_filter = make_filter(car_info)
    
    if main_filter:
        main_docs = vector_store.similarity_search(query, k=1, filter=main_filter)
        unique_main_docs = list({doc.page_content: doc for doc in main_docs}.values())
    else:
        unique_main_docs = []
    
    sub_docs = vector_store.similarity_search(query, k=1)
    unique_sub_docs = list({doc.page_content: doc for doc in sub_docs}.values())
    all_docs = unique_main_docs + unique_sub_docs

    context_text = '\n'.join([doc.page_content for doc in all_docs])

    return context_text


# 세션 상태 초기화
def initialize_session_state():
    if 'page' not in st.session_state:
        st.session_state.page = 'setup'
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'car_info' not in st.session_state:
        st.session_state.car_info = {}


# 차량 설정 페이지
def setup_page(car_data=None):
    st.markdown("""
    <div class="header-container">
        <h1>🔧 AutoFix 자동차 진단센터 🔧</h1>
        <div class="tool-icons">🚗 🔍 ⚙️ 🛠️</div>
        <h3>정확한 진단을 위해 차량 정보를 선택해주세요</h3>
        <p>※ 차종을 모르시면 브랜드만 선택하고 진행하셔도 됩니다</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🚗 자동차 브랜드")
        brand = ["브랜드 선택", 'kia', 'hyundai']
        selected_brand = st.selectbox("브랜드를 선택하세요", brand, key="brand_select")
        
        if selected_brand != "브랜드 선택":
            st.markdown("### 🚙 차종 선택")
            models = list(car_data[selected_brand].keys())
            
            model_input = st.text_input(
                "차종을 입력하세요 (자동완성)", 
                placeholder=f"예: {', '.join(models[:2])}", 
                key="model_input"
            )
            
            if model_input:
                filtered_models = [model for model in models if model_input.lower() in model.lower()]
                if filtered_models:
                    selected_model = st.selectbox("매칭되는 차종", filtered_models, key="model_select")
                else:
                    st.warning("⚠️ 매칭되는 차종이 없습니다. 아래 목록에서 선택해주세요.")
                    selected_model = st.selectbox("전체 차종 목록", ["차종 선택"] + models, key="model_select_all")
                    if selected_model == "차종 선택":
                        selected_model = None
            else:
                selected_model = st.selectbox("차종을 선택하세요", ["차종 선택"] + models, key="model_select_default")
                if selected_model == "차종 선택":
                    selected_model = None
        else:
            selected_model = None
    
    with col2:
        if selected_brand != "브랜드 선택":
            if selected_model:
                engines = car_data[selected_brand][selected_model]["engines"]
                
                st.markdown("### ⚙️ 엔진 타입")
                selected_engine = st.selectbox("엔진을 선택하세요", ["엔진 선택"] + engines, key="engine_select")
                if selected_engine == "엔진 선택":
                    selected_engine = None
            else:
                st.info("📝 차종을 먼저 선택해주세요")
                selected_engine = None
        else:
            st.info("📝 브랜드를 먼저 선택해주세요")
            selected_engine = None
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🔍 진단 시작하기", type="primary", use_container_width=True):
            if selected_brand != "브랜드 선택":
                st.session_state.car_info = {"브랜드": selected_brand or "미지정", "차종": selected_model or "미지정", "엔진": selected_engine or "미지정"}
                st.session_state.page = 'chat'
                st.rerun()
            else:
                st.error("❌ 최소한 브랜드는 선택해주세요!")


# 채팅 페이지
def chat_page(chain, vector_store):
    car_info = st.session_state.car_info

    st.markdown(f"""
    <div class="header-container">
        <h1>🔧 AutoFix AI 진단상담</h1>
        <div class="car-info-box">
            🚗 {car_info['차종']} 
            ⚙️ {car_info['엔진']}
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    if st.session_state.chat_history:
        st.markdown("### 💬 상담 내역")
        for message in st.session_state.chat_history:
            if message["type"] == "user":
                st.markdown(f'<div class="chat-message user-message">👤 <strong>고객:</strong><br>{message["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="chat-message bot-message">🤖 <strong>AI 진단사:</strong><br>{message["content"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="chat-message bot-message">
            🤖 <strong>AI 진단사:</strong><br>
            안녕하세요! AutoFix AI 진단센터입니다.<br>
            차량의 증상이나 문제점을 자세히 설명해주시면 데이터베이스를 기반으로 정확한 진단을 도와드리겠습니다.<br><br>
            <strong>예시:</strong><br>
            • "시동 걸 때 엔진에서 덜덜거리는 소리가 나요"<br>
            • "가속할 때 차가 떨려요"<br>
            • "엔진 경고등이 켜졌어요"<br>
            • "P0171 오류 코드가 나왔어요"
        </div>
        """, unsafe_allow_html=True)
    
    # ✅ 채팅 입력 (엔터 전송 + 자동 초기화)
    if prompt := st.chat_input("차량의 증상이나 문제를 입력하세요..."):
        st.session_state.chat_history.append({
            "type": "user",
            "content": prompt,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })

        with st.spinner("🤖 AI가 데이터베이스를 검색하여 진단하고 있습니다..."):
            response = perform_rag_diagnosis(prompt, chain, vector_store, car_info)

        st.session_state.chat_history.append({
            "type": "bot",
            "content": response,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })

        st.rerun()
    
    with st.sidebar:
        st.markdown("### 🚗 등록 차량")
        st.info(f"""
        **차종:** {car_info['차종']}  
        **엔진:** {car_info['엔진']}  
        """)
        
        if st.button("🔄 차량 변경하기"):
            st.session_state.page = 'setup'
            st.session_state.chat_history = []
            st.cache_resource.clear()
            st.rerun()
            
        st.markdown("---")
        st.markdown("### 📊 진단 정보")
        st.success("""
        ✅ **RAG 기반 진단**  
        • 실시간 데이터베이스 검색  
        • 차량별 맞춤 진단  
        • 고장 원인 분석  
        • GSW 오류 코드 해석   
        • 정비 가이드 제공  
        """)
        
        st.markdown("---")
        st.markdown("### 📞 응급연락처")
        st.error("""
        🚨 **응급상황**  
        • 화재/사고 : 119  
        • 한국도로공사: 1588-2504  
        • 기아 긴급출동: 080-200-2000  
        • 현대 긴급출동: 080-600-6000
        """)
        
        if st.session_state.chat_history:
            st.markdown("---")
            if st.button("🗑️ 대화내역 삭제"):
                st.session_state.chat_history = []
                if 'user' in store:
                    store['user'].clear()
                st.rerun()


# RAG 진단 수행
def perform_rag_diagnosis(query, chain, vector_store, car_info):
    main_filter, sub_filter = make_filter(car_info)
    
    if main_filter:
        main_docs = vector_store.similarity_search(query, k=1, filter=main_filter)
        unique_main_docs = list({doc.page_content: doc for doc in main_docs}.values())
    else:
        unique_main_docs = []
    
    sub_docs = vector_store.similarity_search(query, k=1)
    unique_sub_docs = list({doc.page_content: doc for doc in sub_docs}.values())
    all_docs = unique_main_docs + unique_sub_docs

    context_text = '\n'.join([doc.page_content for doc in all_docs])
    response = chain.invoke({'query': query, 'content': context_text}, config={'configurable': {'session_id': 'user'}})

    return response


# 메인 실행
def main():
    initialize_session_state()
    vector_store, chain = initialize_rag_system()
    
    st.session_state["CAR_DATA"] = extract_car_data_from_db(vector_store)
    car_data = st.session_state.get("CAR_DATA")

    if st.session_state.page == 'setup':
        setup_page(car_data)
    elif st.session_state.page == 'chat':
        chat_page(chain, vector_store)

if __name__ == "__main__":
    main()