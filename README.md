# SKN17-3rd-Team6
> SK네트웍스 Family AI캠프 17기 - 3차 프로젝트 6팀  
  개발 기간: 2025.09.08 ~ 2025.09.25

<br>

# 🔖 Conents
1. [팀 소개](#1-팀-소개)
2. [프로젝트 개요](#2-프로젝트-개요)
3. [기술 스택 & 사용한 모델](#3-기술-스택과-모델)
4. [시스템 아키텍쳐](#4-시스템-아키텍처)
5. [WBS](#5-WBS)
6. [요구사항 명세서](#6-요구사항-명세서)
7. [수집한 데이터 및 전처리 요약](#7-데이터-및-전처리)
8. [DB 연동 구현 코드](#8-Vector-DB-연동)
9. [모델 선정 과정](#9-모델-선정-과정)
10. [테스트 계획](#10-테스트-계획-및-결과)
11. [프로그램 성능 개선 노력](#11-성능-개선-과정)
12. [수행 결과(테스트/시연 페이지)](#12-시연-페이지)
13. [한 줄 회고](#13-한-줄-회고)

<br>
<br>

# 1. 팀 소개
#### 팀명: 💡 RAGents
> RAG + AGents의 합성어. RAG를 활용해 사용자에게 최적화된 챗봇을 만드는 능력자들.
#### 프로젝트명: 🚗🔧 자동차 고장 및 이상현상 정비 챗봇 

### 팀 구성원
| 양정민 | 전상아 | 주수빈 | 최동현 | 홍문봉 |
|------------|------------|------------|------------|------------|
| [@Yangmin](https://github.com/Yangmin3)| [@sang-a-le](https://github.com/sang-a-le) | [@Subin-Ju](https://github.com/Subin-Ju) | [@donghyun4957](https://github.com/donghyun4957) | [@Glowcloudy](https://github.com/Glowcloudy) |


<br>
<br>


# 2. 프로젝트 개요
## 💡 프로젝트 소개
본 프로젝트는 현대/기아 **GSW 정비 매뉴얼**과 외부 데이터(네이버 블로그, 지식인)를 활용하여 차량 이상현상(진동, 소음, 경고등 등)에 대한 진단 및 **DIY 정비 가능 여부**를 안내하는 **Retrieval-Augmented Generation(RAG) 기반 챗봇**입니다.

<br>

## ✅ 프로젝트 필요성

| <img width="1304" height="874" alt="image" src="https://github.com/user-attachments/assets/151b78f2-3148-40f4-b826-b3c010cf9782" /> | <img width="1280" height="1180" alt="image" src="https://github.com/user-attachments/assets/2e8aa593-3542-41b9-a29a-0c85bc2f7820" /> |
|---------------------------|---------------------------|
|[소비자민원평가-자동차; AS에 민원 집중](https://www.consumernews.co.kr/news/articleView.html?idxno=739584) | [반복되는 엔진경고등에도 원인 못찾고 방치... 뒤늦게 고장 판정되면 시간ㆍ비용은 소비자 몫](https://www.consumernews.co.kr/news/articleView.html?idxno=739584) |

2025년 7월 기준, 대한민국 자동차 등록 수는 2천640만 8천대로 집계되었습니다. 대한민국 전체 인구의 절반 가량이 차량을 보유하고 있으며, 그 중 현대와 기아 자동차는 국산차 전체 판매 점유율의 92%를 차지하는 만큼 그 규모가 매우 큽니다.  

최대 판매 규모를 자랑하는 현대 자동차와 기아 자동차는 그 민원 처리 방식 역시 매우 훌륭하나, 그럼에도 불구하고 소비자들은 차량에 고장이 생겼을 경우 다음과 같은 세 가지 문제에 직면하게 됩니다.  
```
1. 고장 원인 진단 및 처리를 받았으나 정확한 원인을 알지 못하는 경우  
2. 고장 원인을 어느정도 알고 있으나 당장 서비스센터에 가야 하는지 그 심각성을 쉽게 짐작하기 어려운 경우  
3. 서비스 센터에서 고장 원인 조차 제대로 파악하지 못한 경우
```

저희의 챗봇은 이러한 문제를 겪고 있는 전국 약 2천 650만 명의 차량 보유자를 위해, 언제 어디서든 **쉽고 간편하게 고장 원인에 대해 일반 자동차 이용자도 파악할 수 있게 돕는** 챗봇을 만들게 되었습니다.  

세 가지 문제를 해결하기 위해, 저희 팀은 다음과 같이 챗봇을 개발하였습니다.  

- **블로그/지식인 크롤링 내용**을 통해 보다 다양한 차량 고장 관련 키워드의 정보를 학습하고  
- **현대자동차/기아자동차 공식 GSW 문서**를 통해 높은 신뢰도를 갖는 정확한 고장 원인 진단 및 해결 방법에 대해 안내할 수 있도록 하며
- 사용자가 서비스 센터에서 고장 진단 및 처리를 받을 시 **정확한 이해**를 돕고자 했습니다.

<br>

## 🎯 프로젝트 목표
- 사용자가 입력한 **이상현상 설명 또는 계기판 경고등**을 기반으로 진단 수행
- **차종·엔진 스펙을 반영한 GSW 기반 표준 진단** + **외부 사용자 경험 데이터(블로그+지식인 크롤링) 결합**
- 경미한 문제의 경우 **DIY 수리 절차 안내**, 중대한 문제는 **정비소 방문 권고**


<br>
<br>

# 3. 기술 스택과 모델
| **Language✍️** | **Tools🪛** | **Embedding😄** | **Vector DB📚** | **LLM🤖** | **FrameWork🪟** | **Demo💡** | **Collaborate🔧** |
|-------------------|----------------|---------------|---------------|-------------|----------------|---------------|--------------|
| ![Python](https://img.shields.io/badge/-Python-3776AB?logo=python&logoColor=white) | ![VS Code](https://img.shields.io/badge/-VS%20Code-007ACC?logo=visualstudiocode&logoColor=white)<br> ![RunPod](https://img.shields.io/badge/-RunPod-5F43DC?logo=cloud&logoColor=white) | ![Hugging Face](https://img.shields.io/badge/-HuggingFace-FFD21F?logo=huggingface&logoColor=black)<br> <sub><a href="https://huggingface.co/dragonkue/snowflake-arctic-embed-l-v2.0-ko">사용한 HF 모델</a></sub> | ![FAISS](https://img.shields.io/badge/-FAISS-009999?logo=meta&logoColor=white) |![EEVE](https://img.shields.io/badge/-EEVE-8A2BE2?logo=alibaba&logoColor=white)<br> | ![LangChain](https://img.shields.io/badge/-LangChain-F9AB00?logo=LangChain&logoColor=white) | ![Streamlit](https://img.shields.io/badge/-Streamlit-FF4B4B?logo=streamlit&logoColor=white) |  ![Discord](https://img.shields.io/badge/-Discord-5865F2?logo=discord&logoColor=white)<br> ![GitHub](https://img.shields.io/badge/-GitHub-181717?logo=github&logoColor=white) |


<br>
<br>

# 4. 시스템 아키텍처

```mermaid
flowchart LR
    subgraph Input
        A1[사용자 텍스트 입력] --> P1
    end

    subgraph Preprocessing
        P1[텍스트 전처리\n토큰화·차종 추출]
    end

    subgraph Retrieval
        R1[GSW 벡터DB\n차종·엔진 기반 1차 검색]
        R2[블로그/지식인 벡터DB\n키워드 기반 2차 검색]
    end

    subgraph RAG
        G1[검색 결과 통합]
        G2[LLM 기반 응답 생성]
    end

    Input --> Preprocessing
    Preprocessing --> Retrieval
    Retrieval --> RAG
    RAG --> Out[최종 답변 제공]

```
<br>

# 5. WBS
![WBS 이미지](./readme_image/WBS.png)

<br>
<br>

# 6. 요구사항 명세서
<img width="1408" height="754" alt="image" src="https://github.com/user-attachments/assets/a317911f-9946-44c6-8868-142b357f16c1" />

<br>
<br>

# 7. 데이터 및 전처리

## 📂 데이터 출처 및 구조화

### 1. 현대/기아 GSW (정비 매뉴얼)
![GSW 페이지 GIF](./readme_image/1.gif)
> PDF 정비지침서 → PDF파일 크롤링 → 텍스트 추출 → JSON 변환

- 단위: **고장진단 절차별 레코드**

- 메타데이터: ``차종``, ``엔진``, ``출처``

```json
"고장진단": [
  {
    "title": "보안 및 차량시동시스템+빌트인캠 영상 녹화 안 됨",
    "content": "퓨즈 단선/개조 확인, DVRS 고장 코드 점검, 모듈·카메라 점검",
    "type": "현대",
    "차종": "EV6(CV)",
    "엔진": "160KW+270KW(4WD)"
  }
]
```

### 2. 네이버 블로그
비공식 사용자 경험 데이터

본문 content 중심으로 **Chunk 단위 분할 및 임베딩**

```json
"문제 키워드": [
  {
    "title": "블로그 제목",
    "content": "블로그 내용",
    "type": "블로그",
    "출처": "https://blog.naver.com/example",
    "차종": null,
    "엔진": null
  }
]
```

### 3. 네이버 지식인
- Q&A 형식

- ``title(질문)`` + ``content(답변)``을 하나의 문서로 통합 후 임베딩

```json
"문제 키워드": [
  {
    "title": "질문",
    "content": ["답변1", "답변2"],
    "type": "지식인",
    "출처": "https://kin.naver.com/qna/example",
    "차종": null,
    "엔진": null
  }
]
```
<br>

## 🏭 데이터 전처리


<br>

## 🔍 검색 및 임베딩 전략
### 1. 임베딩 단위
- 블로그: 본문 → Chunk 분할 후 임베딩

- 지식인: 질문 + 답변 통합 → 단일 벡터 임베딩

- GSW: 고장진단 단위 → 레코드 단위 임베딩

### 2. 검색 우선순위

1. **1차 검색** : GSW 내부 문서 (차종·엔진 기반)

2. **2차 검색** : 블로그/지식인 크롤링 (차종·엔진 없음 → 공통 키워드 기반)

3. **최종 답변** : 두 검색 결과 통합 + 신뢰도 가중치 적용

<br>

## 🧩 핵심 워크플로우 (RAG 기반)
```mermaid
sequenceDiagram
    participant User as 사용자
    participant Pre as 전처리 모듈
    participant DB1 as GSW 벡터DB
    participant DB2 as 블로그/지식인 벡터DB
    participant LLM as LLM 엔진
    
    User->>Pre: "차량 이상현상 입력" 또는 "계기판 이미지 업로드"
    Pre->>DB1: 차종·엔진 기반 1차 검색
    DB1-->>Pre: 표준 진단 결과
    Pre->>DB2: 키워드 기반 2차 검색
    DB2-->>Pre: 외부 사용자 경험
    Pre->>LLM: 통합 검색 결과 전달
    LLM-->>User: 종합 진단 및 대응 방안 제공
```
<br>
<br>

# 8. Vector DB 연동
### 사용한 벡터 DB: FAISS
**FAISS 선택 이유**
```
1. 고성능 유사성 검색: 벡터 데이터에서 가장 가까운 이웃 벡터를 효과적으로 찾아내는 Nearest Neighbor Search 기능을 제공하므로 빠른 검색 효율을 내기에 적합.
2. GPU 지원을 통한 가속 연산: GPU 연결 시 병렬 연산이 지원되어 처리 속도를 크게 향상시킬 수 있음.
3. 다양한 인덱스 유형: 여러 인덱스 유형을 지원하여 데이터 크기와 요구에 맞는 최적화된 검색 성능을 제공함.
4. 대규모 데이터 처리 성능: 다른 Vector DB에 비해 대규모 고차원 벡터 데이터를 빠르게 처리할 수 있는 능력이 뛰어남.   
→ 전통적인 방법으로는 효율적으로 처리하기 어려운 대규모 데이터 처리에 매우 적합. (약 8만 개 + 이후 계속 확장할 예정)
5. 커뮤니티 지원 및 확장성: 많은 엔지니어들이 지속적으로 개선하고 있으므로 확장성ㆍ안정성 면에서 우위를 점하는 DB.

```

- 📁 [vector_store.py] 파일 확인
- 🔗 [벡터 DB 구축(Google Drive)](https://drive.google.com/drive/folders/116zAgunFJb1ZxaShQKSVPMpM7oNylaXX?usp=sharing)

<br>

## ✅ DB 연동 구현 내용

<br>
<br>

# 9. 모델 선정 과정

<br>
<br>

# 10. 테스트 계획 및 결과
## ✏️ 테스트 계획
| **정확도 항목** | **평가 항목** | **질문** | **의도**|
|---------------|------------------------|-----------------|--------------|
| **1** | 의성어, 의태어를 잘 인식하는지 | 1. 하부에서 그그극, 드르륵, 드드득 소리가 나.<br>2. 에어컨 필터가 덜렁거려. | 1. 해당 의성어를 '소음'으로 인식<br>2. 에어컨 필터가 떨어진 '현상'으로 인식 |
| **2** | 매뉴얼 내용을 잘 이해하고 요약해서 반환하는지 | - 길이 별 구분<br>1. kia 그랜드카니발(VQ)의 스티어링 시스템 일반사항 고장 진단법(572자)<br>2.kia K9(RJ)의 드라이브샤프트 및 액슬의 소음(197자)<br>3. Hyudai G80(RG3 EV)의 로어 페달 또는 스폰지 페달 고장진단(69자) | 상세한 차량 모델 및 증상을 반환했을 때 기존 매뉴얼 내용을 기본으로 요약된 내용 및 사용자 친화적 표현 방식으로 답변을 반환 |
| **3** | 꼬리질문에 대한 대응도 | 1. 브레이크가 밀리는 데 왜 그런거야? | 해당 질문 후 4) 항목에 대해 질문했을 때, 이전 질문 내용을 기억하고, 적절히 답변 |
| **4** | 사용자의 질문을 잘 이해하고 답변하는지 | 2. 브레이크 패드가 이상이 있는 것처럼 보여. 지금 바로 정비소에 가야 해? | 2. 정비소에 가야하는 지를 물어보는 질문에서 요점을 파악하고, 브레이크 패드에 대한 설명이 아닌 정비소 방문 여부에 대해 반환 |

<br>

## 📑 테스트 결과

<br>
<br>

# 11. 성능 개선 과정
## ⚠️ 기술적 고려사항
1. **신뢰도 가중치**

- GSW > 지식인 > 블로그 순

- 외부 데이터는 신뢰도 평가 지표 필요 (예: 답변 채택 여부, 댓글 반응 등)

2. **데이터 불균형**

- 블로그/지식인 ≫ GSW → GSW 기반 검색 우선 적용

- 메타데이터 필터링 강화로 차량별 매칭 정확도 확보


3. **키워드 크롤링 한계**

- 자동차 무관 데이터 혼입 → 사전 필터링 필요

- 불필요 키워드 자동 제거 로직 포함

## 🚀 기대 효과
운전자: 정비소 방문 전 **자가 진단 가능**

정비 효율: **문제 사전 파악 → 불필요 방문 감소**

데이터적 측면: **기업 매뉴얼 + 사용자 경험 데이터 융합**

기술적 확장성: **추후 다른 브랜드 차량 GSW·포럼 데이터로 확장 가능**

<br>
<br>

# 12. 시연 페이지

<br>
<br>

# 13. 한 줄 회고
- 양정민 : 
- 전상아 :
- 주수빈 : 
- 최동현 : 
- 홍문봉 : 
