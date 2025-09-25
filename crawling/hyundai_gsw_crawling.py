import os
import re
import time
import json
import requests
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from custom_langchain import refine_text

def clean_text(raw_text):
    """
    - 줄바꿈 제거
    - 연속 공백 제거
    - '•' 기준 문단 분리
    """
    text_no_newline = re.sub(r'\n', ' ', raw_text)
    text_cleaned = re.sub(r'\s+', ' ', text_no_newline).strip()
    
    paragraphs = [p.strip() for p in text_cleaned.split('•') if p.strip()]
    return paragraphs


def traverse_folder(folder_id, model, current_path=None):
    if current_path is None:
        current_path = []

    try:
        folder = driver.find_element(By.ID, folder_id)
        folder_name = folder.text

        folder.click()
        time.sleep(1)

        # 현재 경로에 폴더 이름 추가
        new_path = current_path + [folder_name]

        # 하위 dt 확인
        sub_container_id = folder_id.replace("IN", "SUB")
        sub_dt_list = driver.find_elements(By.XPATH, f'//*[@id="{sub_container_id}"]/dt')

        if sub_dt_list:  # 하위 폴더가 있으면
            dt_spans = driver.find_elements(By.XPATH, f'//*[@id="{sub_container_id}"]/dt/span')
            for dt_span in dt_spans:
                dt_id = dt_span.get_attribute("id")
                traverse_folder(dt_id, model, new_path)
        else:  # 하위 폴더 없으면 dd(문서) 확인
            sub_dd_list = driver.find_elements(By.XPATH, f'//*[@id="{sub_container_id}"]/dd')
            for dd in sub_dd_list:
                title = dd.get_attribute("title")
                if "고장진단" in title:
                    print("고장진단 문서 발견:", title)
                    dd.click()
                    time.sleep(1)
                    
                    content = driver.page_source
                    soup = BeautifulSoup(content, "html.parser")
                    text_content = soup.find("div", {"id": "contentData"}).get_text(separator="\n", strip=True)
                    paragraphs = clean_text(text_content)
                    refined_text = refine_text(model, "\n".join(paragraphs))

                    # 파일명 생성
                    file_name = "_".join(new_path + [title]) + ".txt"
                    file_name = file_name.replace("/", "_")
                    print(file_name)

                    # 파일 저장
                    save_dir = "crawl_result"
                    os.makedirs(save_dir, exist_ok=True)
                    file_path = os.path.join(save_dir, file_name)
                    file_path = file_path.replace(" ", "")
                    with open(file_path, "w", encoding="utf-8") as f:
                        f.write(json.dumps(refined_text, ensure_ascii=False, indent=2))
                    print("크롤링 및 저장 완료:", file_path)

    except Exception as e:
        print("오류 발생:", e)

load_dotenv()
ID = os.getenv("ID")
PW = os.getenv("PW")
HF_TOKEN = os.getenv('HF_TOKEN')

endpoint = HuggingFaceEndpoint(
    repo_id='openai/gpt-oss-20b',
    task='text-generation',
    max_new_tokens=1024,
    huggingfacehub_api_token=HF_TOKEN
)

model = ChatHuggingFace(
    llm=endpoint,
    verbose=True
)


# 1. Selenium으로 로그인
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
driver.get("https://gsw.hyundai.com/")

time.sleep(2)

next_btn = driver.find_element(By.XPATH, '//*[@id="login"]/div[2]/div[3]/div/a/span')
next_btn.click()
time.sleep(1)

driver.find_element(By.ID, "Userid").send_keys(ID)
driver.find_element(By.ID, "Passwd").send_keys(PW)
driver.find_element(By.ID, "LoginButton").click()

cookies = driver.get_cookies()

next_btn = driver.find_element(By.XPATH, '//*[@id="PROGRESS_WHEEL"]/div[9]/div[3]/div/a/span').click()
driver.find_element(By.XPATH, '//*[@id="PROGRESS_WHEEL"]/div[8]/div[3]/div/a/span').click()
driver.find_element(By.XPATH, '//*[@id="gnb"]/ul/li[1]/a').click()

time.sleep(1)  # 로그인 완료 기다리기
driver.find_element(By.XPATH, '//*[@id="mdlCd"]').click()
driver.find_element(By.XPATH, '//*[@id="mdlCd"]/option[2]').click() # 차종 선택, for 문 돌려가며 가능 2~115까지

driver.find_element(By.XPATH, '//*[@id="year"]').click()
driver.find_element(By.XPATH, '//*[@id="year"]/option[2]').click() # 연식 선택, 2번으로 가장 최신 연도 선택

driver.find_element(By.XPATH, '//*[@id="engCd"]').click()
eng_select = driver.find_element(By.ID, "engCd")
options = eng_select.find_elements(By.TAG_NAME, "option")
print("엔진 옵션 개수:", len(options))
driver.find_element(By.XPATH, '//*[@id="engCd"]/option[2]').click() # 연식 선택, for 문 돌려가며 가능 2~115까지

time.sleep(5)

driver.find_element(By.ID, "IN_1").click()
time.sleep(1)

top_dt_list = driver.find_elements(By.XPATH, '//*[@id="SUB_1"]/dt') # 일반사항 / 엔진 기계 시스템 / 엔진 전장 시스템 / .... 히터 및 에어컨 장치
for dt in top_dt_list:
    dt_span = dt.find_element(By.TAG_NAME, "span")
    dt_id = dt_span.get_attribute("id")
    traverse_folder(dt_id, model)

