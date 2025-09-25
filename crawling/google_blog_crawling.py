import os
import json
import requests
from dotenv import load_dotenv
from crawling import get_content

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CX = os.getenv("GOOGLE_CX")

keywords = ["자동차 소음", "자동차 브레이크 밀림"]
results = {}

for keyword in keywords:
    contents = []
    seen_links = set()

    for start in range(1, 100, 10):
        google_url = f"https://www.googleapis.com/customsearch/v1?q={keyword}&key={GOOGLE_API_KEY}&cx={GOOGLE_CX}&num=10&start={start}"
        response = requests.get(google_url)

        if response.status_code == 200:
            data = response.json()
            for item in data["items"]:
                link = item.get('link')
                if not link or link in seen_links:
                    continue
                blog_title, blog_content = get_content(item)
                if blog_content != "내용 없음":
                    content = {'제목': blog_title, '내용': blog_content, '블로그': item['displayLink'], '링크': link}
                    contents.append(content)
                    seen_links.add(link)
        else:
            print(f"Error {response.status_code}: {response.text}")
    
    results[keyword] = contents
    print(f"키워드 {keyword}에는 총 {len(contents)} 개 결과 존재")

with open("google_search_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=4)

print("JSON 파일로 저장 완료")