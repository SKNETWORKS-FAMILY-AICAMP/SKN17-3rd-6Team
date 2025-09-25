import os
import numpy as np
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
path = "./merged_results/blog_merged.json"
client = OpenAI()

new_content = []

instruction = """
    당신은 능력있는 블로그 글 요약가 입니다.
    아래 블로그 글을 요약해주세요.
    너무 짧게 요약하지 말되, 요약한 글은 최대 800자가 넘지 않아야 합니다.
"""

with open('./merged_results/index_list.json', 'r', encoding='utf-8') as f:
    id_list = json.load(f)

split_n = len(id_list) // 4

with open(path, 'r',  encoding='utf-8') as f:
    blog_total = json.load(f)

blog = [blog_total[i] for i in id_list]

for i in range(split_n*1, split_n*2):
    data = blog[i]
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": instruction},
                {"role": "user", "content": data['content']},
            ],
            
            temperature=1,
            max_tokens=1000,
            top_p =1,
        )

        refined_content = response.choices[0].message.content
        data['content'] = refined_content
        new_content.append(data)
        print(f"{i}번째 완료")

        if i % 100 == 0:
            with open(f"./merged_results/blog_summarized_1_{i}.json", 'w', encoding='utf-8') as f:
                json.dump(new_content, f, ensure_ascii=False, indent=4)

    except Exception as e:
        print(f"{i}번째 처리 중 오류 발생: {e}")
        # 실패한 경우도 로그 남겨두고 넘어가도록
        data['content'] = f"[요약 실패: {e}]"
        new_content.append(data)
        continue

with open('./merged_results/blog_summarized_final.json', 'w', encoding='utf-8') as f:
    json.dump(new_content, f, ensure_ascii=False, indent=4)