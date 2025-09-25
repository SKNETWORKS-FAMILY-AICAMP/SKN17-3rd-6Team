import os
import json

blog_path = './blog_results'
kin_path = './kin_results'

blog_files = [f for f in os.listdir(blog_path) if f.endswith('.json')]
print("Blog JSON files:", blog_files)

kin_files = [f for f in os.listdir(kin_path) if f.endswith('.json')]
print("Kin JSON files:", kin_files)

blog_merged = []
kin_merged = []
blog_sources = set()
kin_sources = set()

for blog_file in blog_files:
    path = os.path.join(blog_path, blog_file)
    with open(path, 'r', encoding='utf-8') as f:
        blog = json.load(f)
    for data in blog:
        source = data.get('출처')
        if source not in blog_sources:
            blog_merged.append(data)
            blog_sources.add(source)

for kin_file in kin_files:
    path = os.path.join(kin_path, kin_file)
    with open(path, 'r', encoding='utf-8') as f:
        kin = json.load(f)
    for data in kin:
        source = data.get('출처')
        if source not in kin_sources:
            kin_merged.append(data)
            kin_sources.add(source)

with open('./merged_results/blog_merged.json', 'w', encoding='utf-8') as f:
    json.dump(blog_merged, f, ensure_ascii=False, indent=4)

with open('./merged_results/kin_merged.json', 'w', encoding='utf-8') as f:
    json.dump(kin_merged, f, ensure_ascii=False, indent=4)

print("Merged blog items saved:", len(blog_merged))
print("Merged kin items saved:", len(kin_merged))


#----------- 블로그 요약 merge ---------
with open("blog_summarized_final_1.json", "r", encoding="utf-8") as f1:
    list1 = json.load(f1)   

with open("blog_summarized_final.json", "r", encoding="utf-8") as f2:
    list2 = json.load(f2)  

with open("blog_summarized_final_3.json", "r", encoding="utf-8") as f3:
    list3 = json.load(f3)  

with open("blog_summarized_final_4.json", "r", encoding="utf-8") as f4:
    list4 = json.load(f4)  


## 두 리스트 합치기
merged_blog_data = list1 + list2 + list3 + list4


# 새로운 파일로 저장
with open("merged_blog_summary.json", "w", encoding="utf-8") as out:
    json.dump(merged_blog_data, out, ensure_ascii=False, indent=2)

print(f"합친 개수: {len(merged_blog_data)}")


# 파일로드 
with open("merged_blog_summary.json", "r", encoding="utf-8") as f1:
    merged_blog_data = json.load(f1)

#-------------- 파일 4등분 ---------------
data_sorted = sorted(merged_blog_data, key=lambda x: x["title"])

# 4등분 크기 계산
chunk_size = len(data_sorted) / 4


# 4개의 파일로 저장
for i in range(4):
    if i < 4 :
        start = int(i * chunk_size)
        end = int(start + chunk_size)
        chunk = data_sorted[start:end]

    elif i == 4 :
        start = int(i * chunk_size)
        end = len(data_sorted)
        chunk = data_sorted[start:end]

    filename = f"merged_blog_summary_{i+1}.json"
    with open(filename, "w", encoding="utf-8") as out:
        json.dump(chunk, out, ensure_ascii=False, indent=2)

    print(f"저장된 파일 : {filename} | 시작: {start} | 종료: {end} | 길이: {len(chunk)}")

