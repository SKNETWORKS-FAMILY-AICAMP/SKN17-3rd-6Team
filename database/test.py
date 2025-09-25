import os
import numpy as np
import json
from openai import OpenAI
from dotenv import load_dotenv

# index_list = np.random.randint(low=0, high=100000, size=5000, dtype=int)

# with open("./merged_results/index_list.json", "w", encoding="utf-8") as f:
#     json.dump(index_list.tolist(), f, ensure_ascii=False, indent=4)

with open('./merged_results/index_list.json', 'r', encoding='utf-8') as f:
    id = json.load(f)
    print(id)