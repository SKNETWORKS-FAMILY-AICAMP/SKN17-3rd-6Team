import os, json, re, unicodedata, glob
import pdfplumber
from collections import OrderedDict

# ---------- 텍스트 정리 ----------
def norm(s: str) -> str:
    if s is None:
        return ""
    s = unicodedata.normalize("NFKC", str(s))
    s = s.replace("\xa0", " ")
    s = re.sub(r"[\r\n\t]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip(" -:·|")
    return s.strip()

def key_id(s: str) -> str:
    return re.sub(r"[^0-9a-z가-힣]+", "", norm(s).lower())

def dedup_keep_order(seq):
    seen = set()
    out = []
    for x in seq:
        k = key_id(x)
        if k and k not in seen:
            seen.add(k)
            out.append(norm(x))
    return out

# ---------- 표 파서 ("현상 | 예상 원인" 양식만) ----------
def parse_table_target(page):
    result = OrderedDict()
    current_ph = None
    tables = page.extract_tables() or []

    for table in tables:
        if not table or len(table[0]) < 2:
            continue

        header = [norm(x) for x in table[0]]
        if not ("현상" in header[0] and "원인" in header[1]):
            continue  # 다른 양식이면 skip

        for row in table[1:]:
            if not row or len(row) < 2:
                continue
            ph, cause = norm(row[0]), norm(row[1])

            if not ph and not cause:
                continue
            if ph:
                current_ph = ph
            ph = current_ph

            if not ph or not cause:
                continue

            result.setdefault(ph, [])
            if cause not in result[ph]:
                result[ph].append(cause)
    return result

# ---------- 카테고리 변환 ----------
def transform_categories(parts):
    cats = parts[3:]  # 앞 3개 무시
    cats = dedup_keep_order(cats)
    if len(cats) == 2:
        return [cats[0], "없음", "고장진단"]
    elif len(cats) == 3:
        return [cats[0], cats[1], "고장진단"]
    elif len(cats) >= 4:
        return [cats[0], cats[1] + " " + cats[2], "고장진단"]
    else:
        return []

# ---------- 병합 ----------
def merge_faultmaps(dst: dict, src: dict):
    for ph, causes in src.items():
        dst.setdefault(ph, [])
        for c in causes:
            if c not in dst[ph]:
                dst[ph].append(c)

def insert_nested_merge(root: dict, keys: list[str], value: dict):
    cur = root
    for k in keys[:-1]:
        cur = cur.setdefault(k, {})
    leaf = keys[-1]
    if leaf not in cur:
        cur[leaf] = value
    else:
        if isinstance(cur[leaf], dict):
            merge_faultmaps(cur[leaf], value)
        else:
            cur[leaf] = value

# ---------- 단일 PDF 처리 ----------
def parse_pdf(path):
    parent_folder = os.path.basename(os.path.dirname(os.path.abspath(path))) or "ROOT"
    result = {parent_folder: OrderedDict()}
    found_valid = False

    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            # header line에서 카테고리 추출
            text = page.extract_text() or ""
            header_line = None
            for line in text.split("\n"):
                if ">" in line:
                    header_line = line
                    break
            if not header_line:
                continue  # > 없는 PDF는 skip

            parts = [norm(p) for p in header_line.split(">")]
            cats = transform_categories(parts)
            if not cats:
                continue

            # 표 데이터 추출
            table_data = parse_table_target(page)
            if table_data:
                insert_nested_merge(result[parent_folder], cats, table_data)
                found_valid = True

    return result if found_valid else None

# ---------- 여러 PDF 처리 ----------
def parse_pdfs(folder_path, output_json):
    final_result = OrderedDict()
    pdf_files = glob.glob(os.path.join(folder_path, "**/*.pdf"), recursive=True)

    for pdf_file in pdf_files:
        data = parse_pdf(pdf_file)
        if not data:  # 대상 양식 아니면 skip
            continue
        for folder, content in data.items():
            if folder not in final_result:
                final_result[folder] = content
            else:
                for k, v in content.items():
                    if k not in final_result[folder]:
                        final_result[folder][k] = v
                    else:
                        insert_nested_merge(final_result[folder], [k], v)

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(final_result, f, ensure_ascii=False, indent=2)

    print("총 PDF 개수:", len(pdf_files))
    print("대상 양식 추출 개수:", sum(1 for pdf in pdf_files if parse_pdf(pdf)))
    print("저장 완료:", output_json)


if __name__ == "__main__":
    folder_path = "./EV5(OV1k)_160KW"
    output_file = "1.json"                    # 항상 같은 이름
    parse_pdfs(folder_path, output_file)