from bs4 import BeautifulSoup

from bs4 import BeautifulSoup
from typing import List, Dict

import json, os


def parse_course_table(html_snippet: str) -> List[Dict[str, str]]:
    soup = BeautifulSoup(html_snippet, "html.parser")
    table = soup.find("table", {"id": "table1"})
    
    if not table:
        raise ValueError("No table with id='table1' found in HTML.")

    rows = table.find_all("tr")
    courses = []

    for row in rows[1:]:  # Skip the header
        cells = row.find_all("td")
        if not cells or len(cells) < 10:
            continue  # Skip rows with insufficient data

        try:
            course_info = {
                "title": cells[0].get_text(strip=True),
                "subject": cells[1].get_text(strip=True),
                "course_number": cells[2].get_text(strip=True),
                "section": cells[3].get_text(strip=True),
                "credit_hours": cells[4].get_text(strip=True),
                "crn": cells[5].get_text(strip=True),
                "term": cells[6].get_text(strip=True),
                "meeting_time": cells[7].get_text(strip=True),
                "campus": cells[8].get_text(strip=True),
                "status": cells[9].get_text(strip=True),
                "attribute": cells[10].get_text(strip=True) if len(cells) > 10 else "",
            }
            courses.append(course_info)
        except Exception as e:
            print(f"Error parsing row: {e}")
            continue

    return courses


with open("/Users/lee/School/Projects/Course-Advisor/utils/html_snippet.html", "r", encoding="utf-8") as f:
    html_snippet = f.read()


parsed_courses = parse_course_table(html_snippet)

filename = "courses.json"

# 1. 기존 JSON 파일이 존재하면 불러오기
if os.path.exists(filename):
    with open(filename, "r", encoding="utf-8") as f:
        existing_courses = json.load(f)
else:
    existing_courses = []

# 2. 새로운 데이터를 기존 리스트에 추가
existing_courses.extend(parsed_courses)

# 3. 덮어쓰기 저장
with open(filename, "w", encoding="utf-8") as f:
    json.dump(existing_courses, f, ensure_ascii=False, indent=2)