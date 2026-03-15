import requests
import os
import time
from bs4 import BeautifulSoup

EROWID_BASE_URI = 'https://erowid.org/experiences/exp.php'
EXP_COUNT = 10000
DELAY = 1.5  # seconds between requests

def _sanitize_substance_for_path(substance):
    substance = substance.strip().lower()
    substance = substance.replace('/', '+').replace('\\', '+')
    return substance if substance else 'unknown'

def extract_experience_text(text):
    try:
        begin_delimiter = '<!-- Start Body -->'
        begin = text.index(begin_delimiter) + len(begin_delimiter)
        end = text.index('<!-- End Body -->')
        return text[begin:end].strip()
    except ValueError:
        return ''

session = requests.Session()
session.headers.update({'User-Agent': 'Mozilla/5.0'})

for index in range(1, EXP_COUNT):
    try:
        response = session.get(EROWID_BASE_URI, params={'ID': index}, timeout=10)
        response.raise_for_status()

        responseText = response.text
        experienceText = extract_experience_text(responseText)

        if not experienceText:
            print(f"[{index}] empty experience, skipping")
            continue

        soup = BeautifulSoup(responseText, "html.parser")  # html5lib was overkill here
        
        substance_tag = soup.find('div', {'class': 'substance'})
        if not substance_tag:
            print(f"[{index}] no substance tag, skipping")
            continue

        drug = substance_tag.getText().strip().lower()
        safe_drug = _sanitize_substance_for_path(drug)

        folder_path = f'./experiences/{safe_drug}'
        os.makedirs(folder_path, exist_ok=True)  # no need for manual exists check

        file_path = f'{folder_path}/{index}.txt'
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(experienceText)

        print(f"[{index}] {drug} → saved")

    except requests.RequestException as e:
        print(f"[{index}] network error: {e}")
    except Exception as e:
        print(f"[{index}] error: {e}")

    time.sleep(DELAY)