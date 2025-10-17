# ==============================================================================
#  IMPORTS & SETUP
# ==============================================================================
import streamlit as st
from pathlib import Path
import re
from PIL import Image
import pytesseract
import pandas as pd
import requests
from io import BytesIO, StringIO
import csv
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
import time
import fitz  # PyMuPDF
import tempfile
import os
import logging
import json
from datetime import datetime

# Imports from the new PDF processing module
import pdfplumber
from urllib.parse import urlparse, urlunparse
import urllib.request
from typing import Tuple, List, Any, Dict, Optional
from random import uniform

# ==============================================================================
#  CONFIGURATION
# ==============================================================================

# ⛔ NOTE: This now correctly reads from st.secrets or environment variables.
try:
    MISTRAL_API_KEY = st.secrets.get("MISTRAL_API_KEY", os.getenv("MISTRAL_API_KEY"))
except (AttributeError, KeyError):
    MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

MISTRAL_MODEL = "mistral-large-latest"
MISTRAL_ENDPOINT = "https://api.mistral.ai/v1/chat/completions"


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Skills configuration for analysis
SKILLS_TO_ASSESS = [
    'JavaScript', 'Python', 'Node', 'React', 'Java', 'Springboot', 'DSA',
    'AI', 'ML', 'PHP', '.Net', 'Testing', 'AWS', 'Django', 'PowerBI', 'Tableau'
]
SKILL_COLUMNS = [f'{skill}_Probability' for skill in SKILLS_TO_ASSESS]

# ==============================================================================
#  SESSION STATE INITIALIZATION
# ==============================================================================
if 'comprehensive_results' not in st.session_state:
    st.session_state.comprehensive_results = []
if 'analysis_running' not in st.session_state:
    st.session_state.analysis_running = False
if 'last_analysis_mode' not in st.session_state:
    st.session_state.last_analysis_mode = ""
if 'shortlisting_mode' not in st.session_state:
    st.session_state.shortlisting_mode = "Probability Wise (Default)"


# ==============================================================================
#  UTILS — SANITIZATION & SAFE OPS
# ==============================================================================

CONTROL_CHARS_RE = re.compile(r'[\x00-\x1f\x7f-\x9f]')

def safe_str(x: Any) -> str:
    """Convert any value to a clean string without control characters."""
    try:
        s = str(x)
    except Exception:
        s = ""
    return CONTROL_CHARS_RE.sub('', s)

def coerce_probability_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure all probability columns are valid ints (Arrow-friendly)."""
    for col in SKILL_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
    return df

# ##############################################################################
#  START OF THE FIX
# ##############################################################################
# I have replaced the original `sanitize_json_text` function with this more
# robust version. The main improvement is its ability to find and isolate the
# JSON object, removing any conversational text or markdown around it.

def sanitize_json_text(text: str) -> str:
    """
    Cleans model JSON by:
    1. Removing markdown fences and any surrounding conversational text.
    2. Isolating the core JSON object or array.
    3. Removing control characters and escaping invalid backslashes.
    4. Attempting to fix truncation errors by closing unterminated strings
       and balancing both curly braces and square brackets.
    """
    if not isinstance(text, str):
        text = str(text)
    text = text.strip()

    # Step 1: Find the start of the JSON (can be '{' or '[')
    first_brace = text.find('{')
    first_bracket = text.find('[')
    
    start_pos = -1
    if first_brace == -1 and first_bracket == -1:
        logger.warning("Could not find a JSON start '{' or '[' in the AI response.")
        return '{"error": "No valid JSON object or array found in response"}'
    
    if first_brace != -1 and first_bracket != -1:
        start_pos = min(first_brace, first_bracket)
    elif first_brace != -1:
        start_pos = first_brace
    else: # first_bracket != -1
        start_pos = first_bracket
        
    # Find the corresponding last brace or bracket
    last_brace = text.rfind('}')
    last_bracket = text.rfind(']')
    end_pos = max(last_brace, last_bracket)
    
    if end_pos < start_pos:
        logger.warning("Could not find a valid JSON structure {{...}} or [...] in the AI response.")
        return '{"error": "No valid JSON structure found in response"}'

    # Isolate the actual JSON content
    text = text[start_pos : end_pos + 1]

    # Step 2: Remove control characters.
    text = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", text)

    # Step 3: Escape single backslashes that aren't part of a valid escape sequence.
    text = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', text)
    
    # Step 4: Robustly fix truncation errors.
    # A. Close an unterminated string if it's likely the cause of an error.
    # An odd number of quotes usually means one is left open at the end.
    if text.count('"') % 2 != 0:
        text += '"'

    # B. Balance both braces and brackets to fix incomplete structures.
    open_braces, close_braces = text.count("{"), text.count("}")
    if open_braces > close_braces:
        text += "}" * (open_braces - close_braces)

    open_brackets, close_brackets = text.count("["), text.count("]")
    if open_brackets > close_brackets:
        text += "]" * (open_brackets - close_brackets)

    return text

def relaxed_json_loads(text: str) -> dict:
    """
    Parses JSON robustly by first calling a powerful sanitization function.
    """
    sanitized_text = sanitize_json_text(text)
    try:
        return json.loads(sanitized_text)
    except json.JSONDecodeError as e:
        # If parsing fails even after cleaning, log the error and the problematic text.
        # The exception will be caught by the calling worker function.
        logger.error(f"JSON parsing failed even after sanitization: {e}")
        logger.debug(f"--- Sanitized text that failed parsing ---\n{sanitized_text}\n-----------------------------")
        raise  # Re-raise the exception

# ##############################################################################
#  END OF THE FIX
# ##############################################################################

def safe_join_texts(items: Any) -> str:
    """
    Join a list-like into a string; if not list-like, stringify safely.
    """
    if isinstance(items, list):
        return " ".join(safe_str(x) for x in items)
    return safe_str(items)

def is_present_str(s: Any) -> bool:
    """Check if a value indicates current/present in a forgiving way."""
    if s is None:
        return False
    return 'present' in safe_str(s).lower() or 'current' in safe_str(s).lower()

# ==============================================================================
#  NEW: ROBUST PDF DOWNLOADING & EXTRACTION LOGIC
# ==============================================================================

def download_pdf(pdf_url: str, output_pdf_path: str) -> Tuple[bool, str]:
    """
    Downloads a PDF from a URL, handling Google Docs, Google Drive (with a robust
    method to bypass rate-limiting), S3, and other URLs.
    """
    try:
        parsed_url = urlparse(pdf_url)
        is_google_doc = "docs.google.com" in parsed_url.netloc and "/document/" in parsed_url.path
        is_google_drive = "drive.google.com" in parsed_url.netloc and ("open" in parsed_url.path or "file" in parsed_url.path or "/d/" in parsed_url.path)
        is_s3 = ".s3.amazonaws.com" in parsed_url.netloc

        if is_google_doc:
            doc_id = parsed_url.path.split('/d/')[1].split('/')[0]
            export_url_parts = list(parsed_url)
            export_url_parts[2] = f"/document/d/{doc_id}/export"
            export_url_parts[4] = "format=pdf"
            pdf_url = urlunparse(export_url_parts)
            logger.info(f"Google Docs URL detected, using export URL: {pdf_url}")
            urllib.request.urlretrieve(pdf_url, output_pdf_path)

        elif is_google_drive:
            file_id = None
            if "id=" in parsed_url.query:
                file_id = parsed_url.query.split("id=")[1].split("&")[0]
            elif "/d/" in parsed_url.path:
                file_id = parsed_url.path.split("/d/")[1].split("/")[0]

            if not file_id:
                raise ValueError("Could not extract file ID from Google Drive URL")
            
            logger.info(f"Google Drive URL detected for file ID: {file_id}. Using direct request method.")

            # Use a session to handle cookies properly
            session = requests.Session()
            URL = "https://docs.google.com/uc?export=download"
            
            # Initial request to get the download confirmation token
            response = session.get(URL, params={'id': file_id}, stream=True, timeout=30)
            
            # This token is sometimes needed to confirm the download (e.g., for large files)
            token = None
            for key, value in response.cookies.items():
                if key.startswith('download_warning'):
                    token = value
                    break

            # If a token was found, a second request is needed with the token
            if token:
                params = {'id': file_id, 'confirm': token}
                response = session.get(URL, params=params, stream=True, timeout=30)

            # Now, write the content to the file
            response.raise_for_status()
            with open(output_pdf_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

        else:  # Handles S3 and other direct URLs
            logger.info(f"Standard or S3 URL detected: {pdf_url}")
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            response = requests.get(pdf_url, headers=headers, stream=True, timeout=30, allow_redirects=True)
            response.raise_for_status()
            with open(output_pdf_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

        # Validation checks
        if not os.path.exists(output_pdf_path) or os.path.getsize(output_pdf_path) == 0:
            raise ValueError("Downloaded PDF is empty or does not exist.")
        with open(output_pdf_path, 'rb') as f:
            first_bytes = f.read(5)
            if not first_bytes.startswith(b'%PDF'):
                raise ValueError(f"Downloaded file is not a valid PDF. Header starts with: {first_bytes}")

        logger.info(f"PDF downloaded successfully to: {output_pdf_path}")
        return True, f"PDF downloaded to {output_pdf_path}"

    except requests.exceptions.RequestException as e:
        error_msg = f"Download failed: {str(e)}"
        logger.error(f"Failed to download PDF from {pdf_url}: {error_msg}")
        return False, error_msg
    except ValueError as e:
        error_msg = f"Invalid PDF file or URL: {str(e)}"
        logger.error(f"Error validating {output_pdf_path}: {error_msg}")
        return False, error_msg
    except Exception as e:
        error_msg = f"An unexpected error occurred during download: {str(e)}"
        logger.error(f"Unexpected error for {pdf_url}: {error_msg}")
        return False, error_msg


def extract_urls_from_pdf_annotations(pdf_path: str) -> List[str]:
    """Extracts hyperlinks from PDF annotations using PyMuPDF."""
    urls = []
    try:
        doc = fitz.open(pdf_path)
        for page in doc:
            links = page.get_links()
            for link in links:
                if link.get("uri"):
                    urls.append(link["uri"])
        doc.close()
    except Exception as e:
        logger.warning(f"Error extracting hyperlinks from {pdf_path}: {e}")
    return urls

def extract_text_and_urls_from_pdf(pdf_path: str) -> Tuple[str, List[str]]:
    """
    Extracts text from a PDF using a robust fallback strategy (pdfplumber -> fitz -> OCR)
    and also extracts embedded hyperlinks.
    Returns (extracted_text, extracted_urls).
    """
    text_content = ""
    extracted_urls = []

    # Strategy 1: pdfplumber
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_content += page_text + "\n"
        logger.info(f"Extraction with pdfplumber successful for {pdf_path}")
    except Exception as e:
        logger.warning(f"pdfplumber failed for {pdf_path}: {e}. Trying next method.")

    # Strategy 2: fitz
    if not text_content.strip():
        try:
            doc = fitz.open(pdf_path)
            for page in doc:
                text_content += page.get_text() + "\n"
            doc.close()
            logger.info(f"Extraction with fitz successful for {pdf_path}")
        except Exception as e:
            logger.warning(f"fitz failed for {pdf_path}: {e}. Trying OCR.")

    # Strategy 3: OCR with Tesseract
    if not text_content.strip() and st.session_state.get('enable_ocr', True):
        logger.info(f"Falling back to OCR for {pdf_path}")
        try:
            doc = fitz.open(pdf_path)
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                pix = page.get_pixmap(dpi=200)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                text_content += pytesseract.image_to_string(img) + "\n"
            doc.close()
            logger.info(f"OCR extraction completed for {pdf_path}")
        except Exception as e:
            logger.error(f"OCR extraction failed for {pdf_path}: {e}")

    # Extract clickable hyperlinks
    extracted_urls = extract_urls_from_pdf_annotations(pdf_path)

    return text_content, list(set(extracted_urls))

# ==============================================================================
#  CORE AI & API FUNCTIONS (Mistral)
# ==============================================================================

def analyze_text_with_mistral(prompt: str) -> str:
    """
    Calls Mistral La Plateforme chat completions API with the given prompt.
    Returns the model's text content (string). If any error, returns a JSON error string.
    Includes retry with exponential backoff on 429 and timeouts.
    """
    if not MISTRAL_API_KEY:
        return json.dumps({"error": "Missing Mistral API key."})

    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": MISTRAL_MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are an expert resume parser. Always return STRICT JSON only—"
                    "no markdown fences, no commentary. Fill missing values with empty strings "
                    '"" or [] where appropriate.'
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
        "max_tokens": 4096,
    }

    attempts = 5
    backoff_base = 0.8

    for attempt in range(1, attempts + 1):
        try:
            resp = requests.post(MISTRAL_ENDPOINT, headers=headers, json=payload, timeout=120)
            if resp.status_code == 200:
                data = resp.json()
                content = (
                    data.get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "")
                )
                return content.strip()
            elif resp.status_code in (408, 429, 500, 502, 503, 504):
                # backoff and retry
                wait_s = (backoff_base ** attempt) + uniform(0, 0.6)
                logger.warning(f"Mistral transient error {resp.status_code}. Retrying in {wait_s:.2f}s...")
                time.sleep(wait_s)
                continue
            else:
                logger.error(f"Mistral API Error: {resp.status_code} - {resp.text}")
                return json.dumps({"error": f"API Error Status {resp.status_code}: {resp.text}"})
        except requests.exceptions.RequestException as e:
            wait_s = (backoff_base ** attempt) + uniform(0, 0.6)
            logger.warning(f"RequestException (attempt {attempt}/{attempts}): {e}. Backing off {wait_s:.2f}s")
            time.sleep(wait_s)
            continue

    return json.dumps({"error": "Mistral request failed after retries."})

# ==============================================================================
#  HELPER & FORMATTING FUNCTIONS
# ==============================================================================

def get_github_repo_count(username):
    """Fetches GitHub public repo count and always returns it as a string."""
    if not username:
        return ""
    try:
        response = requests.get(f'https://api.github.com/users/{username}', timeout=10)
        if response.status_code == 200:
            return str(response.json().get('public_repos', ""))
        else:
            return ""
    except requests.exceptions.RequestException:
        return ""

def extract_github_username(url):
    match = re.search(r'github\.com/([^/]+)', url)
    return match.group(1) if match else None

def format_mobile_number(raw_number: str) -> str:
    """
    Cleans and formats a mobile number string to a fixed 10-digit format.
    Removes country codes like +91, 91, 0 and non-digit characters.
    Returns a 10-digit string or an empty string if the format is invalid.
    """
    raw_number = safe_str(raw_number)
    if not raw_number.strip():
        return ""
    digits_only = re.sub(r'\D', '', raw_number)
    if len(digits_only) == 12 and digits_only.startswith('91'):
        return digits_only[2:]
    elif len(digits_only) == 11 and digits_only.startswith('0'):
        return digits_only[1:]
    elif len(digits_only) == 10:
        return digits_only
    else:
        return ""

def sort_links(links_list):
    linkedin_link, github_link = "", ""
    other_links = []
    for link in links_list:
        link = safe_str(link)
        if 'linkedin.com/in/' in link and not linkedin_link:
            linkedin_link = link
        elif 'github.com' in link and not github_link:
            github_link = link
        else:
            other_links.append(link)
    if github_link:
        username = extract_github_username(github_link)
        if username:
            github_link = f"https://github.com/{username}"
    return linkedin_link, github_link, other_links

def get_latest_experience(exp_list: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(exp_list, list) or not exp_list:
        return None
    # Prefer current; else first
    for exp in exp_list:
        if not isinstance(exp, dict):
            continue
        end_date_str = safe_str(exp.get('endDate', ''))
        if is_present_str(end_date_str):
            return exp
    # fallback to first dict-like
    for exp in exp_list:
        if isinstance(exp, dict):
            return exp
    return None

def get_highest_education_institute(edu_data):
    """Determines the highest education institute from the parsed education data."""
    if not isinstance(edu_data, dict):
        return ""
    if edu_data.get('masters_doctorate') and safe_str(edu_data['masters_doctorate'].get('collegeName')):
        return safe_str(edu_data['masters_doctorate']['collegeName'])
    if edu_data.get('bachelors') and safe_str(edu_data['bachelors'].get('collegeName')):
        return safe_str(edu_data['bachelors']['collegeName'])
    if edu_data.get('diploma') and safe_str(edu_data['diploma'].get('collegeName')):
        return safe_str(edu_data['diploma']['collegeName'])
    if edu_data.get('intermediate_puc_12th') and safe_str(edu_data['intermediate_puc_12th'].get('collegeName')):
        return safe_str(edu_data['intermediate_puc_12th']['collegeName'])
    if edu_data.get('ssc_10th') and safe_str(edu_data['ssc_10th'].get('collegeName')):
        return safe_str(edu_data['ssc_10th']['collegeName'])
    return ""

def check_tesseract_installation():
    try:
        pytesseract.get_tesseract_version()
        return True
    except pytesseract.TesseractNotFoundError:
        return False

def format_projects(projects):
    """
    Normalize projects list into "Title - Details" lines.
    Accept strings or dicts; ignore unsupported types gracefully.
    """
    formatted_list = []
    if not isinstance(projects, list):
        return ""
    for p in projects:
        if isinstance(p, str):
            parts = p.split(" - ", 1)
            if len(parts) == 2:
                formatted_list.append(f"**{parts[0].strip()}** - {parts[1].strip()}")
            else:
                formatted_list.append(safe_str(p).strip())
        elif isinstance(p, dict):
            title = safe_str(p.get('title', '')).strip()
            details = safe_str(p.get('details', '')).strip()
            if title and details:
                formatted_list.append(f"**{title}** - {details}")
            elif title:
                formatted_list.append(f"**{title}**")
            elif details:
                formatted_list.append(details)
        else:
            formatted_list.append(safe_str(p).strip())
    return "\n".join(formatted_list)

def calculate_skill_probabilities(data):
    """
    Robust scorer that tolerates lists/dicts, missing keys, and non-strings.
    """
    scores = {column: 0 for column in SKILL_COLUMNS}
    if not isinstance(data, dict):
        return scores

    # Collect text pools safely
    skills_list = data.get('skills', []) if isinstance(data.get('skills', []), list) else []
    skills_text = " ".join(safe_str(x) for x in skills_list).lower()

    projects = data.get('projects', [])
    if isinstance(projects, list):
        projects_text = " ".join(safe_str(x) for x in projects).lower()
    else:
        projects_text = safe_str(projects).lower()

    certifications = data.get('certifications', [])
    certs_text = " ".join(safe_str(x) for x in certifications).lower() if isinstance(certifications, list) else safe_str(certifications).lower()

    experience_text = ""
    exp_list = data.get('experience', [])
    if isinstance(exp_list, list):
        exp_chunks = []
        for exp in exp_list:
            if isinstance(exp, dict):
                desc = exp.get('description', '')
                # description may be list/string
                if isinstance(desc, list):
                    exp_chunks.append(" ".join(safe_str(x) for x in desc))
                else:
                    exp_chunks.append(safe_str(desc))
                # also consider job title
                exp_chunks.append(safe_str(exp.get('jobTitle', '')))
                # and company
                exp_chunks.append(safe_str(exp.get('companyName', '')))
        experience_text = " ".join(exp_chunks).lower()
    else:
        experience_text = safe_str(exp_list).lower()

    education_text = safe_str(data.get('education', {})).lower()
    foundational_text = f"{skills_text} {projects_text} {experience_text} {education_text}"

    for skill in SKILLS_TO_ASSESS:
        score = 0
        skill_lower = skill.lower()
        # special handling for ".Net"
        if skill == '.Net':
            # literal ".net" word boundary
            skill_pattern = r'(?i)(?<![a-zA-Z0-9])\.net(?![a-zA-Z0-9])'
        else:
            skill_pattern = r'\b' + re.escape(skill_lower) + r'\b'

        if re.search(skill_pattern, foundational_text):
            score += 10
        if re.search(skill_pattern, experience_text):
            score += 30
        if re.search(skill_pattern, projects_text):
            score += 20
        if re.search(skill_pattern, certs_text):
            score += 20
        scores[f'{skill}_Probability'] = score
    return scores

# ==============================================================================
#  MAIN WORKER FUNCTIONS
# ==============================================================================

def process_resume_for_shortlisting(row, resume_index, user_requirements):
    """
    UPDATED WORKER: Implements "Focused Analysis".
    The AI first determines the user query's intent (e.g., certification, experience)
    and then scores the resume based on that specific focus.
    """
    user_id = row['user_id']
    resume_link = row['Resume link']

    result = {
        'User ID': user_id,
        'Resume Link': resume_link,
        'Overall Probability': 0,
        'Projects Probability': 0,
        'Skills Probability': 0,
        'Experience Probability': 0,
        'Other Probability': 0,
        'Overall Remarks': "Error processing",
        'Projects Remarks': "",
        'Skills Remarks': "",
        'Experience Remarks': "",
        'Other Remarks': ""
    }

    temp_pdf_path = None
    try:
        logger.info(f"Shortlisting resume #{resume_index + 1} for user {user_id} with Focused Analysis.")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            temp_pdf_path = tmp.name
        download_success, download_msg = download_pdf(resume_link, temp_pdf_path)
        if not download_success:
            raise ValueError(f"Download Error: {download_msg}")
        resume_text, _ = extract_text_and_urls_from_pdf(temp_pdf_path)
        if not resume_text.strip():
            raise ValueError("Could not extract any text from the PDF.")

        # =================================================================================
        # NEW: "Focused Analysis" Prompt Logic
        # This new prompt is much more intelligent. It instructs the AI to first analyze
        # the user's requirement and then adapt its scoring logic accordingly. This directly
        # addresses the issue where a specific query like "Java Certification" was unfairly
        # penalized by low scores in irrelevant categories like Projects or Experience.
        # =================================================================================
        prompt = f"""
You are an intelligent and focused technical resume evaluator. Your goal is to precisely evaluate a resume based on the user's specific criteria.

**Your TWO-STEP evaluation process:**

**STEP 1: ANALYZE THE USER'S INTENT**
First, read the "Required Criteria" below and determine its primary focus. Is the user asking for:
- A specific **CERTIFICATION or EDUCATION**? (e.g., "seeking for Programming in JAVA Certification", "Must have a PMP certification", "Requires an MBA").
- Specific **WORK EXPERIENCE**? (e.g., "5+ years as a product manager", "experience with enterprise-level software").
- A list of **SKILLS or TECHNOLOGIES**? (e.g., "React, Node, MongoDB").
- A **HOLISTIC** role? (A full job description with a mix of skills, experience, and education).

**STEP 2: PERFORM A FOCUSED EVALUATION BASED ON THE INTENT**
After determining the intent, you will evaluate the resume across four areas (Projects, Skills, Experience, Other). HOWEVER, your final "Overall Probability" must be calculated based on the primary intent you identified:

- **If the intent is CERTIFICATION/EDUCATION:**
  - The "Overall Probability" should ALMOST ENTIRELY be based on whether the specific certification/degree is found in the 'Other' or 'Education' sections.
  - If the exact item is found, the **Overall Probability must be 90 or higher**.
  - Weakness in other areas (Projects, Skills, Experience) should be noted in their remarks but have minimal impact on the final overall score.
  - *Example "Overall Remarks" for a match: "Excellent fit. The resume explicitly lists the required Java Certification."*

- **If the intent is EXPERIENCE:**
  - The "Overall Probability" must primarily reflect the relevance of the candidate's work history described in the 'Experience' section.

- **If the intent is SKILLS:**
  - The "Overall Probability" should be a balanced score based on where the skills are mentioned (Projects, Skills section, Experience).

- **If the intent is HOLISTIC:**
  - ONLY in this case, use a weighted average to calculate the "Overall Probability": Projects (40%), Skills (30%), Experience (20%), Other (10%).

**YOUR TASK:**
Evaluate the resume based on the criteria using the focused two-step method described above. You must provide scores and remarks for all four areas, but ensure your **Overall Probability** and **Overall Remarks** strictly follow the focused intent logic.

Return your answer as **pure JSON only**, with no markdown or commentary.

JSON Structure:
{{
  "projects_probability": "integer (0–100)",
  "projects_remarks": "string",
  "skills_probability": "integer (0–100)",
  "skills_remarks": "string",
  "experience_probability": "integer (0–100)",
  "experience_remarks": "string",
  "other_probability": "integer (0–100)",
  "other_remarks": "string",
  "overall_probability": "integer (0–100)",
  "overall_remarks": "string"
}}

Required Criteria:
---
{user_requirements}
---

Resume Text:
---
{resume_text}
---
"""
        mistral_response_text = analyze_text_with_mistral(prompt)

        data = {}
        try:
            data = relaxed_json_loads(mistral_response_text)
            if not isinstance(data, dict):
                raise ValueError(f"AI returned data that is not a JSON object. Type: {type(data)}")
            if "error" in data:
                raise ValueError(data["error"])
        except Exception as e:
            raise ValueError(f"Error parsing AI response: {e}")

        result.update({
            'Overall Probability': data.get('overall_probability', 0),
            'Projects Probability': data.get('projects_probability', 0),
            'Skills Probability': data.get('skills_probability', 0),
            'Experience Probability': data.get('experience_probability', 0),
            'Other Probability': data.get('other_probability', 0),
            'Overall Remarks': data.get('overall_remarks', 'N/A'),
            'Projects Remarks': data.get('projects_remarks', 'N/A'),
            'Skills Remarks': data.get('skills_remarks', 'N/A'),
            'Experience Remarks': data.get('experience_remarks', 'N/A'),
            'Other Remarks': data.get('other_remarks', 'N/A'),
        })

    except Exception as e:
        logger.error(f"Failed shortlisting {user_id} ({resume_link}): {e}")
        error_msg = f"Error: {e}"
        result.update({
            'Overall Remarks': error_msg,
            'Projects Remarks': error_msg,
            'Skills Remarks': error_msg,
            'Experience Remarks': error_msg,
            'Other Remarks': error_msg,
        })

    finally:
        if temp_pdf_path and os.path.exists(temp_pdf_path):
            try:
                os.remove(temp_pdf_path)
            except OSError as e:
                logger.error(f"Error removing temporary file {temp_pdf_path}: {e}")

    return result


def process_resume_comprehensively(row, resume_index, analysis_type):
    """Worker for comprehensive data extraction."""
    user_id = row['user_id']
    resume_link = row['Resume link']
    
    base_columns = [
        'User ID', 'Resume Link', 'Full Name', 'Mobile Number', 'Email ID', 
        'LinkedIn Link', 'GitHub Link', 'Other Links', 'Skills', 'Projects', 
        'Latest Experience Company Name', 'Latest Experience Job Title', 
        'Latest Experience Start Date', 'Latest Experience End Date', 'Currently Working? (Yes/No)',
        'Years of IT Experience', 'Years of Non-IT Experience', 'City', 'State',
        'Certifications', 'Awards', 'Achievements', 'GitHub Repo Count',
        'Highest Education Institute Name',
        'Masters/Doctorate Course Name', 'Masters/Doctorate College Name', 'Masters/Doctorate Department Name', 'Masters/Doctorate Year of Completion', 'Masters/Doctorate Percentage',
        'Bachelors Course Name', 'Bachelors College Name', 'Bachelors Department Name', 'Bachelors Year of Completion', 'Bachelors Percentage',
        'Diploma Name', 'Diploma College Name', 'Diploma Department Name', 'Diploma Year of Completion', 'Diploma Percentage',
        'Intermediate / PUC / 12th Name', 'Intermediate / PUC / 12th College Name', 'Intermediate / PUC / 12th Department Name', 'Intermediate / PUC / 12th Year of Completion', 'Intermediate / PUC / 12th Percentage',
        'SSC / 10th Name', 'SSC / 10th College Name', 'SSC / 10th Year of Completion', 'SSC / 10th Percentage'
    ]
    all_columns = base_columns + SKILL_COLUMNS
    result = {col: "" for col in all_columns}
    for col in SKILL_COLUMNS:
        result[col] = 0

    result['User ID'] = user_id
    result['Resume Link'] = resume_link
    
    temp_pdf_path = None
    mistral_response_text_for_logging = "" # Variable to hold response for logging
    try:
        logger.info(f"Processing resume #{resume_index + 1} for user {user_id} with analysis type: {analysis_type}")
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            temp_pdf_path = tmp.name

        download_success, download_msg = download_pdf(resume_link, temp_pdf_path)
        if not download_success:
            raise ValueError(f"Download Error: {download_msg}")

        resume_text, clickable_links = extract_text_and_urls_from_pdf(temp_pdf_path)
        if not resume_text.strip():
            raise ValueError("Could not extract any text from the PDF.")

        prompt = ""
        # ==============================================================================
        #  START OF THE FIX
        # ==============================================================================
        # The prompt for "All Data" has been significantly hardened to prevent the
        # AI from making internal syntax errors like missing commas.
        if analysis_type == "All Data":
            prompt = f"""
You are a machine that strictly outputs JSON. Analyze the resume text and generate a single, valid JSON object with the requested information.

**JSON Syntax Rules (Follow Strictly):**
1.  **Keys and Strings:** All keys and all string values MUST be enclosed in double quotes (").
2.  **Commas:** This is critical. A comma (,) MUST separate every key-value pair in an object (except the last one). A comma (,) MUST separate every element in an array (except the last one). Missing commas will make the output invalid.
3.  **Booleans and Null:** Use `true`, `false`, and `null` without quotes.
4.  **No Extra Text:** The entire output must be ONLY the JSON object. Do not include `json` markdown fences, comments, or any explanatory text before or after the JSON.

**JSON Structure to use:**
{{
  "fullName": "string", "mobileNumber": "string", "email": "string",
  "address": {{"city": "string", "state": "string"}}, "textLinks": ["list of strings"],
  "skills": ["list of strings"], "certifications": ["list of strings"], "awards": ["list of strings"],
  "achievements": ["list of strings"], "yearsITExperience": "float or string", "yearsNonITExperience": "float or string",
  "education": {{
    "masters_doctorate": {{"courseName": "string", "departmentName": "string", "completionYear": "string", "percentage": "string", "collegeName": "string"}},
    "bachelors": {{"courseName": "string", "departmentName": "string", "completionYear": "string", "percentage": "string", "collegeName": "string"}},
    "diploma": {{"courseName": "string", "departmentName": "string", "completionYear": "string", "percentage": "string", "collegeName": "string"}},
    "intermediate_puc_12th": {{"schoolName": "string", "departmentName": "string", "completionYear": "string", "percentage": "string", "collegeName": "string"}},
    "ssc_10th": {{"schoolName": "string", "completionYear": "string", "percentage": "string", "collegeName": "string"}}
  }},
  "experience": [
    {{
      "companyName": "string", 
      "jobTitle": "string", 
      "startDate": "string", 
      "endDate": "string", 
      "description": "string or list of strings"
    }}
  ],
  "projects": ["list of strings or list of objects like {{\"title\": \"string\", \"details\": \"string\"}}"]
}}

Fill missing values with an empty string "", an empty list [], or null. Now, parse the following resume text.

Resume Text:
---
{resume_text}
---
"""
        # ==============================================================================
        #  END OF THE FIX
        # ==============================================================================
        elif analysis_type == "Personal Details":
            prompt = f"""
Analyze the provided resume text and extract ONLY the personal details into a pure JSON object.
- Fill missing values with an empty string "" or [].
- The entire response MUST be ONLY the JSON object, with no extra text, no markdown fences, and no explanations.
JSON Structure: {{"fullName": "string", "mobileNumber": "string", "email": "string", "address": {{"city": "string", "state": "string"}}, "textLinks": ["list of strings"]}}
Resume Text: --- {resume_text} ---
"""
        elif analysis_type == "Skills & Projects":
            prompt = f"""
Analyze the provided resume text and extract ONLY the skills, projects, experience, and related achievements into a pure JSON object.
- For projects, the format should be "Project Title - Technology Stack Used".
- In the experience section, provide a detailed description of responsibilities and technologies for each role.
- The entire response MUST be ONLY the JSON object.
JSON Structure: {{ "skills": ["list of strings"], "certifications": ["list of strings"], "awards": ["list of strings"], "achievements": ["list of strings"], "experience": [{{ "companyName": "string", "jobTitle": "string", "startDate": "string", "endDate": "string", "description": "string"}}], "projects": ["list of strings"]}}
Resume Text: --- {resume_text} ---
"""
        
        mistral_response_text = analyze_text_with_mistral(prompt)
        mistral_response_text_for_logging = mistral_response_text
        data = relaxed_json_loads(mistral_response_text)
        if not isinstance(data, dict): raise ValueError(f"AI returned non-dict data. Type: {type(data)}")
        if "error" in data: raise ValueError(data["error"])

        if analysis_type in ["All Data", "Personal Details"]:
            addr = data.get('address', {}) if isinstance(data.get('address', {}), dict) else {}
            result.update({
                'Full Name': safe_str(data.get('fullName', "")), 'Mobile Number': format_mobile_number(data.get('mobileNumber', "")),
                'Email ID': safe_str(data.get('email', "")), 'City': safe_str(addr.get('city', "")), 'State': safe_str(addr.get('state', "")),
            })
            text_links = data.get('textLinks', [])
            if not isinstance(text_links, list): text_links = [safe_str(text_links)] if text_links else []
            all_links = sorted(list(set([safe_str(x) for x in text_links] + clickable_links)))
            linkedin, github, others = sort_links(all_links)
            result.update({'LinkedIn Link': linkedin, 'GitHub Link': github, 'Other Links': "\n".join(others)})
            if github: result['GitHub Repo Count'] = get_github_repo_count(extract_github_username(github))

        if analysis_type in ["All Data", "Skills & Projects"]:
            result.update(calculate_skill_probabilities(data))
            result.update({
                'Skills': ", ".join([safe_str(s) for s in data.get('skills', []) if isinstance(data.get('skills', []), list)]),
                'Projects': format_projects(data.get('projects', [])),
                'Certifications': "\n".join([safe_str(c) for c in data.get('certifications', []) if isinstance(data.get('certifications', []), list)]),
                'Awards': "\n".join([safe_str(a) for a in data.get('awards', []) if isinstance(data.get('awards', []), list)]),
                'Achievements': "\n".join([safe_str(a) for a in data.get('achievements', []) if isinstance(data.get('achievements', []), list)]),
            })
            latest_exp = get_latest_experience(data.get('experience', []))
            if latest_exp:
                end_date = latest_exp.get('endDate', "")
                result.update({
                    'Latest Experience Company Name': safe_str(latest_exp.get('companyName', "")),
                    'Latest Experience Job Title': safe_str(latest_exp.get('jobTitle', "")),
                    'Latest Experience Start Date': safe_str(latest_exp.get('startDate', "")),
                    'Latest Experience End Date': safe_str(end_date), 'Currently Working? (Yes/No)': "Yes" if is_present_str(end_date) else "No"
                })

        if analysis_type == "All Data":
            result.update({'Years of IT Experience': safe_str(data.get('yearsITExperience', "")), 'Years of Non-IT Experience': safe_str(data.get('yearsNonITExperience', ""))})
            edu = data.get('education', {}) if isinstance(data.get('education', {}), dict) else {}
            result['Highest Education Institute Name'] = get_highest_education_institute(edu)
            edu_levels = {
                'masters_doctorate': ('Masters/Doctorate', 'courseName'), 'bachelors': ('Bachelors', 'courseName'),
                'diploma': ('Diploma', 'courseName'), 'intermediate_puc_12th': ('Intermediate / PUC / 12th', 'schoolName'),
                'ssc_10th': ('SSC / 10th', 'schoolName')
            }
            for key, (prefix, name_key) in edu_levels.items():
                level_data = edu.get(key, {}) if isinstance(edu.get(key, {}), dict) else {}
                result[f'{prefix} Name'] = safe_str(level_data.get(name_key, '')) if name_key else ''
                result[f'{prefix} Course Name'] = safe_str(level_data.get('courseName', ''))
                result[f'{prefix} College Name'] = safe_str(level_data.get('collegeName', ''))
                result[f'{prefix} Department Name'] = safe_str(level_data.get('departmentName', ''))
                result[f'{prefix} Year of Completion'] = safe_str(level_data.get('completionYear', ''))
                result[f'{prefix} Percentage'] = safe_str(level_data.get('percentage', ''))

    except json.JSONDecodeError as e:
        error_msg = "Error: AI returned a malformed response that could not be parsed."
        logger.error(f"Failed processing {user_id} ({resume_link}): JSONDecodeError - {e}")
        logger.debug(f"--- Full Malformed Response ---\n{mistral_response_text_for_logging}\n-----------------------------")
        result['Full Name'] = error_msg
    except Exception as e:
        logger.error(f"Failed processing {user_id} ({resume_link}): {e}")
        result['Full Name'] = f"Error: {e}"
    
    finally:
        if temp_pdf_path and os.path.exists(temp_pdf_path):
            try: os.remove(temp_pdf_path)
            except OSError as e: logger.error(f"Error removing temp file {temp_pdf_path}: {e}")
            
    return result

# ==============================================================================
#  BATCH PROCESSING & UI
# ==============================================================================
def process_resumes_in_batches_live(df, batch_size, worker_function, display_columns, **kwargs):
    st.session_state.comprehensive_results = []
    
    progress_text = st.empty()
    progress_bar = st.progress(0)
    results_placeholder = st.empty()

    tasks = [(row, i) for i, row in df.iterrows()]

    with ThreadPoolExecutor(max_workers=batch_size) as executor:
        futures = {executor.submit(worker_function, row_data, i, **kwargs): i for row_data, i in tasks}
        
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            result = future.result()
            st.session_state.comprehensive_results.append(result)
            progress = (i + 1) / len(df)
            progress_text.markdown(f"**Processing... {i+1}/{len(df)} resumes completed.**")
            progress_bar.progress(progress)
            
            temp_df = pd.DataFrame(st.session_state.comprehensive_results)
            if "All Data" in st.session_state.get('last_analysis_mode', ''):
                temp_df = coerce_probability_columns(temp_df)

            cols_to_show = [col for col in display_columns if col in temp_df.columns]
            results_placeholder.dataframe(temp_df[cols_to_show], height=400, use_container_width=True)
    
    progress_text.success(f"**✅ Analysis Complete! {len(df)}/{len(df)} resumes processed.**")

# ==============================================================================
#  MAIN STREAMLIT APPLICATION
# ==============================================================================
def main():
    st.set_page_config(page_title="AI Resume Analyzer", layout="wide", page_icon="📄")
    
    # --- Settings are moved to the sidebar ---
    with st.sidebar:
        st.header("⚙️ Configuration")
        st.info(f"**Model:** {MISTRAL_MODEL}")
        st.session_state.batch_size = st.slider(
            "Concurrency", 
            1, 10, 3, 
            help="Number of resumes to process in parallel. Lower is safer for API rate limits."
        )
        st.session_state.enable_ocr = st.checkbox(
            "Enable OCR for Scanned PDFs", 
            value=True, 
            help="Slower but necessary for image-based PDFs."
        )
        if not check_tesseract_installation() and st.session_state.get('enable_ocr'):
            st.error("Tesseract is not installed or not in your PATH. OCR will not function.")

    # --- Main Page Starts Here ---
    st.subheader("Step 1: Provide Resume Data")
    
    input_method = st.radio("Choose input method:", ["Upload CSV", "Paste Text"], horizontal=True, label_visibility="collapsed", index=1)
    df_input = None
    
    if input_method == "Upload CSV":
        uploaded_file = st.file_uploader("Upload a CSV file with 'user_id' and 'Resume link' columns.", type="csv", label_visibility="collapsed")
        if uploaded_file: 
            try: df_input = pd.read_csv(uploaded_file, dtype=str).fillna("")
            except Exception as e: st.error(f"Error reading CSV file: {e}")
    else:
        text_data = st.text_area("Paste data here (user_id [Tab] resume_link)", height=150, label_visibility="collapsed", placeholder="user1\thttp://example.com/resume.pdf\nuser2\thttps://drive.google.com/...")
        if text_data:
            try: df_input = pd.read_csv(StringIO(text_data), header=None, names=['user_id', 'Resume link'], dtype=str, sep='\t').fillna("")
            except Exception as e: st.error(f"Could not parse text. Error: {e}")

    if df_input is not None:
        df_input.dropna(subset=['user_id', 'Resume link'], inplace=True)
        df_input = df_input[df_input['Resume link'].str.strip() != ''].reset_index(drop=True)
        if not all(col in df_input.columns for col in ['user_id', 'Resume link']):
            st.error("Input data must contain 'user_id' and 'Resume link' columns.")
        else:
            st.success(f"Successfully loaded {len(df_input)} resume entries.")
            st.markdown("---")

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Step 2: Priority-Based Shortlisting")
                user_requirements = st.text_area(
                    "Enter Job Description or Requirements for Shortlisting",
                    placeholder="e.g., 'Seeking a senior Python developer with 5+ years of experience...' or 'Java Certification'",
                    height=150,
                    help="Paste any text here: a full job description, a list of keywords, or a detailed prompt. The analysis will focus on your query's intent."
                )
                
                st.write("") # Spacer
                if user_requirements.strip():
                    st.info("**Mode:** Priority-Based Shortlisting (Focused Analysis)")
                else:
                    st.info(f"**Mode:** Comprehensive Extraction")

            with col2:
                st.subheader("Step 3: Comprehensive Data Extraction")
                analysis_type = st.selectbox("Choose data to extract (used if shortlisting is empty):", ("All Data", "Personal Details", "Skills & Projects"), help="This analysis runs only if the 'Job Description' box is left empty.")
                
                st.write("") 
                st.subheader("Step 4: Start Analysis")
                
                shortlisting_mode = "Probability Wise (Default)"
                if user_requirements.strip():
                    shortlisting_mode = st.selectbox(
                        "Choose Shortlisting Mode",
                        options=["Probability Wise (Default)", "Priority Wise (P1 / P2 / P3 Bands)"]
                    )
                    button_text = f"🚀 Start Shortlisting for {len(df_input)} Resumes"
                else:
                    button_text = f"🚀 Start '{analysis_type}' Extraction for {len(df_input)} Resumes"

                start_button = st.button(button_text, use_container_width=True, type="primary")

            live_results_container = st.container()

            if start_button:
                st.session_state.analysis_running = True
                with live_results_container:
                    if user_requirements.strip():
                        st.session_state.last_analysis_mode = "shortlisting"
                        st.session_state.shortlisting_mode = shortlisting_mode 
                        shortlisting_cols = [
                            'User ID', 'Resume Link', 'Overall Probability', 'Overall Remarks',
                            'Projects Probability', 'Projects Remarks', 'Skills Probability', 'Skills Remarks',
                            'Experience Probability', 'Experience Remarks', 'Other Probability', 'Other Remarks'
                        ]
                        process_resumes_in_batches_live(
                            df=df_input, batch_size=st.session_state.batch_size, worker_function=process_resume_for_shortlisting,
                            display_columns=shortlisting_cols, user_requirements=user_requirements.strip()
                        )
                    else:
                        st.session_state.last_analysis_mode = analysis_type
                        extraction_cols = [
                            'User ID', 'Resume Link', 'Full Name', 'Mobile Number', 'Email ID', 'LinkedIn Link', 'GitHub Link', 'GitHub Repo Count', 'Other Links', 'City', 'State',
                            'Years of IT Experience', 'Years of Non-IT Experience', 'Highest Education Institute Name', 'Skills'
                        ] + SKILL_COLUMNS + [
                            'Projects', 'Latest Experience Company Name', 'Latest Experience Job Title', 'Latest Experience Start Date', 'Latest Experience End Date', 'Currently Working? (Yes/No)',
                            'Certifications', 'Awards', 'Achievements',
                        ]
                        process_resumes_in_batches_live(
                            df=df_input, batch_size=st.session_state.batch_size, worker_function=process_resume_comprehensively,
                            display_columns=extraction_cols, analysis_type=analysis_type
                        )
                st.session_state.analysis_running = False
            
    if st.session_state.comprehensive_results:
        st.markdown("---")
        st.subheader("Step 5: Review & Download Results")
        
        final_df = pd.DataFrame(st.session_state.comprehensive_results).fillna("")
        
        if st.session_state.last_analysis_mode == "shortlisting":
            prob_cols = ['Overall Probability', 'Projects Probability', 'Skills Probability', 'Experience Probability', 'Other Probability']
            for col in prob_cols:
                if col in final_df.columns:
                    final_df[col] = pd.to_numeric(final_df[col], errors='coerce').fillna(0)

            if st.session_state.get('shortlisting_mode') == "Priority Wise (P1 / P2 / P3 Bands)":
                
                def assign_priority_band(probability):
                    if probability >= 90: return 'P1'
                    elif 75 <= probability < 90: return 'P2'
                    elif 60 <= probability < 75: return 'P3'
                    else: return 'Not Shortlisted'

                final_df['Priority Band'] = final_df['Overall Probability'].apply(assign_priority_band)
                
                band_order = ['P1', 'P2', 'P3', 'Not Shortlisted']
                final_df['Priority Band'] = pd.Categorical(final_df['Priority Band'], categories=band_order, ordered=True)

                display_cols = [
                    'User ID', 'Resume Link', 'Priority Band', 'Overall Probability', 'Overall Remarks',
                    'Projects Probability', 'Projects Remarks', 'Skills Probability', 'Skills Remarks',
                    'Experience Probability', 'Experience Remarks', 'Other Probability', 'Other Remarks'
                ]
                
                final_df_ordered = final_df.sort_values(by=['Priority Band', 'Overall Probability'], ascending=[True, False])
                final_df_ordered = final_df_ordered.reindex(columns=display_cols, fill_value='')
                file_name = f"resume_shortlist_prioritywise_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

            else: # Default "Probability Wise" mode
                display_cols = [
                    'User ID', 'Resume Link', 'Overall Probability', 'Overall Remarks',
                    'Projects Probability', 'Projects Remarks', 'Skills Probability', 'Skills Remarks',
                    'Experience Probability', 'Experience Remarks', 'Other Probability', 'Other Remarks'
                ]
                final_df_ordered = final_df.reindex(columns=display_cols, fill_value='').sort_values(by='Overall Probability', ascending=False)
                file_name = f"resume_shortlist_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        else:
            final_df = coerce_probability_columns(final_df)
            final_column_order = [
                'User ID', 'Resume Link', 'Full Name', 'Mobile Number', 'Email ID', 'LinkedIn Link', 'GitHub Link', 'GitHub Repo Count', 'Other Links', 'City', 'State',
                'Years of IT Experience', 'Years of Non-IT Experience', 'Highest Education Institute Name', 'Skills'
            ] + SKILL_COLUMNS + [
                'Projects', 'Latest Experience Company Name', 'Latest Experience Job Title', 'Latest Experience Start Date', 'Latest Experience End Date', 'Currently Working? (Yes/No)',
                'Certifications', 'Awards', 'Achievements', 'Masters/Doctorate Course Name', 'Masters/Doctorate College Name', 'Masters/Doctorate Department Name', 'Masters/Doctorate Year of Completion', 'Masters/Doctorate Percentage',
                'Bachelors Course Name', 'Bachelors College Name', 'Bachelors Department Name', 'Bachelors Year of Completion', 'Bachelors Percentage',
                'Diploma Course Name', 'Diploma College Name', 'Diploma Department Name', 'Diploma Year of Completion', 'Diploma Percentage',
                'Intermediate / PUC / 12th Name', 'Intermediate / PUC / 12th College Name', 'Intermediate / PUC / 12th Department Name', 'Intermediate / PUC / 12th Year of Completion', 'Intermediate / PUC / 12th Percentage',
                'SSC / 10th Name', 'SSC / 10th College Name', 'SSC / 10th Year of Completion', 'SSC / 10th Percentage'
            ]
            final_df_ordered = final_df.reindex(columns=final_column_order, fill_value='')
            file_name = f"resume_analysis_{st.session_state.last_analysis_mode.replace(' ', '_').lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

        st.dataframe(final_df_ordered, use_container_width=True)
        csv_buffer = final_df_ordered.to_csv(index=False).encode('utf-8')
        st.download_button(label="📥 Download Results as CSV", data=csv_buffer, file_name=file_name, mime="text/csv", use_container_width=True)

if __name__ == "__main__":
    if not MISTRAL_API_KEY:
        st.error("Missing MISTRAL_API_KEY. Set it via st.secrets or an environment variable.", icon="🚨")
    else:
        main()