import re
import time
import os
from huggingface_hub import InferenceClient

HF_TOKEN = os.environ.get("HF_TOKEN")
client = InferenceClient(provider="novita", api_key=HF_TOKEN)


def preprocess_text(text):
    """
    Cleans raw policy text extracted from PDFs or user input.
    Removes PDF page artifacts, non-ASCII characters, and extra spaces.
    """
    text = re.sub(r'\d+\s*\|\s*P\s*a\s*g\s*e', '', text)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def call_with_retry(messages, max_tokens=1000, temperature=0.4, retries=3):
    """
    Calls Hugging Face chat completion API.
    Retries up to 3 times on 429 rate limit.
    """
    for attempt in range(retries):
        try:
            response = client.chat_completion(
                model="meta-llama/Llama-3.1-8B-Instruct",
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if "429" in str(e) and attempt < retries - 1:
                wait = 30 * (attempt + 1)
                print(f"Rate limited. Waiting {wait}s before retry {attempt + 2}/{retries}...")
                time.sleep(wait)
            else:
                raise e
    raise Exception("Max retries reached. Please wait a moment and try again.")


def detect_policy_info(text):
    """
    Extracts policy name and organisation from the document.
    Returns dict: {"policy_name": str, "organisation": str}
    """
    prompt = (
        "Read the beginning of this policy document and extract:\n"
        "1. The name of the policy\n"
        "2. The name of the organisation or company that published it\n\n"
        "If you cannot find one, write Unknown for that field.\n\n"
        "Policy Document (first part):\n"
        + text[:2000] +
        "\n\nRespond in this EXACT format with no extra text:\n"
        "POLICY_NAME: <policy name here>\n"
        "ORGANISATION: <organisation name here>"
    )
    try:
        raw = call_with_retry(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,
            temperature=0.2
        )
        policy_name = "Unknown Policy"
        organisation = "Unknown Organisation"
        for line in raw.splitlines():
            line = line.strip()
            if line.startswith("POLICY_NAME:"):
                policy_name = line.replace("POLICY_NAME:", "").strip()
            elif line.startswith("ORGANISATION:"):
                organisation = line.replace("ORGANISATION:", "").strip()
        return {"policy_name": policy_name, "organisation": organisation}
    except Exception:
        return {"policy_name": "Policy Document", "organisation": "Unknown Organisation"}


def detect_sections(text):
    """
    Automatically identifies up to 6 main sections of any policy document.
    Falls back to generic sections if detection fails.
    """
    prompt = f"""You are a policy analyst. Read the following policy document.

Identify the main sections or topics covered in this policy.
List up to 6 most important section names, one per line.
Do not include section numbers. Just the topic names.
Example output:
Vision and Mission
Objectives
Key Strategies
Implementation
Monitoring and Reporting

Policy Document:
{text[:3000]}

List the main sections:"""

    raw = call_with_retry(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200,
        temperature=0.3
    )
    sections = []
    for line in raw.strip().split('\n'):
        line = line.strip().lstrip('-*123456789. ')
        if line and len(line) > 3:
            sections.append(line)
    if len(sections) < 2:
        sections = [
            "Purpose and Scope",
            "Key Objectives",
            "Strategies and Actions",
            "Targets and Goals",
            "Monitoring and Reporting"
        ]
    return sections[:6]


def generate_summary(text):
    """
    Generates a structured policy summary using Llama-3.1-8B.
    Auto-detects sections and summarises each one.
    Returns dict: Section name -> summary content
    """
    sections = detect_sections(text)

    section_keys = []
    for s in sections:
        key = re.sub(r'[^A-Z]', '_', s.upper())
        key = re.sub(r'_+', '_', key).strip('_')
        section_keys.append(key)

    format_lines = "\n".join([f"{k}: <your summary here>" for k in section_keys])
    section_list = "\n".join([f"- {s}" for s in sections])

    prompt = f"""You are a professional policy analyst. Read the following policy document carefully.

Write a structured summary covering EXACTLY these sections:
{section_list}

For each section write 2-4 clear concise sentences in your own words.
Do NOT copy sentences directly from the document.
Do NOT include any preamble or extra explanation.

Policy Document:
{text[:6000]}

Respond in this EXACT format with no extra text:
{format_lines}"""

    raw = call_with_retry(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1200,
        temperature=0.4
    )

    result = {}
    for key, display_name in zip(section_keys, sections):
        match = re.search(rf'{re.escape(key)}:\s*(.*?)(?=\n[A-Z][A-Z_]+:|$)', raw, re.DOTALL)
        if match:
            content = match.group(1).strip()
            if content:
                result[display_name] = content

    if not result:
        result["Policy Summary"] = raw

    return result