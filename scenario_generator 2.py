import re
import time
import os
from huggingface_hub import InferenceClient

HF_TOKEN = os.environ.get("HF_TOKEN")
client = InferenceClient(provider="novita", api_key=HF_TOKEN)


def call_with_retry(messages, max_tokens=1000, temperature=0.5, retries=3):
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
                print(f"Rate limited. Waiting {wait}s before retry...")
                time.sleep(wait)
            else:
                raise e
    raise Exception("Max retries reached. Please wait a moment and try again.")


def generate_custom_scenario(summary_dict, custom_scenario):
    """
    Generates a short bullet-point adapted policy draft for any user scenario.
    Step 1: AI selects relevant sections for this scenario.
    Step 2: AI writes 3 bullet points per section.
    Returns dict: Section name -> bullet point content
    """
    all_sections = {name: cnt for name, cnt in summary_dict.items() if cnt.strip()}
    all_sections_text = "\n".join([f"- {name}" for name in all_sections.keys()])

    # Step 1 - Find relevant sections
    relevance_prompt = f"""You are a policy advisor. Given this scenario: "{custom_scenario}"

Here are the available policy sections:
{all_sections_text}

Which of these sections would actually need to change or be adapted for this scenario?
Only list sections that are truly relevant and would be different.
Do NOT include Vision, Mission, Purpose or introductory sections.
List only the section names, one per line, nothing else."""

    relevant_raw = call_with_retry(
        messages=[{"role": "user", "content": relevance_prompt}],
        max_tokens=200,
        temperature=0.3
    )

    relevant_names = []
    for line in relevant_raw.strip().split("\n"):
        line = line.strip().lstrip("-*123456789. ")
        for name in all_sections.keys():
            if line.lower() in name.lower() or name.lower() in line.lower():
                if name not in relevant_names:
                    relevant_names.append(name)

    if not relevant_names:
        skip_keywords = ["vision", "mission", "purpose", "introduction", "background", "scope"]
        relevant_names = [
            name for name in all_sections.keys()
            if not any(kw in name.lower() for kw in skip_keywords)
        ]

    filtered_dict = {name: all_sections[name] for name in relevant_names if name in all_sections}

    sections_text = "\n\n".join([
        f"Section - {name}:\n{cnt}"
        for name, cnt in filtered_dict.items()
    ])

    section_names = list(filtered_dict.keys())
    section_keys = []
    for s in section_names:
        key = re.sub(r'[^A-Z]', '_', s.upper())
        key = re.sub(r'_+', '_', key).strip('_')
        section_keys.append(key)

    format_lines = "\n".join([f"{k}: <adapted content>" for k in section_keys])

    # Step 2 - Generate focused bullet points combining summary + scenario context
    prompt = f"""You are a policy advisor. For the scenario "{custom_scenario}", write a SHORT focused summary for each section.

STRICT RULES:
- Write exactly 3 bullet points per section, each starting with "- "
- Each bullet must be ONE short sentence
- Combine what the policy says AND how it applies to this scenario
- Be specific and practical
- No paragraphs, no extra explanation

Policy sections for context:
{sections_text}

Respond in this EXACT format:
{format_lines}"""

    time.sleep(5)

    raw = call_with_retry(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1200,
        temperature=0.6
    )

    result = {}
    for key, display_name in zip(section_keys, section_names):
        match = re.search(rf'{re.escape(key)}:\s*(.*?)(?=\n[A-Z][A-Z_]+:|$)', raw, re.DOTALL)
        if match:
            cnt = match.group(1).strip()
            if cnt:
                result[display_name] = cnt
        else:
            result[display_name] = filtered_dict.get(display_name, "")

    return result


def is_question(text):
    """
    Detects if the user typed a question rather than a scenario.
    """
    question_keywords = [
        "what", "who", "when", "where", "why", "how",
        "tell me", "explain", "describe", "list", "give me",
        "summarize", "summary", "main goals", "key points",
        "what are", "what is", "can you"
    ]
    text_lower = text.lower().strip()
    return any(text_lower.startswith(kw) or kw in text_lower for kw in question_keywords)


def answer_policy_question(summary_dict, question):
    """
    Answers a direct question about the policy in short bullet points.
    Returns str: bullet point answer
    """
    policy_context = "\n\n".join([
        f"{name}:\n{cnt}"
        for name, cnt in summary_dict.items()
        if cnt.strip()
    ])

    prompt = f"""You are a helpful policy assistant. Answer this question about the policy clearly and simply.

Question: {question}

Policy Content:
{policy_context}

Rules:
- Answer in 3-5 short bullet points maximum
- Use simple plain language anyone can understand
- Do NOT write a full policy draft
- Do NOT repeat the question
- Be direct and concise"""

    time.sleep(3)

    return call_with_retry(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=400,
        temperature=0.4
    )