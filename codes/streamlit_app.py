import streamlit as st
from pypdf import PdfReader
from summarizer import preprocess_text, generate_summary, detect_policy_info
from scenario_generator import generate_custom_scenario, answer_policy_question, is_question

st.set_page_config(page_title="AI Policy Analysis System", layout="wide")

# -------------------------------------------------------
# Hero Header
# -------------------------------------------------------
st.markdown("""
<div style="
    background: linear-gradient(135deg, #0f2847 0%, #1a3a5c 50%, #1e5c3a 100%);
    border-radius: 16px;
    padding: 28px 36px;
    margin-bottom: 8px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.18);
    border: 1px solid rgba(255,255,255,0.08);
">
    <div style="
        font-family: 'Playfair Display', serif;
        font-size: 42px;
        font-weight: 900;
        color: #ffffff;
        letter-spacing: 0.5px;
        line-height: 1.2;
        text-shadow: 0 3px 12px rgba(0,0,0,0.4);
    ">
        AI-Assisted Policy Analysis System
    </div>
    <div style="margin-top:12px;">
        <span style="
            background: rgba(45,138,78,0.3);
            border: 1px solid rgba(45,138,78,0.6);
            color: #ffffff;
            border-radius: 20px;
            padding: 3px 14px;
            font-size: 11px;
            font-weight: 600;
            font-family: 'Inter', sans-serif;
            letter-spacing: 0.5px;
            margin-right: 8px;
        ">Llama 3.1 - Hugging Face</span>
        <span style="
            background: rgba(37,99,235,0.3);
            border: 1px solid rgba(37,99,235,0.6);
            color: #ffffff;
            border-radius: 20px;
            padding: 3px 14px;
            font-size: 11px;
            font-weight: 600;
            font-family: 'Inter', sans-serif;
            letter-spacing: 0.5px;
        ">Generative AI</span>
    </div>
</div>
""", unsafe_allow_html=True)

# -------------------------------------------------------
# Subtitle Banner
# -------------------------------------------------------
st.markdown("""
<div style='
    background: linear-gradient(135deg, #1a3a5c, #2c5282);
    border-radius: 10px;
    padding: 14px 22px;
    margin-bottom: 20px;
    border-left: 5px solid #2d8a4e;
'>
    <span style='
        color: #ffffff;
        font-size: 15px;
        font-weight: 500;
        font-family: Inter, sans-serif;
        letter-spacing: 0.2px;
    '>
        Upload or paste a policy document, generate a structured summary, and adapt it for different scenarios.
    </span>
</div>
""", unsafe_allow_html=True)

# -------------------------------------------------------
# Styling
# -------------------------------------------------------
st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Playfair+Display:wght@600&display=swap');

:root {
    --navy:       #1a3a5c;
    --navy-light: #2c5282;
    --green:      #2d8a4e;
    --green-light:#e8f5ee;
    --blue:       #2563eb;
    --white:      #ffffff;
    --border:     #e2e8f0;
    --shadow:     0 2px 12px rgba(0,0,0,0.08);
}

.stApp {
    background-color: #b8d4f0;
    background-image:
        radial-gradient(ellipse at 0% 0%, rgba(147,197,253,0.4) 0%, transparent 50%),
        radial-gradient(ellipse at 100% 100%, rgba(191,219,254,0.4) 0%, transparent 50%);
    font-family: 'Inter', sans-serif;
    color: #0a0a0a;
}
#MainMenu, footer, header { visibility: hidden; }

h2 {
    font-family: 'Playfair Display', serif !important;
    color: var(--navy) !important;
    font-size: 20px !important;
    padding-bottom: 10px;
    border-bottom: 2px solid var(--border);
    margin-bottom: 18px !important;
}
h3 { color: var(--navy) !important; font-size: 15px !important; }

.stButton > button {
    background: linear-gradient(135deg, var(--navy), var(--navy-light)) !important;
    color: #ffffff !important;
    font-weight: 600 !important;
    font-family: 'Inter', sans-serif !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 10px 22px !important;
    letter-spacing: 0.2px !important;
    box-shadow: 0 3px 10px rgba(26,58,92,0.2) !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 5px 16px rgba(26,58,92,0.3) !important;
}

[data-testid="stFileUploader"] {
    background: var(--white);
    border: 2px dashed var(--border);
    border-radius: 10px;
    padding: 10px;
}

textarea {
    background: var(--white) !important;
    border: 1.5px solid var(--border) !important;
    border-radius: 8px !important;
    color: #0a0a0a !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 14px !important;
}

.section-card {
    background: #dbeeff;
    border-radius: 12px;
    padding: 18px 22px;
    margin-bottom: 12px;
    border-top: 2px solid #000000;
    border-right: 2px solid #000000;
    border-bottom: 2px solid #000000;
    border-left: 5px solid var(--green);
    box-shadow: var(--shadow);
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}
.section-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(0,0,0,0.1);
}
.section-title {
    font-size: 11px;
    font-weight: 700;
    color: var(--green);
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 8px;
}

.draft-card {
    background: #dbeeff;
    border-radius: 12px;
    padding: 18px 22px;
    margin-bottom: 12px;
    border-top: 2px solid #000000;
    border-right: 2px solid #000000;
    border-bottom: 2px solid #000000;
    border-left: 5px solid var(--blue);
    box-shadow: var(--shadow);
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}
.draft-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(0,0,0,0.1);
}
.draft-title {
    font-size: 11px;
    font-weight: 700;
    color: var(--blue);
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 8px;
}

.section-content {
    font-size: 14px;
    color: #0a0a0a;
    line-height: 1.85;
}

.coverage-item {
    font-size: 13px;
    color: #0a0a0a;
    padding: 3px 0;
}

[data-testid="stDownloadButton"] > button {
    background: var(--green-light) !important;
    border: 1.5px solid var(--green) !important;
    color: var(--green) !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    border-radius: 7px !important;
    padding: 6px 16px !important;
}
[data-testid="stDownloadButton"] > button:hover {
    background: #d1f0de !important;
}

hr { border-color: var(--border) !important; }

</style>
""", unsafe_allow_html=True)


# -------------------------------------------------------
# Helper Functions
# -------------------------------------------------------
def read_pdf(file):
    """Extracts text from all pages of an uploaded PDF."""
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + " "
    return text


def render_summary_card(title, content):
    """Renders a summary card with green left border."""
    st.markdown(f"""
    <div class="section-card">
        <div class="section-title">{title}</div>
        <div class="section-content">{content}</div>
    </div>
    """, unsafe_allow_html=True)


def render_draft_card(title, content):
    """Renders a draft card with bullet points and blue left border."""
    lines = content.replace("- ", "\n- ").split("\n")
    bullets = [l.strip().lstrip("- ").strip() for l in lines if l.strip().lstrip("- ").strip()]
    formatted = "".join([f"<div style='margin-bottom:6px;'>&#8226; {b}</div>" for b in bullets])
    st.markdown(f"""
    <div class="draft-card">
        <div class="draft-title">{title}</div>
        <div class="section-content">{formatted}</div>
    </div>
    """, unsafe_allow_html=True)


def dict_to_plaintext(d):
    """Converts section dictionary to plain text for download."""
    return "\n\n".join([f"{k}:\n{v}" for k, v in d.items()])


def check_coverage(summary_dict):
    """Checks all detected policy sections are present in the summary."""
    return [(s, bool(content.strip())) for s, content in summary_dict.items()]


# -------------------------------------------------------
# Main Layout
# -------------------------------------------------------
left, right = st.columns(2)


# ======================================================
# LEFT PANEL - Policy Summarisation
# ======================================================
with left:
    st.markdown("<h1 style='font-family:Playfair Display,serif; font-size:28px; font-weight:800; color:#1a3a5c;'>Policy Summarisation</h1>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload Policy PDF", type=["pdf"])
    policy_text = st.text_area("Or Paste Policy Text Here", height=180)

    if st.button("Generate Summary"):
        if uploaded_file is None and policy_text.strip() == "":
            st.warning("Please upload a PDF or paste policy text.")
        else:
            try:
                if uploaded_file is not None:
                    policy_text = read_pdf(uploaded_file)
                if not policy_text.strip():
                    st.error("Could not extract text from PDF. Try pasting manually.")
                else:
                    with st.spinner("Generating structured policy summary..."):
                        clean_text = preprocess_text(policy_text)
                        policy_info = detect_policy_info(clean_text)
                        summary_dict = generate_summary(clean_text)

                    st.session_state["summary_dict"] = summary_dict
                    st.session_state["summary_text"] = dict_to_plaintext(summary_dict)
                    st.session_state["policy_info"] = policy_info

            except Exception as e:
                st.error(f"Summary generation failed: {e}")

    # Always show summary from session state so it stays
    # visible even when scenario buttons are clicked
    if "summary_dict" in st.session_state:

        if "policy_info" in st.session_state:
            info = st.session_state["policy_info"]
            st.markdown(f"""
<div style="
    background: linear-gradient(135deg, #1a3a5c, #2c5282);
    border-radius: 12px;
    padding: 16px 22px;
    margin-bottom: 16px;
    box-shadow: 0 3px 12px rgba(0,0,0,0.12);
">
    <div style="font-family:'Playfair Display',serif; font-size:20px; font-weight:800; color:#ffffff;">
        {info['policy_name']} &nbsp;<span style="font-weight:400; font-size:15px; color:#ffffff;">in</span>&nbsp; {info['organisation']}
    </div>
</div>
""", unsafe_allow_html=True)

        st.subheader("Policy Summary")
        for section, content in st.session_state["summary_dict"].items():
            if content.strip():
                render_summary_card(section, content)

        st.markdown("---")
        st.subheader("Coverage Check")
        checks = check_coverage(st.session_state["summary_dict"])
        col_a, col_b = st.columns(2)
        for i, (section, found) in enumerate(checks):
            with (col_a if i % 2 == 0 else col_b):
                st.markdown(
                    f"<div class='coverage-item'>{'&#10003;' if found else '&#10007;'} {section}</div>",
                    unsafe_allow_html=True
                )

        st.markdown("")
        st.download_button(
            "Download Summary",
            data=st.session_state["summary_text"],
            file_name="policy_summary.txt",
            mime="text/plain"
        )


# ======================================================
# RIGHT PANEL - Scenario Generator
# ======================================================
with right:
    st.markdown("<h1 style='font-family:Playfair Display,serif; font-size:28px; font-weight:800; color:#1a3a5c;'>Scenario-Based Policy Generator</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#0a0a0a; font-size:14px; font-weight:600; background:#d0e8ff; padding:8px 12px; border-radius:6px; border-left:4px solid #1a3a5c;'>Type your own scenarios below. Add as many as you need - works with any policy.</p>", unsafe_allow_html=True)

    if "scenario_list" not in st.session_state:
        st.session_state["scenario_list"] = ["", "", ""]

    col_add, col_remove = st.columns([1, 1])
    with col_add:
        if st.button("Add Scenario"):
            st.session_state["scenario_list"].append("")
            st.rerun()
    with col_remove:
        if st.button("Remove Last") and len(st.session_state["scenario_list"]) > 1:
            st.session_state["scenario_list"].pop()
            st.rerun()

    if "scenario_results" not in st.session_state:
        st.session_state["scenario_results"] = {}

    for i, val in enumerate(st.session_state["scenario_list"]):
        col_in, col_btn = st.columns([4, 1])
        with col_in:
            user_input = st.text_input(
                f"Scenario {i+1}",
                value=val,
                placeholder='e.g. "Flood Disaster", "What are the main goals?"...',
                key=f"scenario_input_{i}"
            )
        with col_btn:
            st.markdown("<div style='margin-top:28px;'></div>", unsafe_allow_html=True)
            generate_clicked = st.button("Go", key=f"gen_btn_{i}")

        if generate_clicked:
            if "summary_dict" not in st.session_state:
                st.warning("Please generate the summary first.")
            elif not user_input.strip():
                st.warning("Please type a scenario or question first.")
            else:
                scenario = user_input.strip()
                try:
                    if is_question(scenario):
                        with st.spinner("Answering..."):
                            answer = answer_policy_question(
                                st.session_state["summary_dict"],
                                scenario
                            )
                        st.session_state["scenario_results"][i] = {
                            "type": "question",
                            "scenario": scenario,
                            "answer": answer
                        }
                    else:
                        with st.spinner(f"Adapting policy for '{scenario}'..."):
                            adapted = generate_custom_scenario(
                                st.session_state["summary_dict"],
                                scenario
                            )
                        st.session_state["scenario_results"][i] = {
                            "type": "scenario",
                            "scenario": scenario,
                            "adapted": adapted
                        }
                except Exception as e:
                    st.error(f"Failed: {e}")

        # Show result for THIS row right below its input box
        if i in st.session_state.get("scenario_results", {}):
            result = st.session_state["scenario_results"][i]
            if result["type"] == "question":
                st.markdown(f"""
<div style="
    background: linear-gradient(135deg, #1a3a5c, #2c5282);
    border-radius: 12px;
    padding: 16px 22px;
    margin-bottom: 16px;
    color: #ffffff;
    box-shadow: 0 4px 16px rgba(0,0,0,0.15);
    border-left: 5px solid #f0c040;
">
    <div style="font-family:'Playfair Display',serif; font-size:16px; font-weight:700; margin-bottom:10px; color:#ffffff;">
        {result["scenario"]}
    </div>
    <div style="font-size:14px; color:#ffffff; line-height:1.8;">
        {result["answer"].replace(chr(10), "<br>")}
    </div>
</div>
""", unsafe_allow_html=True)
                q_text = "Question: " + result["scenario"] + "\n\nAnswer:\n" + result["answer"]
                st.download_button(
                    "Download Answer",
                    data=q_text,
                    file_name="answer_" + result["scenario"][:20].replace(" ", "_") + ".txt",
                    mime="text/plain",
                    key="download_q_row_" + str(i) + "_" + result["scenario"][:10].replace(" ", "_")
                )
            else:
                sections_html = ""
                for section, adapted_content in result["adapted"].items():
                    if adapted_content.strip():
                        raw_lines = adapted_content.split("- ")
                        bullets = [l.strip() for l in raw_lines if l.strip()]
                        bullets_html = "".join(["<div style='margin-bottom:5px; font-size:13px; color:#0a0a0a;'>&#8226; " + b + "</div>" for b in bullets])
                        sections_html += "<div style='margin-bottom:16px;'><div style='font-size:10px; font-weight:700; color:#2d8a4e; text-transform:uppercase; letter-spacing:1px; margin-bottom:6px;'>" + section + "</div>" + bullets_html + "</div>"
                st.markdown(f"""
<div style="
    background: linear-gradient(135deg, #0f2847, #1a3a5c);
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 16px;
    box-shadow: 0 4px 16px rgba(0,0,0,0.15);
    border-left: 5px solid #2d8a4e;
">
    <div style="font-family:'Playfair Display',serif; font-size:17px; font-weight:700; color:#ffffff; margin-bottom:16px;">
        {result["scenario"]}
    </div>
    <div style="background:#ffffff; border-radius:10px; padding:16px 18px;">
        {sections_html}
    </div>
</div>
""", unsafe_allow_html=True)
                st.download_button(
                    f"Download '{result['scenario']}' Draft",
                    data=dict_to_plaintext(result["adapted"]),
                    file_name=f"policy_draft_{result['scenario'][:20].replace(' ', '_')}.txt",
                    mime="text/plain",
                    key=f"download_result_row_{i}_{result['scenario'][:10].replace(' ', '_')}"
                )
            st.markdown("---")