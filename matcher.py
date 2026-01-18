import re
from datetime import datetime
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ============================================================
# SKILLS DATABASE (CS / ENGINEERING)
# ============================================================

SKILLS_DB = [
    # Languages
    "Python",
    "Java",
    "C++",
    "C",
    "C#",
    "JavaScript",
    "TypeScript",
    "Go",
    "Rust",
    "Swift",
    "Kotlin",
    "PHP",
    "Ruby",
    "Scala",
    "R",
    "Matlab",
    # Web / Backend
    "HTML",
    "CSS",
    "React",
    "Angular",
    "Vue",
    "Node.js",
    "Next.js",
    "Django",
    "Flask",
    "FastAPI",
    "Spring Boot",
    # Data / ML
    "Machine Learning",
    "Deep Learning",
    "Data Science",
    "NLP",
    "TensorFlow",
    "PyTorch",
    "Scikit-learn",
    "Pandas",
    "NumPy",
    # Cloud / DevOps
    "AWS",
    "Azure",
    "GCP",
    "Docker",
    "Kubernetes",
    "Terraform",
    "CI/CD",
    "Linux",
    "Git",
    "GitHub",
    # Databases
    "SQL",
    "NoSQL",
    "MongoDB",
    "PostgreSQL",
    "MySQL",
    "Redis",
    # CS Concepts
    "Data Structures",
    "Algorithms",
    "System Design",
    "OOP",
]

# ============================================================
# SKILL EXTRACTION
# ============================================================


def extract_skills(text):
    text = text.lower()
    found = []

    for skill in SKILLS_DB:
        s = skill.lower()
        if len(s) <= 3:
            pattern = rf"(?:^|\s){re.escape(s)}(?:$|[\s,./])"
        else:
            pattern = rf"\b{re.escape(s)}\b"

        if re.search(pattern, text):
            found.append(skill)

    return list(set(found))


# ============================================================
# EXPERIENCE EXTRACTION (STRICT & SAFE)
# ============================================================

EXPERIENCE_HEADERS = [
    "experience",
    "work experience",
    "professional experience",
    "employment history",
    "work history",
]

STOP_HEADERS = [
    "education",
    "projects",
    "skills",
    "certifications",
    "achievements",
    "leadership",
    "interests",
    "summary",
]


def extract_experience(text):
    text_lower = text.lower()

    start_idx = None
    for h in EXPERIENCE_HEADERS:
        m = re.search(rf"(?:^|\n)\s*{h}\s*(?:\:)?\s*(?:\n|$)", text_lower)
        if m:
            start_idx = m.end()
            break

    if start_idx is None:
        return ["0 Years"]

    end_idx = len(text_lower)
    for h in STOP_HEADERS:
        m = re.search(rf"(?:^|\n)\s*{h}\s*(?:\:)?\s*(?:\n|$)", text_lower[start_idx:])
        if m:
            end_idx = start_idx + m.start()
            break

    section = text_lower[start_idx:end_idx]

    ranges = re.findall(
        r"(19\d{2}|20\d{2})\s*[-–]\s*(present|current|19\d{2}|20\d{2})", section
    )

    total = 0.0
    now = datetime.now().year

    for start, end in ranges:
        s = int(start)
        e = now if end in ["present", "current"] else int(end)
        if e >= s:
            total += e - s

    if total < 0.5:
        return ["0 Years"]

    total = min(total, 40)
    return [f"{round(total, 1)} Years"]


# ============================================================
# TEXT SIMILARITY
# ============================================================


def calculate_similarity(job_desc, resumes):
    if not resumes or not job_desc:
        return []

    docs = [job_desc] + resumes
    tfidf = TfidfVectorizer(stop_words="english")
    matrix = tfidf.fit_transform(docs)
    scores = cosine_similarity(matrix[0:1], matrix[1:])
    return scores[0].tolist()


# ============================================================
# WEIGHTED MATCH SCORING (UI SLIDERS)
# ============================================================


def calculate_component_scores(
    job_desc_clean, resumes_clean, job_desc_raw, resumes_raw, weights
):
    similarity_scores = calculate_similarity(job_desc_clean, resumes_clean)
    jd_skills = set(extract_skills(job_desc_raw))

    results = []

    for i, raw in enumerate(resumes_raw):
        resume_skills = set(extract_skills(raw))

        skills_score = (
            len(jd_skills & resume_skills) / len(jd_skills)
            if jd_skills
            else min(1.0, len(resume_skills) / 10)
        )

        exp_list = extract_experience(raw)
        years = 0.0
        for e in exp_list:
            m = re.search(r"(\d+(?:\.\d+)?)", e)
            if m:
                years = max(years, float(m.group(1)))

        exp_score = min(1.0, years / 10.0)

        edu_score = (
            0.5
            if any(x in raw.lower() for x in ["bachelor", "master", "phd", "degree"])
            else 0.0
        )

        raw_score = (
            weights["skills"] * skills_score
            + weights["experience"] * exp_score
            + weights["education"] * edu_score
            + weights["relevance"] * similarity_scores[i]
        )

        total_weight = sum(weights.values()) or 1.0
        final = raw_score / total_weight

        results.append(
            {
                "final_score": round(final * 100, 2),
                "skills_score": round(skills_score * 100, 1),
                "exp_score": round(exp_score * 100, 1),
                "edu_score": round(edu_score * 100, 1),
                "relevance_score": round(similarity_scores[i] * 100, 1),
            }
        )

    return results


# ============================================================
# ATS SCORE (CALIBRATED LIKE REAL ATS)
# ============================================================


def calculate_ats_score(resume_text, job_keywords=None):
    if not resume_text or not isinstance(resume_text, str):
        return 0.0

    text = resume_text.lower()
    wc = len(text.split())
    resume_skills = set(s.lower() for s in extract_skills(resume_text))

    # ---- Skill Saturation (40%)
    if job_keywords:
        jd = set(k.lower() for k in job_keywords)
        ratio = len(resume_skills & jd) / max(1, len(jd))
        if ratio >= 0.6:
            skill_score = 1.0
        elif ratio >= 0.4:
            skill_score = 0.85
        elif ratio >= 0.25:
            skill_score = 0.7
        else:
            skill_score = ratio
    else:
        skill_score = min(1.0, len(resume_skills) / 8.0)

    # ---- Structure (20%)
    sections = ["experience", "education", "skills", "projects", "summary"]
    section_score = sum(1 for s in sections if s in text) / len(sections)

    # ---- Experience (15%) – fresher safe
    exp_list = extract_experience(resume_text)
    years = 0.0
    for e in exp_list:
        m = re.search(r"(\d+(?:\.\d+)?)", e)
        if m:
            years = max(years, float(m.group(1)))

    if years == 0:
        exp_score = 0.6
    elif years <= 2:
        exp_score = 0.7
    elif years <= 5:
        exp_score = 0.85
    else:
        exp_score = 1.0

    # ---- Parseability (15%)
    if wc < 80:
        parse_score = 0.3
    elif wc < 150:
        parse_score = 0.6
    elif wc < 300:
        parse_score = 0.85
    else:
        parse_score = 1.0

    # ---- Project bonus
    project_terms = [
        "designed",
        "implemented",
        "optimized",
        "scalable",
        "distributed",
        "latency",
        "throughput",
        "docker",
        "kubernetes",
        "aws",
    ]
    hits = sum(1 for k in project_terms if k in text)
    project_bonus = 0.12 if hits >= 6 else 0.08 if hits >= 3 else 0.0

    # ---- Penalties
    penalty = 0.0
    if wc < 60:
        penalty += 0.1
    if len(resume_skills) > 40:
        penalty += 0.1

    final = (
        0.40 * skill_score
        + 0.20 * section_score
        + 0.15 * exp_score
        + 0.15 * parse_score
    )

    final = final + project_bonus - min(0.2, penalty)
    final = max(0.0, min(1.0, final))

    return round(final * 100, 2)
