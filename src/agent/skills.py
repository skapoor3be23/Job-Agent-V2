"""Deterministic skill extraction and matching.

Everything the ranking layer and the explanations are built from lives here.
No LLM touches this file: "8 of 10 requirements demonstrated" must be set
arithmetic, not a model's guess, or the headline number is unverifiable.
"""
from __future__ import annotations

import re
from typing import Dict, Iterable, List, Sequence, Set, Tuple

# Canonical skill -> surface forms seen in resumes and JDs.
SKILL_VOCAB: Dict[str, List[str]] = {
    "Python": ["python"],
    "Java": ["java"],
    "C++": ["c++", "cpp"],
    "C": ["c language", "embedded c"],
    "JavaScript": ["javascript", "js"],
    "TypeScript": ["typescript"],
    "Go": ["golang"],
    "Rust": ["rust"],
    "R": ["r programming"],
    "MATLAB": ["matlab"],
    "SQL": ["sql", "postgresql", "postgres", "mysql", "sqlite"],
    "NoSQL": ["nosql", "mongodb", "cassandra", "dynamodb"],
    "Machine Learning": ["machine learning", "ml", "supervised learning"],
    "Deep Learning": ["deep learning", "neural network", "neural networks"],
    "NLP": ["nlp", "natural language processing", "text mining"],
    "Computer Vision": ["computer vision", "opencv", "image processing"],
    "LLM": ["llm", "large language model", "generative ai", "genai", "rag"],
    "PyTorch": ["pytorch", "torch"],
    "TensorFlow": ["tensorflow", "keras"],
    "scikit-learn": ["scikit-learn", "sklearn", "scikit learn"],
    "Pandas": ["pandas"],
    "NumPy": ["numpy"],
    "Data Analysis": ["data analysis", "data analytics", "exploratory data analysis", "eda"],
    "Data Visualization": ["data visualization", "matplotlib", "seaborn", "plotly", "tableau", "power bi"],
    "Statistics": ["statistics", "statistical", "hypothesis testing", "probability"],
    "Docker": ["docker", "containerization", "containers"],
    "Kubernetes": ["kubernetes", "k8s"],
    "AWS": ["aws", "amazon web services", "ec2", "s3 bucket", "sagemaker"],
    "Azure": ["azure"],
    "GCP": ["gcp", "google cloud"],
    "CI/CD": ["ci/cd", "continuous integration", "jenkins", "github actions"],
    "Linux": ["linux", "unix", "bash", "shell scripting"],
    "Git": ["git", "github", "version control", "gitlab"],
    "FastAPI": ["fastapi"],
    "Flask": ["flask"],
    "Django": ["django"],
    "REST API": ["rest api", "restful", "api development"],
    "React": ["react", "reactjs", "react.js"],
    "Node.js": ["node.js", "nodejs"],
    "Streamlit": ["streamlit"],
    "Spark": ["spark", "pyspark"],
    "Hadoop": ["hadoop"],
    "Airflow": ["airflow"],
    "ETL": ["etl", "data pipeline", "data pipelines"],
    "MLOps": ["mlops", "model deployment", "model serving"],
    "ROS": ["ros", "ros 2", "ros2", "robot operating system"],
    "Embedded Systems": ["embedded systems", "microcontroller", "firmware", "rtos"],
    "Robotics": ["robotics", "slam", "kinematics", "motion planning"],
    "Control Systems": ["control systems", "pid", "kalman filter", "ekf"],
    "IoT": ["iot", "internet of things"],
    "CAD": ["cad", "solidworks", "autocad", "fusion 360"],
    "PLC": ["plc", "scada", "ladder logic"],
    "Edge AI": ["edge ai", "edge computing", "tensorrt", "jetson"],
    "Excel": ["excel", "spreadsheet", "vlookup"],
    "Communication": ["communication skills", "written communication", "verbal communication"],
    "Teamwork": ["teamwork", "collaboration", "cross-functional", "team player"],
    "Problem Solving": ["problem solving", "analytical thinking"],
}

# Skills that are usually genuine hard requirements rather than nice-to-haves.
_SOFT_SKILLS = {"Communication", "Teamwork", "Problem Solving", "Excel"}

_ALIAS_INDEX: List[Tuple[str, re.Pattern]] = []
for _canon, _aliases in SKILL_VOCAB.items():
    for _alias in _aliases:
        _ALIAS_INDEX.append((_canon, re.compile(r"(?<![a-z0-9+#.])" + re.escape(_alias) + r"(?![a-z0-9])", re.I)))

_YEARS = re.compile(r"(\d+)\s*\+?\s*(?:to\s*\d+\s*)?year", re.I)
_INTERN_MARKERS = re.compile(r"\b(intern|internship|trainee|fresher|graduate program|entry[- ]level)\b", re.I)
_SENIOR_MARKERS = re.compile(r"\b(senior|lead|principal|staff|manager|head of|architect)\b", re.I)


def extract_skills(text: str) -> List[str]:
    """Canonical skills mentioned in a block of text, in vocabulary order."""
    if not text:
        return []
    found: Set[str] = set()
    for canon, pattern in _ALIAS_INDEX:
        if pattern.search(text):
            found.add(canon)
    return [s for s in SKILL_VOCAB if s in found]


def normalize_skill_list(skills: Iterable[str]) -> List[str]:
    """Map free-text skills (e.g. from the resume profile) onto the canonical
    vocabulary, keeping unrecognised ones as-is so nothing is lost."""
    out: List[str] = []
    seen: Set[str] = set()
    for raw in skills or []:
        if not raw:
            continue
        canon = None
        for name, pattern in _ALIAS_INDEX:
            if pattern.fullmatch(str(raw).strip()) or pattern.search(str(raw)):
                canon = name
                break
        value = canon or str(raw).strip()
        key = value.lower()
        if key not in seen:
            seen.add(key)
            out.append(value)
    return out


def hard_requirements(skills: Sequence[str]) -> List[str]:
    """Drop soft skills so coverage reflects real technical requirements."""
    return [s for s in skills if s not in _SOFT_SKILLS]


def coverage(required: Sequence[str], candidate: Sequence[str]) -> Tuple[float, List[str], List[str]]:
    """Fraction of required skills demonstrated, plus the matched/missing lists.

    Returns (coverage, matched, missing). Coverage is 1.0 when a JD states no
    recognisable technical requirement -- absence of a requirement is not a gap.
    """
    req = hard_requirements(required)
    if not req:
        return 1.0, [], []
    have = {s.lower() for s in candidate}
    matched = [s for s in req if s.lower() in have]
    missing = [s for s in req if s.lower() not in have]
    return len(matched) / len(req), matched, missing


def required_years(jd_text: str) -> float:
    """Smallest explicit year requirement in a JD, 0.0 if none stated."""
    if not jd_text:
        return 0.0
    values = [float(m) for m in _YEARS.findall(jd_text)]
    return min(values) if values else 0.0


def experience_match(jd_text: str, years: float, level: str) -> Tuple[float, List[str]]:
    """Score 0..1 for eligibility, plus any hard blockers.

    A senior-titled role demanding years a student does not have is a real
    blocker and must not be scored away by strong semantic similarity.
    """
    blockers: List[str] = []
    text = jd_text or ""
    needed = required_years(text)
    junior_role = bool(_INTERN_MARKERS.search(text))
    senior_role = bool(_SENIOR_MARKERS.search(text))
    junior_candidate = level in ("student", "intern", "entry")

    if junior_role and junior_candidate:
        score = 1.0
    elif needed <= 0:
        score = 0.8 if not senior_role else 0.45
    elif years >= needed:
        score = 1.0
    else:
        shortfall = needed - years
        score = max(0.0, 1.0 - (shortfall / 5.0))
        if shortfall >= 3:
            blockers.append(f"Requires ~{needed:.0f}+ years; resume shows about {years:.0f}")

    if senior_role and junior_candidate:
        score = min(score, 0.35)
        blockers.append("Posting is for a senior/lead role")

    return round(score, 4), blockers


def role_relevance(title: str, role_families: Sequence[str]) -> float:
    """Token-overlap between a job title and the candidate's role families."""
    if not title or not role_families:
        return 0.0
    stop = {"intern", "internship", "junior", "senior", "the", "and", "of", "a", "i", "ii"}
    title_tokens = {t for t in re.findall(r"[a-z]+", title.lower()) if t not in stop}
    if not title_tokens:
        return 0.0
    best = 0.0
    for family in role_families:
        fam_tokens = {t for t in re.findall(r"[a-z]+", str(family).lower()) if t not in stop}
        if not fam_tokens:
            continue
        overlap = len(title_tokens & fam_tokens) / len(fam_tokens)
        best = max(best, overlap)
    return round(min(1.0, best), 4)


def lexical_similarity(a: str, b: str) -> float:
    """Jaccard fallback used when the embedding API is unavailable."""
    ta = set(re.findall(r"[a-z]{3,}", (a or "").lower()))
    tb = set(re.findall(r"[a-z]{3,}", (b or "").lower()))
    if not ta or not tb:
        return 0.0
    return round(len(ta & tb) / len(ta | tb), 4)
