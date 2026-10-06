import os
import re
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import spacy

from dotenv import load_dotenv

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from sentence_transformers import SentenceTransformer

from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Text,
    DateTime,
)
from sqlalchemy.orm import declarative_base, sessionmaker

from google import genai


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Student Opportunity & Skill Matcher",
    page_icon="🎓",
    layout="wide",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .stApp {
        background: linear-gradient(
            135deg,
            #eef8ff 0%,
            #f7f2ff 50%,
            #fff5fb 100%
        );
    }

    .main-title {
        text-align: center;
        font-size: 42px;
        font-weight: 800;
        color: #243b53;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #52606d;
        margin-bottom: 30px;
    }

    .skill {
        display: inline-block;
        background: #e0f2fe;
        color: #075985;
        padding: 6px 12px;
        margin: 4px;
        border-radius: 15px;
        font-size: 13px;
    }

    .match-score {
        font-size: 32px;
        font-weight: 800;
        text-align: center;
    }

    .info-box {
        background: #eff6ff;
        border-left: 5px solid #3b82f6;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }

    .warning-box {
        background: #fff7ed;
        border-left: 5px solid #f97316;
        padding: 15px;
        border-radius: 10px;
        margin: 15px 0;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


# ============================================================
# GEMINI CLIENT
# ============================================================

gemini_client = None

if GEMINI_API_KEY:
    try:
        gemini_client = genai.Client(
            api_key=GEMINI_API_KEY
        )
    except Exception:
        gemini_client = None


# ============================================================
# DATABASE
# ============================================================

DATABASE_URL = "sqlite:///student_matcher.db"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},
)

Base = declarative_base()


class Student(Base):
    __tablename__ = "students"

    id = Column(
        Integer,
        primary_key=True
    )

    name = Column(
        String(100),
        nullable=False
    )

    college = Column(
        String(200)
    )

    degree = Column(
        String(100)
    )

    year = Column(
        String(50)
    )

    skills = Column(
        Text
    )

    interests = Column(
        Text
    )

    created_at = Column(
        DateTime,
        default=datetime.now
    )


class MatchHistory(Base):
    __tablename__ = "match_history"

    id = Column(
        Integer,
        primary_key=True
    )

    student_name = Column(
        String(100)
    )

    opportunity = Column(
        String(200)
    )

    score = Column(
        String(20)
    )

    created_at = Column(
        DateTime,
        default=datetime.now
    )


Base.metadata.create_all(engine)

SessionLocal = sessionmaker(
    bind=engine
)


# ============================================================
# SPACY MODEL
# ============================================================

@st.cache_resource
def load_spacy():

    try:
        return spacy.load(
            "en_core_web_sm"
        )

    except Exception:
        return spacy.blank("en")


nlp = load_spacy()


# ============================================================
# SENTENCE TRANSFORMER MODEL
# ============================================================

@st.cache_resource
def load_embedding_model():

    try:
        return SentenceTransformer(
            "all-MiniLM-L6-v2"
        )

    except Exception:
        return None


embedding_model = load_embedding_model()


# ============================================================
# SKILL LIST
# ============================================================

KNOWN_SKILLS = [
    "python",
    "java",
    "c",
    "c++",
    "javascript",
    "html",
    "css",
    "sql",
    "mysql",
    "mongodb",
    "pandas",
    "numpy",
    "scikit-learn",
    "machine learning",
    "deep learning",
    "artificial intelligence",
    "ai",
    "data science",
    "data analysis",
    "statistics",
    "nlp",
    "natural language processing",
    "computer vision",
    "tensorflow",
    "pytorch",
    "streamlit",
    "flask",
    "django",
    "react",
    "git",
    "github",
    "excel",
    "power bi",
    "tableau",
    "communication",
    "content writing",
    "video editing",
    "web development",
    "generative ai",
    "transformers",
    "langchain",
]


# ============================================================
# SAMPLE OPPORTUNITIES
# ============================================================

OPPORTUNITIES = [

    {
        "title": "Python Developer Intern",
        "company": "Sample Opportunity",
        "type": "Internship",
        "location": "Remote",
        "skills": [
            "python",
            "sql",
            "git",
            "web development",
        ],
        "description": (
            "Work on Python applications, APIs, "
            "databases and software development projects."
        ),
        "eligibility": (
            "Students with basic Python programming knowledge."
        ),
    },

    {
        "title": "Machine Learning Intern",
        "company": "Sample Opportunity",
        "type": "Internship",
        "location": "Remote",
        "skills": [
            "python",
            "numpy",
            "pandas",
            "scikit-learn",
            "machine learning",
        ],
        "description": (
            "Build machine learning models, "
            "preprocess datasets and evaluate models."
        ),
        "eligibility": (
            "Students interested in AI and machine learning."
        ),
    },

    {
        "title": "Data Science Intern",
        "company": "Sample Opportunity",
        "type": "Internship",
        "location": "Remote",
        "skills": [
            "python",
            "pandas",
            "numpy",
            "statistics",
            "data science",
            "data analysis",
        ],
        "description": (
            "Analyze datasets, create visualizations "
            "and generate useful data insights."
        ),
        "eligibility": (
            "Students with Python and basic statistics."
        ),
    },

    {
        "title": "Generative AI Intern",
        "company": "Sample Opportunity",
        "type": "Internship",
        "location": "Remote",
        "skills": [
            "python",
            "generative ai",
            "transformers",
            "nlp",
            "langchain",
        ],
        "description": (
            "Explore generative AI, language models "
            "and AI-powered applications."
        ),
        "eligibility": (
            "Students interested in modern AI technologies."
        ),
    },

    {
        "title": "NLP Intern",
        "company": "Sample Opportunity",
        "type": "Internship",
        "location": "Remote",
        "skills": [
            "python",
            "nlp",
            "transformers",
            "machine learning",
        ],
        "description": (
            "Develop natural language processing "
            "solutions using AI and machine learning."
        ),
        "eligibility": (
            "Students interested in NLP and AI."
        ),
    },

    {
        "title": "Data Analyst Intern",
        "company": "Sample Opportunity",
        "type": "Internship",
        "location": "Remote",
        "skills": [
            "python",
            "sql",
            "excel",
            "power bi",
            "data analysis",
        ],
        "description": (
            "Analyze business data, create reports "
            "and build data visualizations."
        ),
        "eligibility": (
            "Students interested in data analytics."
        ),
    },

    {
        "title": "AI Research Intern",
        "company": "Sample Opportunity",
        "type": "Internship",
        "location": "Remote",
        "skills": [
            "python",
            "machine learning",
            "deep learning",
            "pytorch",
            "artificial intelligence",
        ],
        "description": (
            "Experiment with AI models and machine "
            "learning techniques for research projects."
        ),
        "eligibility": (
            "Students interested in AI research."
        ),
    },

    {
        "title": "Web Development Intern",
        "company": "Sample Opportunity",
        "type": "Internship",
        "location": "Remote",
        "skills": [
            "html",
            "css",
            "javascript",
            "python",
            "web development",
        ],
        "description": (
            "Build and improve responsive websites "
            "and web applications."
        ),
        "eligibility": (
            "Students interested in web development."
        ),
    },
]


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):

    if not text:
        return ""

    text = str(text).lower()

    text = re.sub(
        r"[^a-zA-Z0-9+#. ]",
        " ",
        text,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


# ============================================================
# NLP PROCESSING
# ============================================================

def process_text(text):

    cleaned = clean_text(text)

    doc = nlp(cleaned)

    words = []

    for token in doc:

        if token.is_alpha and not token.is_stop:

            word = token.lemma_.lower()

            if word:
                words.append(word)

    return " ".join(words)


# ============================================================
# SKILL EXTRACTION
# ============================================================

def extract_skills(text):

    cleaned = clean_text(text)

    found_skills = []

    for skill in KNOWN_SKILLS:

        if skill in cleaned:

            found_skills.append(skill)

    return sorted(
        set(found_skills)
    )


# ============================================================
# TF-IDF SIMILARITY
# ============================================================

def tfidf_similarity(
    student_text,
    opportunity_text
):

    try:

        vectorizer = TfidfVectorizer(
            stop_words="english"
        )

        vectors = vectorizer.fit_transform(
            [
                student_text,
                opportunity_text,
            ]
        )

        score = cosine_similarity(
            vectors[0:1],
            vectors[1:2],
        )[0][0]

        return float(score)

    except Exception:

        return 0.0


# ============================================================
# SEMANTIC SIMILARITY
# ============================================================

def semantic_similarity(
    student_text,
    opportunity_text
):

    if embedding_model is None:
        return 0.0

    try:

        embeddings = embedding_model.encode(
            [
                student_text,
                opportunity_text,
            ]
        )

        score = cosine_similarity(
            [embeddings[0]],
            [embeddings[1]],
        )[0][0]

        return float(score)

    except Exception:

        return 0.0


# ============================================================
# MATCHING ENGINE
# ============================================================

def calculate_match(
    student_skills,
    interests,
    opportunity,
):

    student_skill_set = {
        clean_text(skill)
        for skill in student_skills
    }

    required_skill_set = {
        clean_text(skill)
        for skill in opportunity["skills"]
    }

    matching_skills = (
        student_skill_set
        & required_skill_set
    )

    missing_skills = (
        required_skill_set
        - student_skill_set
    )

    if required_skill_set:

        skill_score = (
            len(matching_skills)
            / len(required_skill_set)
        )

    else:

        skill_score = 0.0

    student_text = process_text(
        " ".join(student_skills)
        + " "
        + interests
    )

    opportunity_text = process_text(
        opportunity["title"]
        + " "
        + opportunity["description"]
        + " "
        + " ".join(opportunity["skills"])
    )

    tfidf_score = tfidf_similarity(
        student_text,
        opportunity_text,
    )

    semantic_score = semantic_similarity(
        student_text,
        opportunity_text,
    )

    # Weighted matching
    final_score = (
        skill_score * 0.50
        + tfidf_score * 0.25
        + semantic_score * 0.25
    )

    return {
        "score": round(
            final_score * 100,
            2,
        ),
        "matching_skills": sorted(
            matching_skills
        ),
        "missing_skills": sorted(
            missing_skills
        ),
        "skill_score": round(
            skill_score * 100,
            2,
        ),
        "tfidf_score": round(
            tfidf_score * 100,
            2,
        ),
        "semantic_score": round(
            semantic_score * 100,
            2,
        ),
    }


# ============================================================
# DATABASE FUNCTIONS
# ============================================================

def save_student(
    name,
    college,
    degree,
    year,
    skills,
    interests,
):

    session = SessionLocal()

    try:

        student = Student(
            name=name,
            college=college,
            degree=degree,
            year=year,
            skills=", ".join(skills),
            interests=interests,
        )

        session.add(student)
        session.commit()

    finally:

        session.close()


def save_match(
    student_name,
    opportunity,
    score,
):

    session = SessionLocal()

    try:

        record = MatchHistory(
            student_name=student_name,
            opportunity=opportunity,
            score=f"{score}%",
        )

        session.add(record)
        session.commit()

    finally:

        session.close()


def load_history():

    session = SessionLocal()

    try:

        records = (
            session.query(MatchHistory)
            .order_by(
                MatchHistory.created_at.desc()
            )
            .limit(20)
            .all()
        )

        rows = []

        for record in records:

            rows.append(
                {
                    "Student": record.student_name,
                    "Opportunity": record.opportunity,
                    "Score": record.score,
                    "Date": record.created_at,
                }
            )

        return pd.DataFrame(rows)

    finally:

        session.close()


# ============================================================
# GEMINI AI CAREER GUIDANCE
# ============================================================

def generate_ai_advice(
    name,
    skills,
    interests,
    results,
):

    if gemini_client is None:

        return (
            "Gemini AI is not connected.\n\n"
            "Please add GEMINI_API_KEY to your .env file."
        )

    top_opportunities = [
        item["opportunity"]["title"]
        for item in results[:5]
    ]

    prompt = f"""
You are an AI career guidance assistant for college students.

Student name:
{name}

Current skills:
{", ".join(skills)}

Career interests:
{interests}

Top matched opportunities:
{", ".join(top_opportunities)}

Give practical and beginner-friendly career guidance.

Include:

1. Best career direction
2. Skills to learn next
3. Project ideas
4. Internship preparation
5. A simple 30-day learning roadmap

Keep the explanation clear and easy for a college student.
"""

    try:

        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )

        return response.text

    except Exception as error:

        return (
            "Gemini could not generate the career plan.\n\n"
            f"Error: {error}"
        )


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="main-title">
        🎓 AI Student Opportunity & Skill Matcher
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="subtitle">
        Find internships and career opportunities that match
        your skills, interests and goals.
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR - STUDENT PROFILE
# ============================================================

with st.sidebar:

    st.header("👤 Student Profile")

    name = st.text_input(
        "Name",
        placeholder="Enter your name",
    )

    college = st.text_input(
        "College",
        placeholder="Enter your college",
    )

    degree = st.selectbox(
        "Degree",
        [
            "B.Tech",
            "B.E",
            "B.Sc",
            "BCA",
            "MCA",
            "Other",
        ],
    )

    year = st.selectbox(
        "Year",
        [
            "1st Year",
            "2nd Year",
            "3rd Year",
            "4th Year",
            "Postgraduate",
        ],
    )

    skills_input = st.text_area(
        "Your Skills",
        placeholder=(
            "Example:\n"
            "Python, SQL, Pandas, Machine Learning"
        ),
    )

    interests = st.text_area(
        "Career Interests",
        placeholder=(
            "Example:\n"
            "Artificial Intelligence, Data Science"
        ),
    )

    find_button = st.button(
        "🔍 Find My Opportunities",
        use_container_width=True,
    )


# ============================================================
# FIND OPPORTUNITIES
# ============================================================

if find_button:

    if not name.strip():

        st.error(
            "Please enter your name."
        )

    elif not skills_input.strip():

        st.error(
            "Please enter at least one skill."
        )

    else:

        entered_skills = [
            clean_text(skill)
            for skill in skills_input.split(",")
            if skill.strip()
        ]

        detected_skills = extract_skills(
            skills_input
        )

        student_skills = sorted(
            set(
                entered_skills
                + detected_skills
            )
        )

        save_student(
            name,
            college,
            degree,
            year,
            student_skills,
            interests,
        )

        results = []

        for opportunity in OPPORTUNITIES:

            match = calculate_match(
                student_skills,
                interests,
                opportunity,
            )

            results.append(
                {
                    "opportunity": opportunity,
                    **match,
                }
            )

        results.sort(
            key=lambda item: item["score"],
            reverse=True,
        )

        st.session_state["results"] = results

        st.session_state["student_name"] = name

        st.session_state["student_skills"] = (
            student_skills
        )

        st.session_state["interests"] = interests


# ============================================================
# RESULTS
# ============================================================

if "results" in st.session_state:

    results = st.session_state["results"]

    student_skills = (
        st.session_state["student_skills"]
    )

    st.markdown(
        "## 📊 Your Profile Analysis"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Skills Detected",
            len(student_skills),
        )

    with col2:

        st.metric(
            "Opportunities Checked",
            len(results),
        )

    with col3:

        st.metric(
            "Best Match",
            f"{results[0]['score']}%",
        )

    # --------------------------------------------------------
    # SKILLS
    # --------------------------------------------------------

    st.markdown(
        "### 🧠 Your Skills"
    )

    skill_html = ""

    for skill in student_skills:

        skill_html += (
            f'<span class="skill">{skill}</span>'
        )

    st.markdown(
        skill_html,
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # OPPORTUNITIES
    # --------------------------------------------------------

    st.markdown(
        "## 🎯 Recommended Opportunities"
    )

    st.markdown(
        """
        <div class="warning-box">
        ⚠️ The opportunities shown in this demo are sample
        opportunities. They are not verified live internship
        listings.
        </div>
        """,
        unsafe_allow_html=True,
    )

    for index, result in enumerate(results):

        opportunity = result["opportunity"]

        with st.container(border=True):

            col1, col2 = st.columns(
                [4, 1]
            )

            with col1:

                st.subheader(
                    f"{index + 1}. "
                    f"{opportunity['title']}"
                )

                st.write(
                    f"**Company:** "
                    f"{opportunity['company']}"
                )

                st.write(
                    f"**Type:** "
                    f"{opportunity['type']}"
                )

                st.write(
                    f"**Location:** "
                    f"{opportunity['location']}"
                )

            with col2:

                st.markdown(
                    f"""
                    <div class="match-score">
                        {result['score']}%
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.caption(
                    "Match Score"
                )

            st.write(
                opportunity["description"]
            )

            st.write(
                f"**Eligibility:** "
                f"{opportunity['eligibility']}"
            )

            if result["matching_skills"]:

                st.success(
                    "Matching skills: "
                    + ", ".join(
                        result["matching_skills"]
                    )
                )

            if result["missing_skills"]:

                st.warning(
                    "Skills to improve: "
                    + ", ".join(
                        result["missing_skills"]
                    )
                )

            with st.expander(
                "📊 View Matching Details"
            ):

                st.write(
                    f"Skill Match: "
                    f"{result['skill_score']}%"
                )

                st.write(
                    f"TF-IDF Similarity: "
                    f"{result['tfidf_score']}%"
                )

                st.write(
                    f"Semantic Similarity: "
                    f"{result['semantic_score']}%"
                )

            if st.button(
                "💾 Save This Match",
                key=f"save_{index}",
            ):

                save_match(
                    st.session_state["student_name"],
                    opportunity["title"],
                    result["score"],
                )

                st.success(
                    "Match saved successfully!"
                )

    # --------------------------------------------------------
    # SKILL GAP
    # --------------------------------------------------------

    st.markdown(
        "## 📚 Overall Skill Gap"
    )

    all_required_skills = set()

    for result in results:

        all_required_skills.update(
            result["opportunity"]["skills"]
        )

    missing_overall = sorted(
        all_required_skills
        - set(student_skills)
    )

    if missing_overall:

        st.write(
            "These skills can improve your chances "
            "of matching more opportunities:"
        )

        for skill in missing_overall:

            st.write(
                f"• **{skill}**"
            )

    else:

        st.success(
            "🎉 Your skills cover the major requirements "
            "of these sample opportunities!"
        )

    # --------------------------------------------------------
    # AI CAREER PLAN
    # --------------------------------------------------------

    st.markdown(
        "## 🤖 AI Career Guidance"
    )

    if st.button(
        "✨ Generate My AI Career Plan",
        use_container_width=True,
    ):

        with st.spinner(
            "Gemini AI is analyzing your profile..."
        ):

            advice = generate_ai_advice(
                st.session_state["student_name"],
                student_skills,
                st.session_state["interests"],
                results,
            )

        st.markdown(
            advice
        )


# ============================================================
# MATCH HISTORY
# ============================================================

st.markdown(
    "## 🕒 Saved Match History"
)

history_df = load_history()

if not history_df.empty:

    st.dataframe(
        history_df,
        use_container_width=True,
        hide_index=True,
    )

else:

    st.info(
        "Your saved matches will appear here."
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.markdown(
    """
    <div style="text-align:center;">
        🎓 <b>AI Student Opportunity & Skill Matcher</b><br>
        Python • Streamlit • NLP • Machine Learning •
        Sentence Transformers • Gemini AI • SQLAlchemy
    </div>
    """,
    unsafe_allow_html=True,
)