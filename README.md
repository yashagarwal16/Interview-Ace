# 💼 Interview Ace

The **Interview Ace** is an AI-powered interview preparation tool that uses the **Gemini 2.5 Flash** model and **Retrieval-Augmented Generation (RAG)** to generate tailored interview questions and model answers based on the user's resume.

## 🔍 Problem Statement

Job seekers often struggle to find role-specific questions and answers that match their experience level. Generic preparation resources waste time and fail to meet specific job demands. This project solves that problem by personalizing interview preparation based on a candidate's **resume**, **skills**, and **target job role**.

## 🚀 Features

* 📄 **Resume Parsing** – Extracts name, email, skills, role, and level from uploaded PDFs using `PyMuPDF` and `spaCy`.

* 🤖 **Gemini LLM Integration** – Connects to the Gemini 2.5 Flash model via API to generate contextual answers.

* 🎯 **Role-Specific Questions** – Uses a JSON-based interview dataset to fetch questions for a specific role and level.

* 📊 **Supports Technical + Behavioral Rounds** – Covers both coding concepts and soft skills.

* 🧠 **Dynamic Prompting** – Builds prompts based on user input or resume if the role/level is not detected.

## 🛠️ Tech Stack

* **Frontend**: Flask + HTML (for file uploads and UI)

* **Backend**: Python (Flask, spaCy, PyMuPDF, Gemini API)

* **LLM Model**: Gemini 2.5 Flash

## 🧪 How It Works

1. ✅ User uploads a PDF resume.

2. ✅ The resume parser extracts skills, role, and level.

3. ✅ If missing, the user is prompted to input the role/level manually.

4. ✅ The role and level are matched against a local JSON dataset.

5. ✅ Questions are retrieved and used to form a prompt.

6. ✅ The prompt is sent to the Gemini API.

7. ✅ Model answers are returned and displayed to the user.

## 📁 Project Structure

```
├── app.py                     # Flask application
├── resume_parser.py           # PDF parsing and NLP
├── granite.py                 # Question extraction and prompt formatting
├── ibm_granite_api.py         # IBM Granite API integration
├── interview_ques.json        # Role/level-based questions dataset
├── templates/
│   ├── upload.html
│   ├── result.html
│   └── ask_role_level.html
└── .gitignore                 # Files to ignore for Git version control

```

## 🧑‍💻 Authors

Built by Yash Agarwal
# Interview-Ace
