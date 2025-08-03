# 💼 Interview Trainer Agent

The **Interview Trainer Agent** is an AI-powered interview preparation tool that uses **IBM watsonx.ai Granite-3.1-2B-Instruct** models and **Retrieval-Augmented Generation (RAG)** to generate tailored interview questions and model answers based on the user's resume.

---

## 🔍 Problem Statement

Job seekers often struggle to find role-specific questions and answers that match their experience level. Generic preparation resources waste time and fail to meet specific job demands. This project solves that problem by personalizing interview preparation based on a candidate's **resume**, **skills**, and **target job role**.

---

## 🚀 Features

- 📄 **Resume Parsing** – Extracts name, email, skills, role, and level from uploaded PDFs using `PyMuPDF` and `spaCy`.
- 🤖 **Granite LLM Integration** – Connects to IBM watsonx.ai Granite-3.1-2B-Instruct to generate contextual answers.
- 🎯 **Role-Specific Questions** – Uses a JSON-based interview dataset to fetch questions for a specific role and level.
- 📊 **Supports Technical + Behavioral Rounds** – Covers both coding concepts and soft skills.
- 🧠 **Dynamic Prompting** – Builds prompts based on user input or resume if role/level not detected.

---

## 🛠️ Tech Stack

- **Frontend**: Flask + HTML (for file uploads and UI)
- **Backend**: Python (Flask, spaCy, PyMuPDF, IBM watsonx.ai API)
- **LLM Model**: Granite-3.1-2B-Instruct
- **Cloud**: IBM Cloud Lite / watsonx.ai

---

## 🧪 How It Works

1. ✅ User uploads a PDF resume.
2. ✅ Resume parser extracts skills, role, and level.
3. ✅ If missing, user is prompted to input role/level manually.
4. ✅ Role and level are matched against a local JSON dataset.
5. ✅ Questions are retrieved and used to form a prompt.
6. ✅ Prompt is sent to Granite LLM via IBM watsonx.ai API.
7. ✅ Model answers are returned and displayed to the user.

---

## 📁 Project Structure

```bash
├── app.py                   # Flask application
├── resume_parser.py         # PDF parsing and NLP
├── granite.py               # Question extraction and prompt formatting
├── ibm_granite_api.py       # IBM Granite API integration
├── interview_ques.json      # Role/level-based questions dataset
├── templates/
│   ├── index.html
│   ├── result.html
│   └── ask_role_level.html



🧑‍💻 Authors
Built by Yash Agarwal


---

Let me know if you'd like a badge (e.g., IBM watsonx, Flask) or a GitHub Actions CI badge added!

