# granite.py

import json

# ✅ Get questions for a specific role and level
def get_questions_by_role_and_level(role, level, filepath="interview_ques.json"):
    with open(filepath, "r") as f:
        data = json.load(f)

    framework = data.get("qualitativeInterviewFramework", [])
    all_questions = []

    for entry in framework:
        if entry["role"].lower() == role.lower():
            for lvl in entry["levels"]:
                if lvl["level"].lower() == level.lower():
                    for comp in lvl["competencyAreas"]:
                        all_questions.extend(comp["qualitativeQuestionExamples"])

    return all_questions

# ✅ Get all available roles
def get_all_roles(filepath="interview_ques.json"):
    with open(filepath, "r") as f:
        data = json.load(f)
    return [item["role"] for item in data.get("qualitativeInterviewFramework", [])]

# ✅ Get all levels for a given role
def get_levels_for_role(role, filepath="interview_ques.json"):
    with open(filepath, "r") as f:
        data = json.load(f)
    for item in data.get("qualitativeInterviewFramework", []):
        if item["role"].lower() == role.lower():
            return [lvl["level"] for lvl in item["levels"]]
    return []

# ✅ Format a prompt to send to Granite
def format_prompt_for_granite(role, level, questions):
    prompt = f"You are an expert interviewer preparing a {level} {role} candidate for interviews.\n"
    prompt += "Provide model answers for the following questions:\n\n"
    for i, question in enumerate(questions, 1):
        prompt += f"{i}. {question}\n"
    prompt += "\nRespond with clear, concise answers that match the expected level."
    return prompt
