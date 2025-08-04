from flask import Flask, request, render_template, redirect, url_for
from resume_parser import extract_resume_info
from granite import (
    get_questions_by_role_and_level,
    get_levels_for_role,
    get_all_roles,
    format_prompt_for_granite
)
from gemini_api import get_gemini_response  # ✅ using Gemini 1.5 Pro
import os

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        uploaded_file = request.files["resume"]
        if uploaded_file.filename == "":
            return "No file uploaded."

        # Save the uploaded file
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], uploaded_file.filename)
        uploaded_file.save(filepath)

        # Extract resume information
        info = extract_resume_info(filepath)
        print("✅ Extracted Info:", info)

        role = info.get("role")
        level = info.get("level")

        all_roles = get_all_roles()

        # Ask user manually if role or level is not detected or mismatched
        if not role or role not in all_roles:
            return render_template("ask_role_level.html", info=info, roles=all_roles)

        levels = get_levels_for_role(role)
        if not level or level not in levels:
            return render_template("ask_role_level.html", info=info, roles=[role], levels=levels)

        # Proceed with pipeline
        questions = get_questions_by_role_and_level(role, level)
        if not questions:
            return f"No questions found for {role} ({level})"

        prompt = format_prompt_for_granite(role, level, questions)
        response = get_gemini_response(prompt)

        return render_template("result.html", info=info, questions=questions, response=response)

    return render_template("index.html")


@app.route("/submit_manual", methods=["POST"])
def submit_manual():
    role = request.form.get("manual_role")
    level = request.form.get("manual_level")
    name = request.form.get("name")
    email = request.form.get("email")
    skills = request.form.get("skills").split(", ")

    levels = get_levels_for_role(role)
    if level not in levels:
        return f"❌ Level '{level}' not valid for role '{role}'."

    info = {"name": name, "email": email, "skills": skills, "role": role, "level": level}

    questions = get_questions_by_role_and_level(role, level)
    if not questions:
        return f"No questions found for {role} ({level})"

    prompt = format_prompt_for_granite(role, level, questions)
    response = get_gemini_response(prompt)

    return render_template("result.html", info=info, questions=questions, response=response)


if __name__ == "__main__":
    app.run(debug=True)
