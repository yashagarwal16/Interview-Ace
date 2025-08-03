from flask import Flask, render_template, request, redirect
from werkzeug.utils import secure_filename
import os

from resume_parser import extract_resume_info
from granite import get_questions_by_role_and_level, get_levels_for_role, get_all_roles, format_prompt_for_granite
from ibm_granite_api import get_granite_response

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["resume"]
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)

            info = extract_resume_info(filepath)
            role = info.get("role")
            level = info.get("level")

            if not role or not level:
                all_roles = get_all_roles()
                return render_template("ask_role_level.html", roles=all_roles, info=info, resume_path=filepath)

            return redirect(f"/result?role={role}&level={level}&resume_path={filepath}")

    return render_template("upload.html")

@app.route("/manual", methods=["POST"])
def submit_manual():
    role = request.form["role"]
    level = request.form["level"]
    resume_path = request.form["resume_path"]
    return redirect(f"/result?role={role}&level={level}&resume_path={resume_path}")

@app.route("/result")
def result():
    role = request.args.get("role")
    level = request.args.get("level")
    resume_path = request.args.get("resume_path")

    info = extract_resume_info(resume_path)
    matched_role = next((r for r in get_all_roles() if role.lower() in r.lower()), role)
    questions = get_questions_by_role_and_level(matched_role, level)

    if not questions:
        return f"❌ No questions found for {matched_role} - {level}"

    prompt = format_prompt_for_granite(matched_role, level, questions)
    try:
        response = get_granite_response(prompt)
    except Exception as e:
        response = f"❌ Granite API error: {str(e)}"

    return render_template("result.html", info=info, role=matched_role, level=level, questions=questions, response=response)

if __name__ == "__main__":
    app.run(debug=True)
