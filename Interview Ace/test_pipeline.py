from resume_parser import extract_resume_info
from granite import (
    get_questions_by_role_and_level,
    get_all_roles,
    get_levels_for_role,
    format_prompt_for_granite
)
from gemini_api import get_gemini_response  # ✅ Use Gemini now

# 🔧 Change this if your resume file is named differently
RESUME_PATH = "Yash_Resume_Data_Science.pdf"

print("🔍 Parsing resume...\n")

# ✅ Extract info from resume
info = extract_resume_info(RESUME_PATH)

print("\n✅ Extracted from Resume:")
for key, value in info.items():
    print(f"{key.capitalize()}: {value}")

# ✅ Ask for role if not found
if info["role"] == "Not found":
    print("\n❗ Could not auto-detect role.")
    all_roles = get_all_roles()
    print("\nAvailable roles:", all_roles)
    info["role"] = input("Please enter your role manually: ")

# ✅ Ask for level if not found or not valid
levels = get_levels_for_role(info["role"])
if not levels:
    print(f"❌ Role '{info['role']}' not found in dataset.")
    exit()

print(f"\nAvailable levels for '{info['role']}': {levels}")
info["level"] = input("Please enter your level manually: ")

# ✅ Extract interview questions
questions = get_questions_by_role_and_level(info["role"], info["level"])

if not questions:
    print("❌ No questions found for the selected role and level.")
    exit()

# ✅ Format prompt for Gemini
prompt = format_prompt_for_granite(info["role"], info["level"], questions)
print("\n📩 Prompt sent to Gemini:\n")
print(prompt)

# ✅ Call Gemini model
print("\n⏳ Waiting for Gemini response...\n")
try:
    response = get_gemini_response(prompt)
    print("\n✅ Gemini Response:\n")
    print(response)
except Exception as e:
    print(f"\n❌ Error from Gemini: {e}")
