from resume_parser import extract_resume_info
from granite import get_questions_by_role_and_level, get_levels_for_role, get_all_roles, format_prompt_for_granite

from ibm_granite_api import get_granite_response

# ✅ Step 1: Extract details from the resume
RESUME_PATH = "Yash_Resume_Data_Science.pdf"  # Change as needed
print("\n🔍 Parsing resume...")
info = extract_resume_info(RESUME_PATH)

# ✅ Step 2: Show extracted info
print("\n✅ Extracted from Resume:")
for key, val in info.items():
    print(f"{key.capitalize()}: {val}")

role = info.get("role")
level = info.get("level")

# ✅ Step 3: Ask manually if role/level not found
got_role = True
if not role or role == "Not found":
    got_role = False
    print("\nCould not auto-detect role. Please enter your role manually:", end=" ")
    role = input().strip()

# Suggest closest matching role
all_roles = get_all_roles()
matched_role = next((r for r in all_roles if role.lower() in r.lower()), None)
if not matched_role:
    print(f"\n❌ Role not found in system.")
    exit()

# ✅ Step 4: Ask for level if missing or not in list
available_levels = get_levels_for_role(matched_role)
if not level or level.lower() not in [l.lower() for l in available_levels]:
    print(f"\nAvailable levels for '{matched_role}': {available_levels}")
    print("Please enter your level manually:", end=" ")
    level = input().strip().capitalize()

# ✅ Step 5: Get questions
questions = get_questions_by_role_and_level(matched_role, level)
if not questions:
    print("\n❌ No questions found for that role/level.")
    exit()

# ✅ Step 6: Format prompt
prompt = format_prompt_for_granite(matched_role, level, questions)
print("\n📩 Prompt sent to Granite:\n")
print(prompt)

# ✅ Step 7: Send to Granite
print("\n⏳ Waiting for Granite response...")
try:
    response = get_granite_response(prompt)
    print("\n✅ Granite Response:\n")
    print(response)
except Exception as e:
    print(f"\n❌ Error from Granite: {e}")
