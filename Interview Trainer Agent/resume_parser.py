import fitz  # PyMuPDF
import re

def extract_resume_info(file_path):
    with fitz.open(file_path) as doc:
        text = "\n".join(page.get_text() for page in doc)

    email = extract_email(text)
    name = extract_name(text)
    skills = extract_skills(text)
    role = extract_role(text)

    return {
        "email": email,
        "name": name,
        "skills": skills,
        "role": role
    }

def extract_email(text):
    match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', text)
    return match.group(0) if match else "Not found"

def extract_name(text):
    lines = text.strip().split('\n')
    for line in lines:
        if re.match(r'^[A-Z][a-z]+\s[A-Z][a-z]+$', line.strip()):
            return line.strip()
    return lines[0] if lines else "Not found"

def extract_skills(text):
    skills_keywords = ['python', 'java', 'sql', 'excel', 'machine learning', 'data analysis', 'communication', 'deep learning', 'pandas', 'numpy', 'html']
    found_skills = set()
    for word in skills_keywords:
        if word.lower() in text.lower():
            found_skills.add(word.lower())
    return list(found_skills) if found_skills else ["Not found"]

def extract_role(text):
    roles = ['data scientist', 'software engineer', 'backend developer', 'frontend developer', 'qa engineer', 'data analyst', 'product manager', 'ux designer', 'machine learning engineer', 'data engineer']
    for role in roles:
        if role.lower() in text.lower():
            return role.title()
    return "Not found"
