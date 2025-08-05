# gemini_api.py
import google.generativeai as genai

genai.configure(api_key="AIzaSyDvPt0numvbuoj1l1HLgurcjoojqhv9Wmc")  # Replace with your actual API key

def get_gemini_response(prompt):
    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        raise Exception(f"Gemini API failed: {e}")
