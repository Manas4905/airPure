import toml
import sys
from google import genai

try:
    key = toml.load(r'c:\Users\EKTA\Downloads\AI-Powered-Air-Purifier-Recommendation-main\AI-Powered-Air-Purifier-Recommendation-main\.streamlit\secrets.toml')['GEMINI_API_KEY']
    client = genai.Client(api_key=key, http_options={'api_version': 'v1alpha'})
    print("Testing API")
    r = client.models.generate_content(model='gemini-2.5-flash', contents='test')
    print("Response text:", getattr(r, 'text', 'No text attr'))
except Exception as e:
    print("ERROR:", e.__class__.__name__, str(e))
