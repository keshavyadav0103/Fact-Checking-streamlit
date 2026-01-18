import os
import json
import re
import requests
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

SERPER_API_KEY = os.getenv("SERPER_API_KEY")


def serper_search(query: str):
    if not SERPER_API_KEY:
        return {}

    try:
        r = requests.post(
            "https://google.serper.dev/search",
            headers={
                "X-API-KEY": SERPER_API_KEY,
                "Content-Type": "application/json",
            },
            json={"q": query, "num": 5},
            timeout=15,
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print("[SERPER FAILED]", e)
        return {}

def extract_json(text: str):
    """
    Extract first JSON object found in text.
    """
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("No JSON object found")
    return json.loads(match.group())

def verify_claim(claim: str) -> dict:
    try:
        search_results = serper_search(claim)

        snippets = []
        for r in search_results.get("organic", []):
            snippets.append(
                f"{r.get('title')} — {r.get('snippet')} ({r.get('link')})"
            )

        context = "\n".join(snippets) or "No reliable web results found."

        llm = ChatGoogleGenerativeAI(
            model="models/gemini-2.5-flash",
            temperature=0,
            api_key=os.getenv("GEMINI_API_KEY"),
        )

        prompt = ChatPromptTemplate.from_template("""
You are a strict JSON API.

TASK:
Verify the factual claim using the web data.

CLAIM:
{claim}

WEB DATA:
{context}

RULES:
- Respond with ONLY valid JSON
- No markdown
- No explanation outside JSON

JSON FORMAT:
{{
  "status": "Verified | Inaccurate | False",
  "explanation": "Short justification with source hints"
}}
""")

        chain = prompt | llm | StrOutputParser()
        raw = chain.invoke({"claim": claim, "context": context}).strip()

        # 🔒 Extract JSON even if Gemini adds text
        return extract_json(raw)

    except Exception as e:
        print("[VERIFICATION FALLBACK]", e)
        return {
            "status": "Unknown",
            "explanation": "Verification failed due to API limits or invalid model output."
        }
