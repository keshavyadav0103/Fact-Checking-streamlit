import os
import json
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

def extract_claims(text: str) -> str:
    """
    Always returns VALID JSON.
    If Gemini fails, returns [].
    """
    try:
        llm = ChatGoogleGenerativeAI(
            model="models/gemini-2.5-flash",
            temperature=0,
            api_key=os.getenv("GEMINI_API_KEY"),
        )

        prompt = ChatPromptTemplate.from_template("""
Extract all factual, verifiable claims from the text below.

Only include claims involving:
- statistics
- dates
- prices
- financial figures
- measurable technical specs

Return JSON ARRAY ONLY.
NO explanation. NO markdown.

FORMAT:
[
  {{
    "claim": "...",
    "type": "..."
  }}
]

TEXT:
{text}
""")

        chain = prompt | llm | StrOutputParser()
        output = chain.invoke({"text": text}).strip()

        # Validate JSON strictly
        json.loads(output)
        return output

    except Exception as e:
        print("[CLAIM_EXTRACTOR FALLBACK]", e)
        return json.dumps([])
