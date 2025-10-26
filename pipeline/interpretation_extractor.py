import json
import os
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, asdict

from openai import OpenAI

# Controlled vocabularies provided by user
ALLOWED_INTERPRETATION_TYPES = [
    "philological_interpretation",
    "historical_interpretation",
    "linguistic_interpretation",
    "semiotic_interpretation",
    "paleographic_interpretation",
    "prosopographic_interpretation",
    "sociological_interpretation",
]

ALLOWED_INTERPRETATION_CRITERIA = [
    "diplomatic_interpretative_transcription",
    "literal_transcription",
    "hypothesis_based",
    "literature_based",
    "comparative_analysis",
    "authoritatively_based",
]

# Proper JSON Schema (Draft-07) for the extractor output
HICO_INTERPRETATION_SCHEMA: Dict[str, Any] = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "$id": "https://example.org/schemas/hico_interpretation.schema.json",
    "title": "HICO Interpretation Metadata",
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "interpretation_type": {
            "type": "array",
            "items": {"type": "string", "enum": ALLOWED_INTERPRETATION_TYPES},
            "minItems": 1,
            "maxItems": 3,
            "uniqueItems": True,
            "description": "hico:hasInterpretationType (allow 1-3)"
        },
        "interpretation_criteria": {
            "type": "array",
            "items": {"type": "string", "enum": ALLOWED_INTERPRETATION_CRITERIA},
            "minItems": 1,
            "uniqueItems": True,
            "description": "hico:hasInterpretationCriterion"
        },
        "certainty": {
            "type": "string",
            "enum": ["possibly", "likely", "highly_likely", "certain"],
        },
        "evidence_summary": {"type": "string"},
        "notes": {"type": "string"}
    },
    "required": ["interpretation_type", "interpretation_criteria"]
}


@dataclass
class InterpretationResult:
    interpretation_type: List[str]
    interpretation_criteria: List[str]
    certainty: Optional[str]
    evidence_summary: Optional[str]
    notes: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "interpretation_type": self.interpretation_type,
            "interpretation_criteria": self.interpretation_criteria,
            "certainty": self.certainty,
            "evidence_summary": self.evidence_summary,
            "notes": self.notes,
        }


class InterpretationExtractor:
    def __init__(self, api_key: str = None):
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

    def _format_qa(self, qa_path: str) -> str:
        if not qa_path or not os.path.exists(qa_path):
            return "No Q&A available."
        try:
            with open(qa_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return "No Q&A available."
        sections = []
        # support both rag_document_qa.json and auto_document_qa.json simple lists or sectioned
        if isinstance(data, dict) and "sections" in data:
            for sname, sdata in data["sections"].items():
                if "questions_and_answers" in sdata:
                    for qa in sdata["questions_and_answers"]:
                        q = qa.get("question") or qa.get("q") or ""
                        a = qa.get("answer") or qa.get("a") or ""
                        if q or a:
                            sections.append(f"Q: {q}\nA: {a}")
        elif isinstance(data, list):
            for qa in data:
                if isinstance(qa, dict):
                    q = qa.get("question") or qa.get("q") or ""
                    a = qa.get("answer") or qa.get("a") or ""
                    if q or a:
                        sections.append(f"Q: {q}\nA: {a}")
        return "\n\n".join(sections[:40]) or "No Q&A available."

    def _format_summaries(self, summaries_path: Optional[str]) -> str:
        if not summaries_path or not os.path.exists(summaries_path):
            return "No summaries available."
        try:
            with open(summaries_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return "No summaries available."
        parts: List[str] = []
        # Accept list of strings or dict with sections
        if isinstance(data, list):
            for item in data[:10]:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    parts.append(item.get("summary") or item.get("text") or "")
        elif isinstance(data, dict):
            # common shape: {"sections": { name: {"summary": str } }}
            sections = data.get("sections") or {}
            for name, sec in list(sections.items())[:10]:
                if isinstance(sec, dict):
                    parts.append(sec.get("summary") or sec.get("text") or "")
        text = "\n\n".join([p for p in parts if p]).strip()
        return text or "No summaries available."

    def _load_json(self, path: str) -> Dict[str, Any]:
        if not os.path.exists(path):
            return {}
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _format_authors(self, authors: Any) -> str:
        if isinstance(authors, str):
            return authors
        if isinstance(authors, list):
            parts = []
            for a in authors:
                if isinstance(a, dict):
                    fam = (a.get("family_name") or "").strip()
                    giv = (a.get("given_name") or "").strip()
                    if fam and giv:
                        parts.append(f"{fam}, {giv}")
                    elif fam:
                        parts.append(fam)
                    elif giv:
                        parts.append(giv)
            return "; ".join(parts) if parts else "Unknown"
        return "Unknown"

    def _build_prompt(self, metadata: Dict[str, Any], qa_text: str, summaries_text: str) -> str:
        title = metadata.get("title") or "Unknown"
        authors = self._format_authors(metadata.get("authors") or metadata.get("authors_list"))
        date = metadata.get("date") or "Unknown"
        return f"""
You analyze a scholarly document and must identify its HiCO interpretation metadata.

HiCO fields to output:
- interpretation_type (hico:hasInterpretationType): choose 1 to 3 from this controlled list only:
  {ALLOWED_INTERPRETATION_TYPES}
- interpretation_criteria (hico:hasInterpretationCriterion): choose one or more from this controlled list only:
  {ALLOWED_INTERPRETATION_CRITERIA}
- certainty: possibly, likely, highly_likely, certain
- evidence_summary: short rationale (1-3 sentences)

Document context:
- Title: {title}
- Authors: {authors}
- Date: {date}

Relevant Q&A snippets:
{qa_text}

Document summaries:
{summaries_text}

Task:
Return only a compact JSON with fields: interpretation_type (string), interpretation_criteria (array of strings), certainty (string), evidence_summary (string), notes (string, optional). Base your answer strictly on the context above.
""".strip()

    def extract(self, entities_path: str, relations_path: str, document_metadata_path: str, qa_path: Optional[str], summaries_path: Optional[str] = None) -> InterpretationResult:
        metadata = self._load_json(document_metadata_path)
        qa_text = self._format_qa(qa_path) if qa_path else "No Q&A available."
        summaries_text = self._format_summaries(summaries_path) if summaries_path else "No summaries available."
        prompt = self._build_prompt(metadata, qa_text, summaries_text)

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a humanities scholar skilled in HiCO ontology annotation."},
                {"role": "user", "content": prompt},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "hico_interpretation",
                    "schema": HICO_INTERPRETATION_SCHEMA
                }
            },
            temperature=0.2,
        )
        data = json.loads(response.choices[0].message.content)
        # Validate/filter to controlled values as a safety net
        raw_types = data.get("interpretation_type", [])
        if isinstance(raw_types, str):
            raw_types = [raw_types]
        itypes = [t for t in (raw_types or []) if t in ALLOWED_INTERPRETATION_TYPES]
        # enforce max 3 and default
        itypes = itypes[:3]
        if not itypes:
            itypes = ["philological_interpretation"] if ALLOWED_INTERPRETATION_TYPES else []
        icrit = [c for c in (data.get("interpretation_criteria", []) or []) if c in ALLOWED_INTERPRETATION_CRITERIA]
        return InterpretationResult(
            interpretation_type=itypes,
            interpretation_criteria=icrit,
            certainty=data.get("certainty"),
            evidence_summary=data.get("evidence_summary"),
            notes=data.get("notes"),
        )


def main():
    base = os.path.abspath(os.path.dirname(__file__))
    documents_dir = os.path.join(base, "documents")
    input_dir = os.path.join(base, "input")
    extractor = InterpretationExtractor()

    for doc_id in os.listdir(documents_dir):
        doc_path = os.path.join(documents_dir, doc_id)
        if not os.path.isdir(doc_path):
            continue
        entities_path = os.path.join(doc_path, "entities.json")
        relations_path = os.path.join(doc_path, "relations.json")
        if not os.path.exists(entities_path) or not os.path.exists(relations_path):
            continue
        document_metadata_path = os.path.join(input_dir, doc_id, "document_metadata.json")
        qa_path = os.path.join(input_dir, doc_id, "rag_document_qa.json")
        if not os.path.exists(qa_path):
            qa_path = os.path.join(input_dir, doc_id, "auto_document_qa.json")
        # Summaries path (optional)
        summaries_path = os.path.join(input_dir, doc_id, "document_summaries.json")
        if not os.path.exists(summaries_path):
            summaries_path = None
        try:
            result = extractor.extract(entities_path, relations_path, document_metadata_path, qa_path, summaries_path)
            out_path = os.path.join(doc_path, "interpretation.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump({"hico": result.to_dict()}, f, indent=2, ensure_ascii=False)
            print(f"Saved {out_path}")
        except Exception as e:
            print(f"Error for {doc_id}: {e}")


if __name__ == "__main__":
    main()
