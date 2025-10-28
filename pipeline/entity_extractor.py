#factual information extractor about entities, their roles, and their relationships
#the first step just uses the json schema to extract each entity information and distinguishes between what's 'factual', what's 'opinionated' and what's 'methodological'
#the second step aligns the results with CIDOC-CRM properties and creates a CIDOC-CRM graph


#input: the auto_document_qa.json or rag_document_qa.json from the previous steps. 
#the script can process a single file or all files in the input directory

import json
import os
import glob
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import argparse

# Define entity types for type checking and prompt formatting
ENTITY_TYPES = [
    "person",
    "reference", 
    "role",
    "place",
    "work",
    "occupation",
    "date",
    "historical_context",
    "organization",
    "language",
    "methodology",
    "genre",
    "concept",
    "event",
    "group",
    "activity",
    "characteristic",
    "theme"
]

from dataclasses import dataclass, asdict
from enum import Enum
from openai import OpenAI

@dataclass
class ExtractedEntity:
    name: str
    type: str  # person, place, work, date, organization, concept
    context: str  # surrounding text where this entity was found
    confidence: float  # 0.0 to 1.0

@dataclass
class ExtractionResult:
    entities: List[ExtractedEntity]
    source_question_ids: List[int]
    source_answers: Dict[int, str]
    original_input_data: Dict[str, Any]  # Store the original input file data
    document_metadata: Dict[str, Any] = None  # Store document metadata from QA file

class FirstExtractor:
    """
    Extracts entities, locations, works, and dates from auto_document_qa.json
    using GPT-4o-mini with JSON schema for structured output.
    """
    
    def __init__(self, api_key: str = None, input_metadata_file: str = None):
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        
        # Use a default path that works on both Windows and Linux
        default_metadata_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "input.json")
        self.input_metadata_file = input_metadata_file or default_metadata_path
        self.document_metadata = self._load_document_metadata()
        
        # JSON schema for structured entity extraction
        self.extraction_schema = {
            "type": "object",
            "properties": {
                "entities": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string",
                                "description": "The exact name of the entity as it appears in the text"
                            },
                            "type": {
                                "type": "string",
                                "enum": ENTITY_TYPES,
                                "description": "The type of entity"
                            },
                            "context": {
                                "type": "string",
                                "description": "The surrounding text where this entity was mentioned"
                            },
                            "confidence": {
                                "type": "number",
                                "minimum": 0.0,
                                "maximum": 1.0,
                                "description": "Confidence score for this extraction (0.0 to 1.0)"
                            }
                        },
                        "required": ["name", "type", "context", "confidence"],
                        "additionalProperties": False
                    }
                }
            },
            "required": ["entities"],
            "additionalProperties": False
        }
        
        self.extraction_prompt_template = """
You are an expert in extracting structured information from academic texts about medieval literature and history.

Your task is to extract entities from the given text and categorize them by type.

DOCUMENT CONTEXT:
- Title: {title}
- Authors: {authors}
- Date: {date}
- Journal/Publisher: {journal}

IMPORTANT: Do not extract the document title "{title}" or authors "{authors}" as entities since they are metadata about the document itself.

ENTITY TYPES:
- person: Individual people (people names, historical figures, mentioned people)
  Examples: Willem van Boudelo, Jacob van Maerlant, Willem, Margareta of Flanders, Margareta (not fictional characters)
- reference: an entity of type person or a work who is specifically cited as a reference in support of one's argument. For instance, "author also mentions [person] as a support to this opinion" or "author x cites [work]". The reference should be made by {authors} to be a valid reference. 
- role: Professional roles, occupations, social positions, functions
  Examples: author, cleric, monk, scholar, nobleman, patrician, scribe, translator, abbot. The role SHOULD NOT be the {authors} role(s). 
- place: Real locations, regions, courts, monasteries, cities, countries...
  Examples: Rome, Constantinople, Ghent, Flanders, Land of Waas, Hulst, Cistercian monastery. 
- work: Literary works, documents, texts, manuscripts, chronicles
  Examples: Van den vos Reynaerde, the Canterbury Tales, the Divine Comedy, Roman de Renart. Works cited as support for a claim by {authors} should be instead 'reference'. (e.g. Lorenzo Valla discussing the Donation of Constantine is 'work', if Lorenzo Valla cites e.g. Jacopo da Velletri as support for his claim is 'reference').
- date: Time periods, centuries, years, specific dates, temporal spans
  Examples: 13th century, 1260, mid-13th century, around 1190, 1248-1263, medieval period
  Format guidelines: Use specific years when mentioned (e.g. "1260"), centuries as "Xth century", 
  periods as "early/mid/late Xth century", ranges as "YYYY-YYYY"
- historical_context: a period of time, a context, or a broader cultural or historical framework, movement or theme.
  Examples: Medieval period, Flemish culture, Christian culture, the Black Death, the 30 years war, the late Middle Ages, Renaissance
- organization: Religious orders, courts, institutions, social groups
  Examples: Cistercian order, grafelijke hof (count's court), Flemish nobility, urban patriciate
- language: a mentioned language name
  Examples: French, Latin, Dutch
- methodology: a mentioned scholarly approach or methodology. This can only be used for methodologies attributed to {authors}. 
  Examples: authorship attribution, literary analysis
- genre: a mentioned genre of literature
  Examples: romance, fabliau, chronicle, satire
- concept: any concept that is mentioned as relevant that does not fall in any other classes above
  Examples: courtly love, chivalrism, profanity, research
- event: a mentioned event
  Examples: the Battle of Hastings, the coronation of Charles V
- group: a mentioned group
  Examples: the nobility, the clergy, the bourgeoisie
- activity: a mentioned activity
  Examples: writing, reading, painting, sculpting
- characteristic: a mentioned characteristic
  Examples: a certain style, a certain theme, a certain subject
- theme: a mentioned theme
  Examples: courtly love, chivalrism, profanity, feudalism... Any concept that is referred to as a theme being discussed 


INSTRUCTIONS:
1. Extract ALL relevant entities from the text content
2. For each entity, provide the exact name as it appears
3. Classify each entity into one of the types above
4. Include sufficient context (surrounding text) to understand the entity's role
5. Assign confidence scores based on how clearly the entity is identified and categorized
6. Focus on entities related to medieval literature, authorship, historical context, and scholarly analysis
7. Exclude the document's own metadata (title, authors) from extraction
8. This is the list of entity types you can use:
{entity_types}
9. Avoid extracting entities that are the same string of the given type, e.g. "historical context" as type "historical_context".

Text to analyze:
"""

    def _load_document_metadata(self) -> Dict[str, Any]:
        """Load document metadata from input.json file and normalize for prompts."""
        try:
            with open(self.input_metadata_file, 'r', encoding='utf-8') as f:
                input_data = json.load(f)

            def _format_authors(auth):
                # Accept string or list[{family_name, given_name}]
                if isinstance(auth, str):
                    return auth
                if isinstance(auth, list):
                    parts = []
                    for a in auth:
                        fam = a.get("family_name", "").strip()
                        giv = a.get("given_name", "").strip()
                        if fam and giv:
                            parts.append(f"{fam}, {giv}")
                        elif fam:
                            parts.append(fam)
                        elif giv:
                            parts.append(giv)
                    return "; ".join(parts) if parts else "Unknown"
                return "Unknown"

            def _normalize(meta: Dict[str, Any]) -> Dict[str, Any]:
                # Deep copy to avoid mutating caller
                m = json.loads(json.dumps(meta))
                # Preserve original authors list if present
                if isinstance(m.get("authors"), list):
                    m["authors_list"] = m["authors"]
                # Compute flat authors string
                m["authors"] = _format_authors(m.get("authors"))

                doc_type = (m.get("type") or "").lower()
                container = m.get("container", {}) if isinstance(m.get("container"), dict) else {}

                # Journal/Book title flatten
                journal_title = m.get("journal")
                if not journal_title:
                    if container:
                        journal_title = container.get("title")
                # Choose journal priority, else publisher
                m["journal"] = journal_title or m.get("publisher") or container.get("publisher") or "Unknown"

                # Publisher flatten
                if not m.get("publisher") and container.get("publisher"):
                    m["publisher"] = container.get("publisher")

                # Volume / Series volume
                if not m.get("volume"):
                    if doc_type == "journal_article":
                        m["volume"] = container.get("volume")
                    elif doc_type == "book_chapter":
                        series = container.get("series", {}) if isinstance(container.get("series"), dict) else {}
                        m["volume"] = (series.get("volume") or container.get("volume"))

                # Pages for chapters
                if doc_type == "book_chapter":
                    if container.get("start_page"):
                        m["start_page"] = container.get("start_page")
                    if container.get("end_page"):
                        m["end_page"] = container.get("end_page")
                    # Backward compatibility if fields named differently
                    m["start_page"] = m.get("start_page") or m.get("startingPage") or m.get("prism:startingPage") or m.get("start_page")
                    m["end_page"] = m.get("end_page") or m.get("endingPage") or m.get("prism:endingPage") or m.get("end_page")

                return m

            # Create a mapping of file_id to normalized metadata
            metadata_map = {}
            for file_info in input_data["files"]:
                raw_meta = file_info.get("document_metadata", {})
                metadata_map[file_info["file_id"]] = _normalize(raw_meta)

            return metadata_map
        except Exception as e:
            print(f"Warning: Could not load metadata from {self.input_metadata_file}: {e}")
            return {}
    
    def load_document_qa(self, file_path: str) -> Dict[str, Any]:
        """Load the auto_document_qa.json file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _get_document_id_from_path(self, file_path: str) -> str:
        """Extract document ID from the file path."""
        # Extract from path like /output/daele_2005/auto_document_qa.json
        import os
        parent_dir = os.path.basename(os.path.dirname(file_path))
        return parent_dir
    
    def _format_extraction_prompt(self, document_id: str) -> str:
        """Format the extraction prompt with document metadata."""
        metadata = self.document_metadata.get(document_id, {})
        
        return self.extraction_prompt_template.format(
            title=metadata.get("title", "Unknown"),
            authors=metadata.get("authors", "Unknown"),
            date=metadata.get("date", "Unknown"),
            journal=metadata.get("journal", metadata.get("publisher", "Unknown")),
            entity_types=", ".join(ENTITY_TYPES)
        )

    def extract_entities_from_text(self, text: str, question_id: int, document_id: str) -> List[ExtractedEntity]:
        """Extract entities using GPT-4o-mini with structured JSON output."""
        try:
            formatted_prompt = self._format_extraction_prompt(document_id)
            
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": formatted_prompt
                    },
                    {
                        "role": "user",
                        "content": text
                    }
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "entity_extraction",
                        "schema": self.extraction_schema
                    }
                },
                temperature=0.1
            )
            
            # Parse the JSON response
            result = json.loads(response.choices[0].message.content)
            entities = []
            
            for entity_data in result["entities"]:
                entity = ExtractedEntity(
                    name=entity_data["name"],
                    type=entity_data["type"],
                    context=entity_data["context"],
                    confidence=entity_data["confidence"]
                )
                entities.append(entity)
            
            return entities
            
        except Exception as e:
            print(f"Error in LLM extraction: {e}")
            return []

    def extract_from_all_questions(self, file_path: str) -> ExtractionResult:
        """Extract entities from all questions (1, 2, 3) in the auto_document_qa.json file."""
        data = self.load_document_qa(file_path)
        document_id = self._get_document_id_from_path(file_path)
        
        all_entities = []
        source_answers = {}
        processed_questions = []
        
        # Process questions 1, 2, and 3
        for section_name, section_data in data["sections"].items():
            for qa in section_data["questions_and_answers"]:
                question_id = qa["question_id"]
                if question_id in [1, 2, 3]:
                    answer_text = qa["answer"]
                    entities = self.extract_entities_from_text(answer_text, question_id, document_id)
                    
                    all_entities.extend(entities)
                    source_answers[question_id] = answer_text
                    processed_questions.append(question_id)
        
        if not processed_questions:
            raise ValueError("No questions with IDs 1, 2, or 3 found in the document")
        
        # Remove duplicate entities (same name and type)
        unique_entities = self._deduplicate_entities(all_entities)
        
        # Extract document metadata from the QA file if available
        qa_document_metadata = data.get("document_metadata", {})
        
        # Print a warning if document_metadata is missing or empty
        if not qa_document_metadata:
            print(f"Warning: No document_metadata found in {file_path}. Output will have empty document_metadata.")
        
        return ExtractionResult(
            entities=unique_entities,
            source_question_ids=sorted(processed_questions),
            source_answers=source_answers,
            original_input_data=data,
            document_metadata=qa_document_metadata
        )
    
    def _deduplicate_entities(self, entities: List[ExtractedEntity]) -> List[ExtractedEntity]:
        """Remove duplicate entities, keeping the one with highest confidence."""
        seen = {}
        for entity in entities:
            key = (entity.name.lower(), entity.type)
            if key not in seen or entity.confidence > seen[key].confidence:
                seen[key] = entity
        return list(seen.values())

    def save_extraction_result(self, result: ExtractionResult, output_path: str):
        """Save extraction results combined with original input data to JSON file."""
        # Start with the original input data
        output_data = result.original_input_data.copy()
        
        # Ensure document metadata is preserved from the QA file
        # Always include document_metadata, even if empty
        output_data["document_metadata"] = result.document_metadata or {}
        
        # Add extraction metadata to the combined output
        output_data["extraction_metadata"] = {
            "source_question_ids": result.source_question_ids,
            "total_entities": len(result.entities),
            "entity_types": list(set(e.type for e in result.entities)),
            "entities": [asdict(entity) for entity in result.entities],
            "source_answers": result.source_answers,
            "extraction_method": "llm_structured_extraction",
            "extractor_version": "1.0"
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

def find_qa_files(input_dir: str) -> List[Tuple[str, str]]:
    """Find all rag_document_qa.json and auto_document_qa.json files in subdirectories.
    
    Returns a list of tuples (document_name, file_path)
    """
    qa_files = []
    
    # Convert to Path object for easier path manipulation
    input_path = Path(input_dir)
    
    # Check if input_dir exists and is a directory
    if not input_path.is_dir():
        raise ValueError(f"Input directory does not exist: {input_dir}")
    
    # Look for subdirectories
    for subdir in input_path.iterdir():
        if subdir.is_dir():
            document_name = subdir.name
            
            # Look for rag_document_qa.json first, then auto_document_qa.json
            rag_file = subdir / "rag_document_qa.json"
            auto_file = subdir / "auto_document_qa.json"
            
            if rag_file.exists():
                qa_files.append((document_name, str(rag_file)))
            elif auto_file.exists():
                qa_files.append((document_name, str(auto_file)))
    
    return qa_files

def ensure_output_dir(output_dir: str) -> None:
    """Create output directory if it doesn't exist."""
    os.makedirs(output_dir, exist_ok=True)

def process_single_file(extractor: FirstExtractor, input_file: str, output_file: str) -> None:
    """Process a single QA file and save the results."""
    try:
        # Ensure output directory exists
        output_dir = os.path.dirname(output_file)
        ensure_output_dir(output_dir)
        
        result = extractor.extract_from_all_questions(input_file)
        
        # Debug output for document_metadata
        if result.document_metadata:
            print(f"Document metadata found: {list(result.document_metadata.keys())}")
        else:
            print(f"Warning: No document metadata found in {input_file}")
            
        extractor.save_extraction_result(result, output_file)
        
        print(f"Successfully extracted {len(result.entities)} entities from questions {result.source_question_ids}")
        print(f"Results saved to: {output_file}")
        
        # Print summary
        entity_types = {}
        for entity in result.entities:
            entity_types[entity.type] = entity_types.get(entity.type, 0) + 1
        
        print("\nEntity types found:")
        for etype, count in entity_types.items():
            print(f"  {etype}: {count}")
            
    except Exception as e:
        print(f"Error processing {input_file}: {e}")

def process_all_files(extractor: FirstExtractor, input_dir: str, output_base_dir: str) -> None:
    """Process all QA files in the input directory and save results to output directories."""
    qa_files = find_qa_files(input_dir)
    
    if not qa_files:
        print(f"No QA files found in {input_dir}")
        return
    
    print(f"Found {len(qa_files)} QA files to process")
    
    for document_name, file_path in qa_files:
        print(f"\nProcessing {document_name} from {file_path}")
        
        # Create output directory for this document in the documents folder
        output_dir = os.path.join("documents", document_name)
        ensure_output_dir(output_dir)
        
        # Set output file path
        output_file = os.path.join(output_dir, "entities.json")
        
        # Process the file
        process_single_file(extractor, file_path, output_file)

def main():
    """Main function to run the entity extractor."""
    parser = argparse.ArgumentParser(description='Extract entities from QA documents')
    parser.add_argument('--input', '-i', 
                       help='Path to the QA JSON file or input directory containing document subdirectories',
                       required=False)
    parser.add_argument('--output', '-o',
                       help='Output path for extracted entities JSON file or base directory for multiple files. Default: schema2cidoc/entities_outputs',
                       default=None)
    parser.add_argument('--metadata-file', '-m',
                       help='Path to input.json metadata file',
                       default=None)
    parser.add_argument('--process-all', '-a', action='store_true',
                       help='Process all documents in the input directory')
    
    args = parser.parse_args()
    
    # Set default paths based on current environment
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Determine input path
    if not args.input:
        if args.process_all:
            # Process all QA files produced by the pipeline retriever
            input_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "documents")
        else:
            input_path = os.path.join(base_dir, "output", "daele_2005", "rag_document_qa.json")
    else:
        input_path = args.input
    
    # Determine output path
    if not args.output:
        # Default output to documents folder
        if args.process_all or os.path.isdir(input_path):
            output_base_dir = "documents"
        else:
            # For single file processing, extract document name from input path
            doc_name = os.path.basename(os.path.dirname(input_path))
            output_base_dir = os.path.join("documents", doc_name, "entities.json")
    else:
        output_base_dir = args.output
    
    # Determine metadata file (prefer pipeline/input.json when running here)
    metadata_file = (
        args.metadata_file
        or os.path.join(os.path.dirname(os.path.abspath(__file__)), "input.json")
    )
    
    # Initialize extractor with metadata file
    extractor = FirstExtractor(input_metadata_file=metadata_file)
    
    # Process files
    if args.process_all or os.path.isdir(input_path):
        # Process all files in directory
        if not os.path.isdir(output_base_dir):
            ensure_output_dir(output_base_dir)
        
        print(f"Processing all documents in {input_path}")
        print(f"Results will be saved to documents/<document_name>/entities.json")
        
        process_all_files(extractor, input_path, output_base_dir)
    else:
        # Process single file
        print(f"Processing single file: {input_path}")
        print(f"Results will be saved to: {output_base_dir}")
        
        # For single file, extract document name from input path
        doc_name = os.path.basename(os.path.dirname(input_path))
        output_file = os.path.join("documents", doc_name, "entities.json")
        
        process_single_file(extractor, input_path, output_file)

if __name__ == "__main__":
    main()