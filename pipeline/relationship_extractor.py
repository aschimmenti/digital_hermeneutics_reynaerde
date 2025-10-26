# Work Schema Generator - Creates JSON schemas for works combining entities into graphs
# Takes input from entity_extractor.py output and generates two types of work representations:
# 1. Factual graph: describes the work and author using only factual data
# 2. Opinionated graph: describes the work and author using opinionated/interpretative data

import json
import os
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum
from openai import OpenAI

# Define entity types for type checking and relation extraction
# Imported from entity_extractor.py but excluding methodology and reference for relation extraction
ENTITY_TYPES = [
    "person",
    "role",
    "place",
    "work",
    "date",
    "historical_context",
    "organization",
    "language",
    "theory",
    "genre",
    "concept"
]

# Full entity types including those excluded from relation extraction
ALL_ENTITY_TYPES = ENTITY_TYPES + ["methodology", "reference"]

@dataclass
class WorkNode:
    id: str
    type: str  # work, author, place, organization, concept, date
    name: str
    confidence: float

@dataclass
class WorkRelation:
    source_id: str
    target_id: str
    relation_type: str  # authored_by, created_in, influenced_by, etc.
    properties: Dict[str, Any]
    confidence: float
    claim_type: str  # "established_fact" or "authorial_argument"

@dataclass
class WorkGraph:
    graph_type: str  # "factual" or "opinionated"
    nodes: List[WorkNode]
    relations: List[WorkRelation]
    metadata: Dict[str, Any]

@dataclass
class WorkSchemaResult:
    factual_graph: WorkGraph
    opinionated_graph: WorkGraph
    original_input_data: Dict[str, Any]
    source_entities: List[Dict[str, Any]]

class WorkSchemaGenerator:
    """
    Generates JSON schemas for works by combining extracted entities into graphs.
    Creates both factual and opinionated representations of works and authors.
    """
    
    def __init__(self, api_key: str = None):
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        

    def load_entity_extraction_result(self, file_path: str) -> Dict[str, Any]:
        """Load the output from entity_extractor.py."""
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def load_few_shot_relations(self, path: str) -> List[Dict[str, Any]]:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data.get("examples", []) if isinstance(data, dict) else []
    
    def _format_questions_and_answers(self, original_data: Dict[str, Any], source_answers: Dict[int, str]) -> str:
        """Format questions and answers with both question text and answer for better context."""
        # If no source answers, return a message indicating this
        if not source_answers:
            return "No source questions and answers available."
            
        # If no original data or no sections, just format the answers directly
        if not original_data or "sections" not in original_data:
            formatted_qa = []
            for qid, answer in source_answers.items():
                formatted_qa.append(f"Q{qid}: (Question text not available)\nA{qid}: {answer}")
            return "\n\n".join(formatted_qa)
        
        # If we have both original data and source answers, format them together
        formatted_qa = []
        for section_name, section_data in original_data["sections"].items():
            if "questions_and_answers" in section_data:
                for qa in section_data["questions_and_answers"]:
                    qid = qa["question_id"]
                    if qid in source_answers:
                        question = qa["question"]
                        answer = source_answers[qid]
                        formatted_qa.append(f"Q{qid}: {question}\nA{qid}: {answer}")
        
        # If we couldn't find any matching questions, fall back to just the answers
        if not formatted_qa:
            for qid, answer in source_answers.items():
                formatted_qa.append(f"Q{qid}: (Question text not available)\nA{qid}: {answer}")
                
        return "\n\n".join(formatted_qa)


    def generate_interpretation_layer(self, entities: List[Dict[str, Any]], source_answers: Dict[int, str], document_metadata: Dict[str, Any], original_data: Dict[str, Any] = None, few_shot_path: Optional[str] = None, few_shot_examples: Optional[List[Dict[str, Any]]] = None) -> WorkGraph:
        """Generate interpretation layer using existing entities from entity_extractor.py."""
        
        entities_text = "\n".join([
            f"- {entity['name']} ({entity['type']})"
            for entity in entities
        ])
        
        # Safely format authors for prompt (list -> "Family, Given; ...")
        def _format_authors(auth):
            if isinstance(auth, str):
                return auth
            if isinstance(auth, list):
                parts = []
                for a in auth:
                    fam = a.get("family_name", "").strip() if isinstance(a, dict) else ""
                    giv = a.get("given_name", "").strip() if isinstance(a, dict) else ""
                    if fam and giv:
                        parts.append(f"{fam}, {giv}")
                    elif fam:
                        parts.append(fam)
                    elif giv:
                        parts.append(giv)
                return "; ".join(parts) if parts else "Unknown"
            return "Unknown"

        _title = document_metadata.get('title', 'Unknown')
        _authors = _format_authors(document_metadata.get('authors', 'Unknown'))
        _date = document_metadata.get('date', 'Unknown')

        examples_text = ""
        if few_shot_examples is None and few_shot_path and os.path.exists(few_shot_path):
            try:
                few_shot_examples = self.load_few_shot_relations(few_shot_path)
            except Exception:
                few_shot_examples = None
        if few_shot_examples:
            chunks = []
            for ex in few_shot_examples:
                ex_md = ex.get("document_metadata", {}) or {}
                ex_entities = ex.get("entities", []) or []
                ex_context = (ex.get("context") or "").strip()
                ex_rels = ex.get("expected_relations", []) or []
                ent_lines = [f"- {e.get('name','')} ({e.get('type','')})" for e in ex_entities]
                block = [
                    "--- Few-shot example ---",
                    f"Title: {ex_md.get('title','')}",
                    f"Authors: {ex_md.get('authors','')}",
                    f"Date: {ex_md.get('date','')}",
                    "Entities:",
                    *(ent_lines or ["- "]),
                ]
                if ex_context:
                    block.append("Context:")
                    block.append(ex_context)
                if ex_rels:
                    block.append("Expected relations (JSON):")
                    try:
                        block.append(json.dumps(ex_rels, ensure_ascii=False))
                    except Exception:
                        pass
                chunks.append("\n".join(block))
            examples_text = "\n\n".join(chunks)
            prompt = f"""
The following document has to do with the authorship of the medieval text "Van den vos Reynaerde". The author discusses the authorship of this text and the cultural context surrounding its creation. In particular, there are two levels to distinguish for your task: 
- What we are talking about (entities, people, locations, places, organizations, etc.). This should reflect the state of things 'before' the authors' claims. 
- What the authors assert, claim, or argue about these entities (interpretation layer). Guidance examples are provided below in FEW-SHOT EXAMPLES. Use them as patterns for structuring nodes and relations with correct claim_type assignments.

FEW-SHOT EXAMPLES (guidance):
{examples_text}

DOCUMENT CONTEXT:
- Title: {_title}
- Authors: {_authors}
- Date: {_date}

EXTRACTED ENTITIES (from previous analysis):
{entities_text}

TASK: Create a Knowledge Graph about what {document_metadata.get('authors', 'Unknown')} argue(s) or express(es) in their work "{document_metadata.get('title', 'Unknown')}". Combine the given entities (nodes) with relations based on the source questions and answers below.

SOURCE QUESTIONS AND ANSWERS:
{self._format_questions_and_answers(original_data, source_answers) if original_data else chr(10).join([f"Q{{qid}}: {{answer}}" for qid, answer in source_answers.items()])}

INSTRUCTIONS:
1. Use ONLY the entities provided above - do not create new entities
2. Nodes should only have: id, type, name, confidence (NO properties or claim_type fields)
3. For each relation, specify claim_type as either:
   - "established_fact": Information presented as established, uncontested facts (e.g., "The Donation of Constantine exists as a document", "Lorenzo Valla was a Renaissance scholar", "Van den vos Reynaerde is a medieval work")
   - "authorial_argument": What the author actively argues, proposes, or claims (e.g., "Daele argues Willem van Boudelo created the work", "Peeters claims it was written in Land of Waas")

CRITICAL DISTINCTION:
- If the text presents something as established historical fact → established_fact
- If the text presents the author's interpretation, hypothesis, or argument → authorial_argument
- Biographical facts (birth, death dates) when uncontested → established_fact
- Authorship attributions that are argued/proposed → authorial_argument

4. For relation properties, include contextual information:
   - "asserted_by": Who makes this claim (use author name from document metadata, or "unknown_source" for established facts without specific attribution)
   - "evidence_type": Type of evidence (e.g., "ProsopographicAnalysis", "TextualAnalysis", "HistoricalRecords")
   - "certainty": Degree of certainty (e.g., "high", "medium", "low", "likely", "possibly")
   - "method": How the conclusion was reached (e.g., "ComparativeAnalysis", "SourceCriticism")

5. Focus on scholarly assertions, hypotheses, and interpretative claims about entities, especially authorship and biographical information
6. Extract authorship attributions, influence theories, methodological approaches
7. Map entity names to the exact names from the extracted entities list

ALLOWED RELATIONSHIP TYPES (use ONLY these):

CREATION RELATIONS:
- created_by: X was created by Y (authorship, production)
  Domain: [work] → Range: [person, organization]
  Example: Van den vos Reynaerde created_by Willem van Boudelo
  Note: this is valid only for works created, not for concepts or other 'created' relations.
  
- created_during: X was created during Y (temporal creation)
  Domain: [work] → Range: [date]
  Example: Van den vos Reynaerde created_during 1250
  
- created_at: X was created at Y (spatial creation)
  Domain: [work] → Range: [place]
  Example: Van den vos Reynaerde created_at Land of Waas

INFLUENCE RELATIONS:
- influenced_by: X was influenced by Y (authorial/cultural influence on creation)
  Domain: [work, person] → Range: [person, organization, historical_context, concept]
  Example: Van den vos Reynaerde influenced_by Cistercian culture
  Example: Willem van Boudelo influenced_by Jan van Dampierre
  NOTE: Use this for influences ON the work or person, not FOR what the work discusses

SPATIAL/TEMPORAL RELATIONS:
- located_in_space: X is spatially located in Y (current or historical location)
  Domain: [person, organization, work] → Range: [place]
  Example: Willem van Boudelo located_in_space Flanders
  
- located_in_time: X is temporally located in Y (contemporary with)
  Domain: [person, work, organization] → Range: [date]
  Example: Willem van Boudelo located_in_time 13th century

MEMBERSHIP RELATIONS:
- associated_with: X is a member of/part of Y (institutional membership)
  Domain: [person] → Range: [organization]
  Example: Willem van Boudelo associated_with counts of Flanders

REFERENCE RELATIONS (STRICT):
- refers_to: X refers to Y (ONLY for work-to-work, work-to-person, work-to-place, work-to-historical_context references WITHIN the medieval text itself)
  Domain: [work] → Range: [work, person, place, historical_context, organization]
  Example: Van den vos Reynaerde refers_to Reynardus Vulpes (another work)
  Example: Van den vos Reynaerde refers_to Bouchard van Avesnes (historical person mentioned IN the text)
  Example: Van den vos Reynaerde refers_to political conflicts (historical events discussed IN the text)
  
  CRITICAL: Do NOT use refers_to for:
  - Concepts as in themes
  - Methodologies (these describe the scholarly article, not the medieval work)
  - Literary analysis concepts (these describe the scholarly approach, not the medieval text's content)
  - Authorial influences (use influenced_by instead)

LINGUISTIC RELATIONS:
- speaks_language: X speaks language Y (linguistic competence of person)
  Domain: [person] → Range: [language]
  Example: Willem van Boudelo speaks_language Old French
  
- written_in_language: X is written in language Y (language of work)
  Domain: [work] → Range: [language]
  Example: Van den vos Reynaerde written_in_language Middle Dutch

CLASSIFICATION RELATIONS:
- has_genre: X belongs to genre Y (literary classification)
  Domain: [work] → Range: [genre]
  Example: Van den vos Reynaerde has_genre beast epic
  
- has_theme: X has theme Y (thematic content of work)
  Domain: [work] → Range: [concept]
  Example: Van den vos Reynaerde has_theme courtly culture
  Example: Van den vos Reynaerde has_theme feudalism
  NOTE: Use this for concepts/themes the work engages with, NOT for references to other works
  
- has_characteristic: X has characteristic Y (stylistic/physical features)
  Domain: [work] → Range: [concept]
  Example: Van den vos Reynaerde has_characteristic acrostic structure
  NOTE: Limited to literary features like writing style, rhyme scheme, physical manuscript features

BIOGRAPHICAL RELATIONS:
- has_occupation: X has occupation Y (professional role)
  Domain: [person] → Range: [role]
  Example: Willem van Boudelo has_occupation monk
  
- place_of_birth: X was born in Y
  Domain: [person] → Range: [place]
  
- date_of_birth: X was born on Y
  Domain: [person] → Range: [date]
  
- place_of_death: X died in Y
  Domain: [person] → Range: [place]
  
- date_of_death: X died on Y
  Domain: [person] → Range: [date]
  
- educated_at: X was educated at Y
  Domain: [person] → Range: [organization, place]
  
- lived_in: X lived in Y (residence/hometown)
  Domain: [person] → Range: [place]
  
- has_expertise_in: X has expertise in Y (scholarly specialization)
  Domain: [person] → Range: [concept, language]
  Example: Willem van Boudelo has_expertise_in Old French literature
  
- has_role: X has role Y in context Z
  Domain: [person] → Range: [role]
  Example: Willem van Boudelo has_role court poet
  NOTE: Must be connected to an activity or event context

RELATION SELECTION GUIDELINES:
1. For work content/themes → use has_theme
2. For work-to-work citations → use refers_to
3. For historical persons/events mentioned IN the medieval text → use refers_to
4. For influences ON creation → use influenced_by
5. For stylistic features → use has_characteristic
6. For authorship claims → use created_by
7. NEVER use refers_to for concepts, methodologies, or analytical frameworks

Generate nodes and relations representing the authors' interpretative claims about the provided entities.
Use ONLY the relationship types listed above with ONLY the given entities.
"""
#role => role va reificato come attività (type of Activity) 
#Activity => P2_has_type => Type / has_time_span nel caso in cui voglio contingentare la cosa nel tempo 

#speaks_language 

#educated_at => activity 
        # Save the full prompt for debugging
        debug_dir = "./debug"
        os.makedirs(debug_dir, exist_ok=True)
        debug_file = os.path.join(debug_dir, 'full_prompt_debug.txt')
        
        with open(debug_file, 'w', encoding='utf-8') as f:
            f.write("=== FULL PROMPT SENT TO MODEL ===\n\n")
            f.write(prompt)
            f.write("\n\n=== END OF PROMPT ===")
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert knowledge graph generator specializing in scholarly interpretations about medieval literature."},
                    {"role": "user", "content": prompt}
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "interpretation_graph",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "nodes": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "id": {"type": "string"},
                                            "type": {"type": "string", "enum": ENTITY_TYPES},
                                            "name": {"type": "string"},
                                            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0}
                                        },
                                        "required": ["id", "type", "name", "confidence"]
                                    }
                                },
                                "relations": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "source_id": {"type": "string"},
                                            "target_id": {"type": "string"},
                                            "relation_type": {"type": "string"},
                                            "properties": {"type": "object"},
                                            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                                            "claim_type": {"type": "string", "enum": ["established_fact", "authorial_argument"]}
                                        },
                                        "required": ["source_id", "target_id", "relation_type", "confidence", "claim_type"]
                                    }
                                }
                            },
                            "required": ["nodes", "relations"]
                        }
                    }
                },
                temperature=0.0,
                max_tokens=16300,
                top_p=1.0
            )
            
            result = json.loads(response.choices[0].message.content)
            
            # Add error handling for node creation
            nodes = []
            for node in result.get("nodes", []):
                try:
                    nodes.append(WorkNode(
                        id=node["id"],
                        type=node["type"],
                        name=node["name"],
                        confidence=node.get("confidence", 0.5)  # Default confidence if missing
                    ))
                except Exception as e:
                    print(f"Error creating node {node.get('id', 'unknown')}: {e}")
            
            # Add error handling for relation creation
            relations = []
            for rel in result.get("relations", []):
                try:
                    relations.append(WorkRelation(
                        source_id=rel["source_id"],
                        target_id=rel["target_id"],
                        relation_type=rel["relation_type"],
                        properties=rel.get("properties", {}),
                        confidence=rel.get("confidence", 0.5),  # Default confidence if missing
                        claim_type=rel.get("claim_type", "interpretation")  # Default claim_type if missing
                    ))
                except Exception as e:
                    print(f"Error creating relation from {rel.get('source_id', 'unknown')} to {rel.get('target_id', 'unknown')}: {e}")
            
            return WorkGraph(
                graph_type="interpretation_layer",
                nodes=nodes,
                relations=relations,
                metadata={"generation_method": "entity_based_interpretation", "source_entities_count": len(entities)}
            )
            
        except Exception as e:
            print(f"Error generating interpretation layer: {e}")
            print(f"Response content: {response.choices[0].message.content if 'response' in locals() else 'No response received'}")
            return WorkGraph("interpretation_layer", [], [], {"error": str(e)})

    def generate_work_schemas(self, entity_extraction_file: str, few_shot_path: Optional[str] = None) -> WorkSchemaResult:
        """Generate both factual and opinionated work schemas from entity extraction results."""
        
        # Load entity extraction data
        data = self.load_entity_extraction_result(entity_extraction_file)
        
        # Extract entities and source answers with proper error handling
        entities = []
        source_answers = {}
        
        # Extract entities
        if "extraction_metadata" in data and "entities" in data["extraction_metadata"]:
            entities = data["extraction_metadata"]["entities"]
        else:
            print(f"Warning: No entities found in {entity_extraction_file}")
            # Try to find entities in other locations
            if "entities" in data:
                entities = data["entities"]
        
        # Extract source answers
        if "extraction_metadata" in data and "source_answers" in data["extraction_metadata"]:
            source_answers = data["extraction_metadata"]["source_answers"]
        else:
            print(f"Warning: No source answers found in extraction_metadata for {entity_extraction_file}")
            # Try to find source answers in other locations
            if "source_answers" in data:
                source_answers = data["source_answers"]
            elif "answers" in data:
                source_answers = data["answers"]
        
        # Convert source_answers keys to integers if they're strings
        if source_answers and all(isinstance(k, str) for k in source_answers.keys()):
            source_answers = {int(k): v for k, v in source_answers.items()}
        
        # Extract document metadata with proper error handling
        document_metadata = {
            "title": "Unknown",
            "authors": "Unknown",
            "date": "Unknown"
        }
        
        # First try to get metadata from document_metadata (this is the primary location)
        if "document_metadata" in data:
            print(f"Found document metadata in 'document_metadata' key")
            document_metadata.update(data["document_metadata"])
        # If not found, try other possible locations
        elif "metadata" in data and any(k in data["metadata"] for k in ["title", "authors", "date"]):
            print(f"Found document metadata in 'metadata' key")
            # Only update with relevant fields
            for field in ["title", "authors", "date"]:
                if field in data["metadata"]:
                    document_metadata[field] = data["metadata"][field]
        
        # Extract title from filename if still not available
        if document_metadata["title"] == "Unknown":
            base_filename = os.path.basename(entity_extraction_file)
            document_metadata["title"] = os.path.splitext(base_filename)[0]
            
        print(f"Document metadata: {document_metadata}")

        
        # Debug print statements
        print(f"Found {len(entities)} entities and {len(source_answers)} source answers")
        if source_answers:
            print(f"Sample source answer keys: {list(source_answers.keys())[:5]}")
        
        # Format Q&A for debugging (full output)
        formatted_qa = self._format_questions_and_answers(data, source_answers)
        # Save full Q&A to debug file
        os.makedirs("./debug", exist_ok=True)
        qa_file = os.path.join("./debug", "full_qa.txt")
        try:
            with open(qa_file, "w", encoding="utf-8") as fqa:
                fqa.write(formatted_qa)
            print(f"Saved full Q&A to {qa_file}")
        except Exception as e:
            print(f"Could not write full Q&A file: {e}")
        # Print the full Q&A to console
        print("\nFULL Q&A:\n" + formatted_qa + "\n")
        
        if few_shot_path is None:
            default_fs = os.path.join(os.path.dirname(__file__), "few_shot_examples_relations.json")
            few_shot_path = default_fs if os.path.exists(default_fs) else None
        interpretation_layer = self.generate_interpretation_layer(entities, source_answers, document_metadata, data, few_shot_path=few_shot_path)
        
        # No facts layer - user will handle this
        facts_layer = None
        
        return WorkSchemaResult(
            factual_graph=facts_layer,
            opinionated_graph=interpretation_layer,
            original_input_data=data,
            source_entities=entities
        )

    def save_work_schemas(self, result: WorkSchemaResult, output_path: str):
        """Save work schema results combined with original input data to JSON file."""
        
        # Start with the original input data
        output_data = result.original_input_data.copy()
        
        # Add work schema metadata (interpretation layer only)
        output_data["work_schema_metadata"] = {
            "interpretation_layer": {
                "graph_type": result.opinionated_graph.graph_type,
                "description": "What the authors assert/claim about the entities",
                "nodes": [asdict(node) for node in result.opinionated_graph.nodes],
                "relations": [asdict(relation) for relation in result.opinionated_graph.relations],
                "metadata": result.opinionated_graph.metadata
            },
            "generation_summary": {
                "total_source_entities": len(result.source_entities),
                "interpretation_layer_nodes": len(result.opinionated_graph.nodes),
                "interpretation_layer_relations": len(result.opinionated_graph.relations)
            }
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

def main():
    """Main function to run the work schema generator."""
    generator = WorkSchemaGenerator()
    
    # Base directory for documents
    documents_dir = "./documents"
    
    # Ensure documents directory exists
    os.makedirs(documents_dir, exist_ok=True)
    
    # Find all document subdirectories
    document_dirs = [d for d in os.listdir(documents_dir) if os.path.isdir(os.path.join(documents_dir, d))]
    
    if not document_dirs:
        print(f"No document directories found in {documents_dir}")
        return
    
    total_processed = 0
    errors = []
    
    # Process each document directory
    for doc_dir in document_dirs:
        doc_path = os.path.join(documents_dir, doc_dir)
        entities_file = os.path.join(doc_path, "entities.json")
        relations_file = os.path.join(doc_path, "relations.json")
        
        # Check if entities.json exists
        if not os.path.exists(entities_file):
            print(f"No entities.json found in {doc_path}")
            continue
        
        try:
            print(f"Processing {entities_file}...")
            result = generator.generate_work_schemas(entities_file)
            generator.save_work_schemas(result, relations_file)
            
            print(f"Successfully generated interpretation layer:")
            print(f"- Interpretation layer: {len(result.opinionated_graph.nodes)} nodes, {len(result.opinionated_graph.relations)} relations")
            print(f"Results saved to: {relations_file}")
            
            total_processed += 1
            
        except Exception as e:
            error_msg = f"Error processing {entities_file}: {e}"
            print(error_msg)
            errors.append(error_msg)
    
    # Print summary
    print(f"\nProcessing complete. {total_processed} files processed.")
    if errors:
        print(f"{len(errors)} errors occurred:")
        for error in errors:
            print(f"- {error}")

if __name__ == "__main__":
    main()
