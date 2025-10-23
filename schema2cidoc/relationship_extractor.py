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


    def generate_interpretation_layer(self, entities: List[Dict[str, Any]], source_answers: Dict[int, str], document_metadata: Dict[str, Any], original_data: Dict[str, Any] = None) -> WorkGraph:
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

        prompt = f"""
The following document has to do with the authorship of the medieval text "Van den vos Reynaerde". The author discusses the authorship of this text and the cultural context surrounding its creation. In particular, there are two levels to distinguish for your task: 
- What we are talking about (entities, people, locations, places, organizations, etc.). This should reflect the state of things 'before' the authors' claims. 
- What the authors assert, claim, or argue about these entities (interpretation layer)

For instance, you are given the following entities and the following text. 
Entities:
"Donation of Constantine", "Rome", "Constantine", "clergyman", "4th century", "8th century", "Lorenzo Valla", "De falso Constantini donatione". 
Text:
[...] the Donation of Constantine was a imperial bull made by Constantine in the 4th century, that a pope in the 8th century used it for political purposes. It was considered true until the Reinaissance. Lorenzo Valla, in his work "De falso Constantini donatione", discredited its authenticity, arguing that the Donation of Constantine is a forgery, and it was actually authored by some unknown clergyman of Rome in the 8th century. Lorenzo Valla analyzed the text and understood it was a forgery.

Therefore you would create nodes and relations with claim_type markers:

EXAMPLE OUTPUT:
{{
  "nodes": [
    {{
      "id": "donation_of_constantine",
      "type": "work",
      "name": "Donation of Constantine",
      "confidence": 0.95,
    }},
    {{
      "id": "constantine",
      "type": "person", 
      "name": "Constantine",
      "confidence": 0.9,
    }},
    {{
      "id": "lorenzo_valla",
      "type": "person",
      "name": "Lorenzo Valla", 
      "confidence": 0.95,
    }},
    {{
      "id": "unknown_clergyman",
      "type": "person",
      "name": "clergyman",
      "confidence": 0.7,
    }},
    {{
      "id": "textual_analysis",
      "type": "concept",
      "name": "Textual analysis",
      "confidence": 0.7,
    }}
  ],
  "relations": [
    {{
      "source_id": "donation_of_constantine",
      "target_id": "constantine", 
      "relation_type": "created_by",
      "properties": {{"asserted_by": "unknown_source"}},
      "confidence": 0.8,
      "claim_type": "established_fact"
    }},
    {{
      "source_id": "donation_of_constantine",
      "target_id": "4th_century", 
      "relation_type": "created_during",
      "properties": {{"asserted_by": "unknown_source"}},
      "confidence": 0.8,
      "claim_type": "established_fact"
    }},
    {{
      "source_id": "donation_of_constantine",
      "target_id": "unknown_clergyman",
      "relation_type": "created_by", 
      "properties": {{"asserted_by": "lorenzo_valla", "evidence_type": "textual_analysis"}},
      "confidence": 0.7,
      "claim_type": "authorial_argument"
    }},
    {{
      "source_id": "donation_of_constantine",
      "target_id": "8th_century",
      "relation_type": "created_during", 
      "properties": {{"asserted_by": "lorenzo_valla", "evidence_type": "textual_analysis"}},
      "confidence": 0.7,
      "claim_type": "authorial_argument"
    }}
  ]
}}

DOCUMENT CONTEXT:
- Title: {_title}
- Authors: {_authors}
- Date: {_date}

EXTRACTED ENTITIES (from previous analysis):
{entities_text}

TASK: Create a Knowledge Graph about the what the {document_metadata.get('authors', 'Unknown')} argue(s) express in their work "{document_metadata.get('title', 'Unknown')}". Combine the given entities (nodes) with the given relations below. 

SOURCE QUESTIONS AND ANSWERS:
{self._format_questions_and_answers(original_data, source_answers) if original_data else chr(10).join([f"Q{qid}: {answer}" for qid, answer in source_answers.items()])}

INSTRUCTIONS:
1. Use ONLY the entities provided above - do not create new entities
2. Nodes should only have: id, type, name, confidence (NO properties or claim_type fields)
3. For each relation, specify claim_type as either:
   - "established_fact": Information presented as established, uncontested facts (e.g., "Constantine was a 4th century emperor", "The Donation of Constantine exists as a document", "Lorenzo Valla was a Renaissance scholar")
   - "authorial_argument": What the author actively argues, proposes, or claims (e.g., "Valla argues the Donation is a forgery", "Valla proposes it was written by an 8th century clergyman")

CRITICAL DISTINCTION:
- If the text says "Constantine was emperor in the 4th century" → established_fact (presented as established)
- If the text says "Valla argues the Donation is a forgery" → authorial_argument (author's argument)
- If the text says "Valla claims the language proves 8th century origin" → authorial_argument (author's reasoning)
- If the text says "The Donation exists as a historical document" → established_fact (historical consensus)
- If the text says "Valla believes the evidence is conclusive" → authorial_argument (author's judgment)
4. For relation properties, include contextual information using entity IDs where applicable:
   - "asserted_by": Entity ID of who makes this claim (use document authors like "lorenzo_valla" or existing entity IDs. Use "unknown_source" only for argued claims that are discussed from the authors, but not "claimed" by the authors.)
   - "evidence_type": Type of evidence (e.g., "textual_analysis", "historical_records", "linguistic_analysis")
   - "certainty": Author's degree of certainty (e.g., "likely", "possibly", "definitively")
   - "method": How the conclusion was reached (e.g., "comparative_analysis", "source_criticism, etc")
5. Focus on scholarly assertions, hypotheses, and interpretative claims about these entities, especially documents and biographic information.
6. Extract authorship attributions, influence theories, methodological approaches
7. Map entity names to the exact names from the extracted entities list

ALLOWED RELATIONSHIP TYPES (use ONLY these):
- created_by: X was created by Y (authorship, production)
  Domain: [work] → Range: [person, organization]
- created_during: X was created during Y (temporal creation)
  Domain: [work] → Range: [date]
- created_in: X was created in Y (spatial creation)
  Domain: [work] → Range: [place]
- influenced_by: X was influenced by Y (author/cultural influence)
  Domain: [work, person] → Range: [person, organization, historical_context, place]
- located_in_space: X is located in space with Y (spatial relationship)
  Domain: [person, organization, work] → Range: [place]
- located_in_time: X is contemporary/located in time with Y (temporal relationship)
  Domain: [person, work, organization] → Range: [date]
- associated_with: X is a member of/part of Y (membership). 
  Domain: [person] → Range: [group, organization, institution]
- refers_to: X refers to Y (work references a concept/context/entity in discourse. An author refers THROUGH a work to something, not by him/herself)
  Domain: [work] → Range: [limited to a reference, e.g. person, organization, place, historical_context, concept]
- speaks_language: X speaks language Y (linguistic competence)
  Domain: [person] → Range: [language]
- written_in_language: X is written in language Y (work language)
  Domain: [work] → Range: [language]
- has_genre: X belongs to genre Y (literary classification)
  Domain: [work] → Range: [genre]
- has_occupation: X has occupation Y (professional role)
  Domain: [person] → Range: [role]
  A person has or had a occupation, e.g. as a scholar, monk, etc.
- place_of_birth: X was born in Y (place of birth)
  Domain: [person] → Range: [place]
  A person was born in a certain place. 
- date_of_birth: X was born on Y (date of birth)
  Domain: [person] → Range: [date]
  A person was born on a certain date. 
- place_of_death: X died in Y (place of death)
  Domain: [person] → Range: [place]
  A person died in a certain place. 
- date_of_death: X died on Y (date of death)
  Domain: [person] → Range: [date]
  A person died on a certain date. 
- educated_at: X was educated at Y (educational background)
  Domain: [person] → Range: [organization/place]
  A person was educated at a certain organization or place. 
- lived_in: X has lived/lives at Y (residence/living place/hometown)
  Domain: [person] → Range: [place]
  A person has lived/lives at a certain place. 
- has_expertise_in: X has expertise in Y (scholarly specialization)
  Domain: [person] → Range: [concept/language/practice]
  A person is expert in a language, in a practice, or in some other knowledge. 
- has_role: X has role Y
  Domain: [person] → Range: [role] + Scope: [activity/event] 
  A person has or had a role in some activity or event. Must be connected to an activity or event to be a valid relation. 
- has_characteristic: X has characteristic Y (work has a certain characteristic)
  Domain: [work] → Range: [concept (limited to literary features, e.g. writing style, rhyme, physical feature, genre...)]

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
                temperature=0.3
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

    def generate_work_schemas(self, entity_extraction_file: str) -> WorkSchemaResult:
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
        
        # Format Q&A for debugging
        formatted_qa = self._format_questions_and_answers(data, source_answers)
        print(f"Formatted Q&A sample (first 200 chars): {formatted_qa[:200]}...")
        
        # Generate interpretation layer using existing entities
        interpretation_layer = self.generate_interpretation_layer(entities, source_answers, document_metadata, data)
        
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
