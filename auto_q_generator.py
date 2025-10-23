import json
import os
from typing import List, Dict, Any
import openai
from summarizer import summarize_document
from utils import build_document_metadata_string

def generate_questions_from_summary(
    cumulative_summary: str,
    section_summaries: List[Dict],
    document_metadata: str,
    openai_client
) -> List[str]:
    """
    Generate 3 contextualized questions based on document summaries using OpenAI tool calling.
    
    Args:
        cumulative_summary: The overall document summary
        section_summaries: List of section-specific summaries
        document_metadata: Document metadata string
        openai_client: OpenAI client instance
    
    Returns:
        List of 3 generated questions
    """
    
    # Create the tool schema for structured question generation
    question_schema = {
        "type": "function",
        "function": {
            "name": "generate_document_questions",
            "description": "Generate 3 contextualized questions about a document based on its summary",
            "parameters": {
                "type": "object",
                "properties": {
                    "questions": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        },
                        "minItems": 3,
                        "maxItems": 3,
                        "description": "Array of exactly 3 questions following the 3-layer template"
                    }
                },
                "required": ["questions"]
            }
        }
    }
    
    # Build context from summaries
    section_context = "\n".join([
        f"Section {s['section_number']}: {s['section_title']}\nSummary: {s['summary']}\n"
        for s in section_summaries
    ])
    
    prompt = f"""
{document_metadata}

The task is to generate three questions, that will be used aftwerwards to generate a nanopublication. The nanopublication relies on three levels: the factual level (which document, entities, characters, etc are presented in the discourse), the opinion level (what is the author's opinion about the subject), and the methodological level (what methods, theories, etc are used to support the claims made by the author).
You are tasked with generating exactly 3 "contextualized" questions about this document based on its summaries. Rely on the title and on the summaries to generate the questions, by understanding what's the fundamental claim made by the authors in the given publication. 
The questions should follow the 3-layer framework:

Layer 1: Subject & Entities - What is the subject of the document? Which entities, artefacts, or objects does the author analyze, comment on, or interpret?
Layer 2: Author Opinion & Intent - What is the author's opinion about the subject(s)/entities? What answers, hypotheses, or interpretations is the author proposing?
Layer 3: Methodology & Evidence - Which disciplines, techniques, and methods do the authors use to support their claims? What is their degree of certainty and what evidence do they provide?

CUMULATIVE SUMMARY:
{cumulative_summary}

SECTION SUMMARIES:
{section_context}


EXAMPLES OF GOOD QUESTIONS:

For a document about the religious symbolism in Caravaggio's "The Calling of Saint Matthew":
Layer 1: "What is the subject of the document regarding Caravaggio's 'The Calling of Saint Matthew'? Which specific paintings, people, objects, and symbols does the author analyze in this painting?"

Layer 2: "What is the author's opinion about the religious symbolism in Caravaggio's 'The Calling of Saint Matthew'? How does this interpretation differ from previous scholarly views on Caravaggio's religious iconography?"

Layer 3: "What art historical methodologies and visual analysis techniques does the author employ to support their claims about the painting's spiritual symbolism? What degree of certainty do they express about their iconographic interpretations?"

For a document titled "The Unreliable Narrator in Charlotte Perkins Gilman's 'The Yellow Wallpaper': A Feminist Reading of Madness and Agency":

Layer 1: "What is the subject of the article "The Unreliable Narrator in Charlotte Perkins Gilman's 'The Yellow Wallpaper': A Feminist Reading of Madness and Agency" Which specific novels, characters, textual elements does the author analyze in this short story?"

Layer 2: "What is the author's opinion about the Yellow Wallpaper's narrator? How does this interpretation challenge traditional psychiatric readings of the text?"

Layer 3: "What feminist literary theory, close reading techniques, and textual analysis methods does the author employ to support their claims about narrative unreliability and female agency? What degree of certainty do they express about their interpretative framework?"

INSTRUCTIONS:
- Generate exactly 3 questions that are specifically tailored to THIS document's content. Be explicit of entities names, characters, etc.
- Each question should be modeled after the layer it belongs to. The questions can be redundant: for instance, asking for the subject of a paper's title even when the subject is explicit in the title. 
- Ensure questions are contextual to the document's overall topic, not on a specific section. If the author is talking about e.g. an artifact history but the main opinion of the author regards the authenticit of the artifact, be sure the question reflects this. 
- Use the document's specific terminology, entities, and concepts

Use the tool to return your 3 questions as a JSON array.
"""

    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system", 
                "content": "You are an expert academic question generator who creates contextualized, analytical questions based on document summaries."
            },
            {"role": "user", "content": prompt}
        ],
        tools=[question_schema],
        tool_choice={"type": "function", "function": {"name": "generate_document_questions"}},
        temperature=0.3
    )
    
    # Extract questions from tool call response
    tool_call = response.choices[0].message.tool_calls[0]
    questions_data = json.loads(tool_call.function.arguments)
    return questions_data["questions"]


def process_input_file(input_file: str = "input.json") -> Dict[str, Any]:
    """
    Process input.json file and generate questions for each document.
    
    Args:
        input_file: Path to input JSON file
    
    Returns:
        Updated configuration with generated questions
    """
    # Load input configuration
    with open(input_file, 'r', encoding='utf-8') as f:
        input_data = json.load(f)
    
    # Initialize OpenAI client
    openai_client = openai.OpenAI()
    
    # Handle both array and single file structures
    if 'files' in input_data and isinstance(input_data['files'], list):
        files_to_process = input_data['files']
        print(f"Found {len(files_to_process)} files to process in {input_file}")
    else:
        files_to_process = [input_data]
        print(f"Using legacy single file format from {input_file}")
    
    updated_files = []
    
    for file_idx, file_config in enumerate(files_to_process, 1):
        file_id = file_config.get('file_id', f'file_{file_idx}')
        print(f"\n{'='*80}")
        print(f"PROCESSING {file_id.upper()} ({file_idx}/{len(files_to_process)})")
        print(f"{'='*80}")
        
        # Extract configuration (ignore existing questions if present)
        doc_meta_dict = file_config.get('document_metadata', {})
        document_metadata = build_document_metadata_string(doc_meta_dict)
        file_path = file_config.get('file_path') or doc_meta_dict.get('file_path')
        
        if not file_path:
            print(f"Error: No file_path specified for {file_id}")
            # Preserve original config without questions
            updated_config = {k: v for k, v in file_config.items() if k != 'questions'}
            updated_files.append(updated_config)
            continue
        
        try:
            print(f"Step 1: Generating summary for {file_path}")
            # Generate document summary
            cumulative_summary, section_summaries = summarize_document(
                file_path=file_path,
                openai_client=openai_client,
                document_metadata=document_metadata,
                max_summary_tokens=400
            )
            
            print(f"Step 2: Generating contextualized questions based on summary")
            # Generate questions from summary
            generated_questions = generate_questions_from_summary(
                cumulative_summary=cumulative_summary,
                section_summaries=section_summaries,
                document_metadata=document_metadata,
                openai_client=openai_client
            )
            
            print(f"Generated {len(generated_questions)} questions:")
            for i, q in enumerate(generated_questions, 1):
                print(f"  Q{i}: {q}")
            
            # Create updated config preserving all original fields except questions
            updated_config = {k: v for k, v in file_config.items() if k != 'questions'}
            updated_config['questions'] = generated_questions
            updated_files.append(updated_config)
            
        except Exception as e:
            print(f"Error processing {file_id}: {str(e)}")
            # Preserve original config without questions on error
            updated_config = {k: v for k, v in file_config.items() if k != 'questions'}
            updated_files.append(updated_config)
    
    # Return updated structure
    if 'files' in input_data:
        return {'files': updated_files}
    else:
        return updated_files[0] if updated_files else {}


def save_output(output_data: Dict[str, Any], output_file: str = "input_with_questions.json"):
    """Save the updated configuration with generated questions."""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    print(f"\nOutput saved to {output_file}")


# Main execution
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Auto Question Generator - Generate contextualized questions from document summaries")
    parser.add_argument("--input", type=str, default="input.json", help="Input JSON file")
    parser.add_argument("--output", type=str, default="input_with_questions.json", help="Output JSON file")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        raise ValueError(f"Input file {args.input} not found")
    
    print("="*80)
    print("AUTO QUESTION GENERATOR")
    print("="*80)
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    
    # Process input file and generate questions
    output_data = process_input_file(args.input)
    
    # Save results
    save_output(output_data, args.output)
    
    print("\n" + "="*80)
    print("PROCESS COMPLETED")
    print("="*80)
