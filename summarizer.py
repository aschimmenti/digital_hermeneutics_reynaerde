import json
import os
import re
from typing import List, Dict, Optional, Tuple

# Import utilities
from utils import extract_sections_with_footnotes


def summarize_document(
    file_path: str,
    openai_client,
    document_metadata: str = "",
    max_summary_tokens: int = 400,
    save_to_file: Optional[str] = None
) -> Tuple[str, List[Dict]]:
    """
    Summarize a document progressively, section by section.
    
    Args:
        file_path: Path to the document file
        openai_client: OpenAI client for generation
        document_metadata: Optional metadata string for context
        max_summary_tokens: Maximum tokens per summary section
        save_to_file: Optional path to save results as JSON
    
    Returns:
        Tuple of (cumulative_summary, section_summaries)
    """
    # Read document
    with open(file_path, 'r', encoding='utf-8') as f:
        full_document_text = f.read()
    
    # Extract sections
    document_sections = extract_sections_with_footnotes(full_document_text)
    print(f"Extracted {len(document_sections)} sections from {file_path}")
    
    section_summaries = []
    cumulative_summary = ""

    for idx, section in enumerate(document_sections):
        prev_context = cumulative_summary.strip()
        section_title = section.get('title', f'Section {idx}')
        section_content = section.get('content', '')

        sys_msg = (
            "You are an expert scholarly summarizer. Produce concise summaries that preserve: "
            "(a) key artefacts/objects of discourse, (b) author intent/research questions, "
            "(c) claims with who/when/source/degree of certainty, and (d) important references/notes."
        )

        if prev_context:
            user_prompt = f"""
{document_metadata}
You will summarize the current section given the cumulative context from previous sections.

Previous cumulative summary (for context):
{prev_context}

Current Section Title: {section_title}
Current Section Content:
{section_content}

Instructions:
- Write a concise summary (<= {max_summary_tokens} tokens) capturing new information and how it relates to prior content.
- Explicitly note continuity or changes in argumentation.
- Mention artefacts, author intents, and claims with provenance when present.
Output format:
Summary:
"""
        else:
            user_prompt = f"""
{document_metadata}
Summarize the following section. Focus on: artefacts/objects, author intent, and key claims with provenance.

Section Title: {section_title}
Section Content:
{section_content}

Output format:
Summary:
"""

        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=max_summary_tokens + 100,
        )

        summary_text = (response.choices[0].message.content or "").strip()
        # Normalize: keep only the text after 'Summary:' if present
        m = re.search(r"Summary:\s*(.*)\Z", summary_text, re.DOTALL | re.IGNORECASE)
        if m:
            summary_text = m.group(1).strip()

        section_summaries.append({
            'section_number': section.get('section_number', idx),
            'section_title': section_title,
            'summary': summary_text,
        })

        # Update cumulative summary by appending, then keep it concise via a brief compress step
        combined = (cumulative_summary + "\n\n" + f"[{section_title}] " + summary_text).strip()

        compress_prompt = f"""
{document_metadata}
Compress the following cumulative summary into a coherent, non-redundant synthesis while preserving:
- artefacts/objects of discourse
- author intent/research questions
- claims with who/when/source/degree of certainty
- important references/notes when explicitly mentioned

Text:
{combined}

Output a concise synthesis:
"""
        compress_resp = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You compress and synthesize scholarly summaries without losing key provenance."},
                {"role": "user", "content": compress_prompt},
            ],
            temperature=0.2,
            max_tokens=max_summary_tokens + 100,
        )
        cumulative_summary = (compress_resp.choices[0].message.content or combined).strip()

    # Save to file if requested
    if save_to_file:
        os.makedirs(os.path.dirname(save_to_file), exist_ok=True)
        output_data = {
            'metadata': {
                'total_sections': len(section_summaries),
                'generation_method': 'progressive_summarization'
            },
            'cumulative_summary': cumulative_summary,
            'section_summaries': section_summaries
        }
        with open(save_to_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        print(f"Summaries saved to {save_to_file}")

    return cumulative_summary, section_summaries


def summarize_multiple_documents(
    file_configs: List[Dict],
    openai_client,
    max_summary_tokens: int = 400,
    output_dir: str = "output"
) -> Dict[str, Tuple[str, List[Dict]]]:
    """
    Summarize multiple documents from configuration list.
    
    Args:
        file_configs: List of file configuration dictionaries
        openai_client: OpenAI client for generation
        max_summary_tokens: Maximum tokens per summary section
        output_dir: Base output directory
    
    Returns:
        Dictionary mapping file_id to (cumulative_summary, section_summaries)
    """
    from utils import build_document_metadata_string
    
    results = {}
    
    for file_idx, file_config in enumerate(file_configs, 1):
        file_id = file_config.get('file_id', f'file_{file_idx}')
        print(f"\n{'='*80}")
        print(f"PROCESSING {file_id.upper()} ({file_idx}/{len(file_configs)})")
        print(f"{'='*80}")
        
        # Extract configuration
        doc_meta_dict = file_config.get('document_metadata') or {}
        document_metadata = build_document_metadata_string(doc_meta_dict)
        file_path = file_config.get('file_path') or doc_meta_dict.get('file_path')
        
        if not file_path:
            print(f"Error: No file_path specified for {file_id}")
            continue
        
        try:
            # Create output path
            output_file = f"{output_dir}/{file_id}/document_summaries.json"
            
            # Summarize document
            cumulative_summary, section_summaries = summarize_document(
                file_path=file_path,
                openai_client=openai_client,
                document_metadata=document_metadata,
                max_summary_tokens=max_summary_tokens,
                save_to_file=output_file
            )
            
            results[file_id] = (cumulative_summary, section_summaries)
            
            # Print results
            print(f"\nFinal Cumulative Summary:")
            print("-" * 40)
            print(f"{cumulative_summary}\n")
            
        except Exception as e:
            print(f"Error processing {file_id}: {str(e)}")
    
    return results


# Main execution for standalone use
if __name__ == "__main__":
    import argparse
    import openai
    
    parser = argparse.ArgumentParser(description="Document Summarizer - Progressive Summarization Only")
    parser.add_argument("--file", type=str, help="Path to document file")
    parser.add_argument("--max-tokens", type=int, default=400, help="Maximum tokens per summary")
    parser.add_argument("--output", type=str, help="Output file path")
    
    args = parser.parse_args()
    
    if not args.file:
        raise ValueError("--file argument is required")
    
    # Initialize OpenAI client
    openai_client = openai.OpenAI()
    
    # Summarize document
    cumulative_summary, section_summaries = summarize_document(
        file_path=args.file,
        openai_client=openai_client,
        max_summary_tokens=args.max_tokens,
        save_to_file=args.output
    )
    
    # Print results
    print(f"\nFinal Cumulative Summary:")
    print("-" * 40)
    print(f"{cumulative_summary}\n")
    
    print(f"Section Summaries:")
    print("-" * 40)
    for summary in section_summaries:
        print(f"Section {summary['section_number']}: {summary['section_title']}")
        print(f"Summary: {summary['summary']}\n")
