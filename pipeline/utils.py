from typing import List, Dict, Optional, Tuple
import os
import json
import re

# ---------------------------
# Parsing and chunking utils
# ---------------------------

def extract_sections_with_footnotes(text: str) -> List[Dict]:
    """
    Extract sections based on markdown-style titles (# Title) and group footnotes with their sections.
    Returns a list of section dicts with title, content, footnotes, and metadata.
    """
    sections: List[Dict] = []

    # Extract all footnote definitions
    footnote_pattern = r"\[\^(\d+)\]:\s*(.+?)(?=\n\[\^|\n\n|\Z)"
    footnotes: Dict[int, str] = {}
    for match in re.finditer(footnote_pattern, text, re.DOTALL):
        footnote_num = int(match.group(1))
        footnote_content = match.group(2).strip()
        footnotes[footnote_num] = footnote_content

    # Remove footnote definitions from main text
    text_without_footnote_defs = re.sub(footnote_pattern, "", text, flags=re.DOTALL)

    lines = text_without_footnote_defs.split("\n")
    current_section: Optional[str] = None
    current_content: List[str] = []

    for line in lines:
        if line.strip().startswith('# ') and len(line.strip()) > 2:
            if current_section is not None:
                section_content = '\n'.join(current_content).strip()
                section_footnotes = extract_section_footnotes(section_content, footnotes)
                sections.append({
                    'title': current_section,
                    'content': section_content,
                    'footnotes': section_footnotes,
                    'section_number': len(sections),
                    'word_count': len(section_content.split()),
                })
            current_section = line.strip()[2:].strip()
            current_content = [line]
        else:
            if current_section is not None:
                current_content.append(line)
            else:
                if not current_section and line.strip():
                    if not sections:
                        current_section = "Introduction/Preamble"
                        current_content = [line]

    if current_section is not None:
        section_content = '\n'.join(current_content).strip()
        section_footnotes = extract_section_footnotes(section_content, footnotes)
        sections.append({
            'title': current_section,
            'content': section_content,
            'footnotes': section_footnotes,
            'section_number': len(sections),
            'word_count': len(section_content.split()),
        })

    return sections


def extract_section_footnotes(section_content: str, all_footnotes: Dict[int, str]) -> Dict[int, str]:
    """Extract footnotes referenced in a specific section."""
    section_footnotes: Dict[int, str] = {}
    footnote_refs = re.findall(r"\[\^(\d+)\]", section_content)
    for ref in footnote_refs:
        footnote_num = int(ref)
        if footnote_num in all_footnotes:
            section_footnotes[footnote_num] = all_footnotes[footnote_num]
    return section_footnotes


def extract_paragraph_footnotes(paragraph: str, section_footnotes: Dict[int, str]) -> Dict[int, str]:
    """Extract footnotes referenced in a specific paragraph."""
    paragraph_footnotes: Dict[int, str] = {}
    footnote_refs = re.findall(r"\[\^(\d+)\]", paragraph)
    for ref in footnote_refs:
        footnote_num = int(ref)
        if footnote_num in section_footnotes:
            paragraph_footnotes[footnote_num] = section_footnotes[footnote_num]
    return paragraph_footnotes


def create_paragraph_chunks_with_footnotes(sections: List[Dict], min_chunk_size: int = 200) -> List[Dict]:
    """
    Create paragraph-based chunks while preserving footnote associations.
    Returns list of chunk dicts with associated footnotes and metadata.
    """
    chunks_with_metadata: List[Dict] = []

    for section in sections:
        section_content = section['content']
        section_title = section['title']
        section_number = section['section_number']
        section_footnotes = section['footnotes']

        paragraphs = re.split(r"\n\s*\n", section_content)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        current_chunk = ""
        current_chunk_footnotes: Dict[int, str] = {}

        for paragraph in paragraphs:
            if current_chunk and len(current_chunk + "\n\n" + paragraph) > 1000:
                if current_chunk:
                    chunks_with_metadata.append({
                        'text': current_chunk.strip(),
                        'section': section_title,
                        'section_number': section_number,
                        'chunk_id': len(chunks_with_metadata),
                        'footnotes': current_chunk_footnotes.copy(),
                        'word_count': len(current_chunk.split()),
                        'type': 'paragraph_group',
                    })
                current_chunk = paragraph
                current_chunk_footnotes = extract_paragraph_footnotes(paragraph, section_footnotes)
            else:
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
                para_footnotes = extract_paragraph_footnotes(paragraph, section_footnotes)
                current_chunk_footnotes.update(para_footnotes)

        if current_chunk and len(current_chunk.split()) >= min_chunk_size // 10:
            chunks_with_metadata.append({
                'text': current_chunk.strip(),
                'section': section_title,
                'section_number': section_number,
                'chunk_id': len(chunks_with_metadata),
                'footnotes': current_chunk_footnotes.copy(),
                'word_count': len(current_chunk.split()),
                'type': 'paragraph_group',
            })

    return chunks_with_metadata


def add_overlap_to_chunks(chunks: List[Dict], overlap_size: int) -> List[Dict]:
    """Add overlapping content between consecutive chunks while preserving footnotes."""
    if len(chunks) <= 1:
        return chunks

    overlapped_chunks: List[Dict] = []
    for i, chunk in enumerate(chunks):
        chunk_text = chunk['text']
        chunk_footnotes = chunk['footnotes'].copy()
        if i > 0:
            prev_chunk = chunks[i - 1]
            prev_text = prev_chunk['text']
            if len(prev_text) > overlap_size:
                overlap_text = prev_text[-overlap_size:]
                sentence_end = overlap_text.find('. ')
                if sentence_end > overlap_size // 2:
                    overlap_text = overlap_text[sentence_end + 2:]
                chunk_text = overlap_text + "\n\n" + chunk_text
                overlap_footnotes = extract_paragraph_footnotes(overlap_text, prev_chunk['footnotes'])
                chunk_footnotes.update(overlap_footnotes)
        updated_chunk = chunk.copy()
        updated_chunk['text'] = chunk_text
        updated_chunk['footnotes'] = chunk_footnotes
        updated_chunk['word_count'] = len(chunk_text.split())
        updated_chunk['has_overlap'] = i > 0
        overlapped_chunks.append(updated_chunk)
    return overlapped_chunks


def format_chunk_with_footnotes(chunk: Dict) -> str:
    """Format a chunk with its associated footnotes for display or processing."""
    text = chunk['text']
    footnotes = chunk.get('footnotes', {})
    if not footnotes:
        return text
    footnote_text = "\n\nFootnotes:\n"
    for num in sorted(footnotes.keys()):
        footnote_text += f"[^{num}]: {footnotes[num]}\n"
    return text + footnote_text


# ---------------------------
# I/O helpers
# ---------------------------

def load_questions_from_file(path: str) -> List[str]:
    """Load questions from a .txt (one per line) or JSON file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Questions file not found: {path}")
    questions: List[str] = []
    if path.lower().endswith('.json'):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, dict) and isinstance(data.get('questions'), list):
            items = data['questions']
        elif isinstance(data, list):
            items = data
        else:
            raise ValueError("JSON questions file must be a list of strings or an object with a 'questions' list")
        for q in items:
            if isinstance(q, str) and q.strip():
                questions.append(q.strip())
    else:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                q = line.strip()
                if q and not q.startswith('#'):
                    questions.append(q if q.endswith('?') else q + '?')
    return questions


def load_questions_and_metadata(path: str) -> Dict:
    """
    Load a JSON file containing:
    - document_metadata: {authors, date, journal, doi, title, publisher, url, etc.}
    - automated_process_metadata: {person_name, llm_model_version}
    - questions: array of strings (three)

    Returns a dict with keys: 'document_metadata', 'automated_process_metadata', 'questions'.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Questions JSON not found: {path}")
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("Questions JSON must be an object with 'questions' array and metadata objects")
    doc_meta = data.get('document_metadata') or {}
    if not isinstance(doc_meta, dict):
        doc_meta = {}
    proc_meta = data.get('automated_process_metadata') or {}
    if not isinstance(proc_meta, dict):
        proc_meta = {}
    questions: List[str] = []
    items = data.get('questions') or []
    if not isinstance(items, list):
        raise ValueError("'questions' must be an array of strings")
    for q in items:
        if isinstance(q, str) and q.strip():
            questions.append(q.strip())
    return {
        'document_metadata': doc_meta,
        'automated_process_metadata': proc_meta,
        'questions': questions,
    }


def build_document_metadata_string(doc_meta: Dict) -> str:
    """Create a concise bibliographic string from a document metadata dict."""
    authors = doc_meta.get('authors') or doc_meta.get('author') or ''
    date = doc_meta.get('date') or doc_meta.get('year') or ''
    title = doc_meta.get('title') or ''
    journal = doc_meta.get('journal') or doc_meta.get('venue') or ''
    publisher = doc_meta.get('publisher') or ''
    doi = doc_meta.get('doi') or ''
    url = doc_meta.get('url') or doc_meta.get('link') or ''

    parts: List[str] = []
    # Normalize authors to a printable string
    authors_str = ''
    if isinstance(authors, str):
        authors_str = authors.strip()
    elif isinstance(authors, list):
        formatted: List[str] = []
        for a in authors:
            if isinstance(a, str):
                if a.strip():
                    formatted.append(a.strip())
            elif isinstance(a, dict):
                fam = (a.get('family_name') or '').strip()
                giv = (a.get('given_name') or '').strip()
                if fam and giv:
                    formatted.append(f"{fam}, {giv}")
                elif fam:
                    formatted.append(fam)
                elif giv:
                    formatted.append(giv)
        authors_str = '; '.join([s for s in formatted if s])
    # Build name + year
    name_year = ' '.join(p for p in [authors_str, f"({date})" if date else ''] if p).strip()
    if name_year:
        parts.append(name_year)
    if title:
        parts.append(title)
    venue = '. '.join(p for p in [journal, publisher] if p)
    if venue:
        parts.append(venue)
    if doi:
        parts.append(f"DOI: {doi}")
    if url and not doi:
        parts.append(url)
    return '. '.join(parts).strip()


def load_few_shot_examples(path: str) -> List[Dict]:
    """Load answer few-shots from list or from {'answers': [...]} object."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Few-shot examples file not found: {path}")
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if isinstance(data, list):
        items = data
    elif isinstance(data, dict) and isinstance(data.get('answers'), list):
        items = data['answers']
    else:
        raise ValueError("Few-shot examples JSON must be a list or an object with 'answers' list")
    cleaned: List[Dict] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        ctx = item.get('context')
        q = item.get('question')
        a = item.get('expected_answer') or item.get('answer')
        if isinstance(ctx, str) and isinstance(q, str) and isinstance(a, str):
            cleaned.append({'context': ctx, 'question': q, 'expected_answer': a})
    return cleaned


def load_few_shot_question_examples(path: str) -> List[Dict]:
    """Load question-generation few-shots from an object with 'questions' list."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Few-shot questions file not found: {path}")
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if not isinstance(data, dict) or not isinstance(data.get('questions'), list):
        return []
    cleaned: List[Dict] = []
    for item in data['questions']:
        if not isinstance(item, dict):
            continue
        doc_ctx = item.get('document_context', '')
        title = item.get('section_title', '')
        content = item.get('section_content', '')
        gen_qs = item.get('generated_questions', [])
        if isinstance(title, str) and isinstance(content, str) and isinstance(gen_qs, list):
            cleaned.append({
                'document_context': doc_ctx,
                'section_title': title,
                'section_content': content,
                'generated_questions': [q for q in gen_qs if isinstance(q, str) and q.strip()],
            })
    return cleaned
