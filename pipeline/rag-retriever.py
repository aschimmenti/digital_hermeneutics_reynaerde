import faiss
import voyageai
import numpy as np
from typing import List, Dict, Tuple, Optional
import hashlib
import json
import os
import re

# Import indexing helpers
from indexer import (
    create_hybrid_index as build_faiss_index,
    save_index as save_faiss_index,
    load_index as load_faiss_index,
)

# Import utilities
from utils import (
    extract_sections_with_footnotes,
    create_paragraph_chunks_with_footnotes,
    add_overlap_to_chunks,
    load_few_shot_examples,
    load_questions_and_metadata,
    build_document_metadata_string,
)

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
PIPE_DATA_DIR = os.path.join(BASE_DIR, "data")

class SimpleRAGPipeline:
    def __init__(self, openai_client, voyage_api_key: Optional[str] = None):
        """
        Initialize the RAG pipeline with Voyage embeddings (no summarization).
        
        Args:
            openai_client: OpenAI client for generation
            voyage_api_key: Your Voyage AI API key (if None, uses VOYAGE_API_KEY env var)
        """
        self.voyage_client = voyageai.Client(api_key=voyage_api_key or os.getenv("VOYAGE_API_KEY"))
        self.openai_client = openai_client
        self.chunks = []
        self.chunk_metadata = []
        self.index = None
        self.embeddings_cache = {}
        self.full_document_text = ""
        self.document_sections = []
        
    def smart_chunk_document(self, text: str, target_chunk_size: int = 800, 
                        overlap: int = 100) -> List[Dict]:
        # Extract sections with their footnotes
        self.document_sections = extract_sections_with_footnotes(text)
        
        # Create paragraph-based chunks
        chunks_with_metadata = create_paragraph_chunks_with_footnotes(
            self.document_sections, 
            min_chunk_size=target_chunk_size // 4
        )
        
        # Add overlap between chunks if needed
        if overlap > 0:
            chunks_with_metadata = add_overlap_to_chunks(chunks_with_metadata, overlap)
        
        return chunks_with_metadata
    
    def create_contextualized_embeddings(self, chunks_with_metadata: List[Dict]) -> np.ndarray:
        """Create Voyage contextualized embeddings that preserve document structure."""
        # Group chunks by section for better context preservation
        sections_chunks = {}
        for chunk in chunks_with_metadata:
            section_number = chunk['section_number']
            if section_number not in sections_chunks:
                sections_chunks[section_number] = []
            sections_chunks[section_number].append(chunk['text'])
        
        # Prepare inputs for contextualized embeddings
        inputs_for_voyage = list(sections_chunks.values())
        
        print(f"Creating contextualized embeddings for {len(inputs_for_voyage)} document sections...")
        
        # Use voyage-context-3 for best quality with context preservation
        embeddings_obj = self.voyage_client.contextualized_embed(
            inputs=inputs_for_voyage,
            model="voyage-context-3",
            input_type="document",
            output_dimension=1024
        )
        
        # Flatten embeddings while maintaining order
        all_embeddings = []
        for result in embeddings_obj.results:
            all_embeddings.extend(result.embeddings)
        
        return np.array(all_embeddings).astype('float32')
    
    def create_hybrid_index(self, embeddings: np.ndarray):
        """Wrapper to build the FAISS index via indexer.py"""
        self.index = build_faiss_index(embeddings)

    def process_document(self, file_path: str):
        """Process a document through the RAG pipeline."""
        print(f"Processing document: {file_path}")
        
        # Read and store the full document
        with open(file_path, 'r', encoding='utf-8') as f:
            self.full_document_text = f.read()
        
        # Create smart chunks
        self.chunks = self.smart_chunk_document(self.full_document_text)
        print(f"Created {len(self.chunks)} chunks from {len(self.document_sections)} sections")
        
        # Create contextualized embeddings
        embeddings = self.create_contextualized_embeddings(self.chunks)
        
        # Build index
        self.create_hybrid_index(embeddings)
        print("Index created successfully")
        
        # Note: Index and metadata saving will be handled by caller with proper output directory
        
    def enhanced_retrieval(self, query: str, k: int = 5, 
                          use_reranking: bool = True) -> List[Dict]:
        """Retrieve relevant chunks with optional reranking."""
        # Embed query using contextualized embeddings
        query_embedding = self.voyage_client.contextualized_embed(
            inputs=[[query]],
            model="voyage-context-3",
            input_type="query",
            output_dimension=1024
        ).results[0].embeddings[0]
        
        query_embedding = np.array([query_embedding]).astype('float32')
        
        # Search in FAISS
        if use_reranking:
            # Retrieve more candidates for reranking
            distances, indices = self.index.search(query_embedding, k * 3)
            candidates = [self.chunks[i] for i in indices[0]]
            
            # Use Voyage reranker for better accuracy
            rerank_results = self.voyage_client.rerank(
                query=query,
                documents=[c['text'] for c in candidates],
                model="rerank-2",
                top_k=k
            )
            
            # Get reranked chunks
            retrieved_chunks = []
            for result in rerank_results.results:
                chunk = candidates[result.index]
                chunk['relevance_score'] = result.relevance_score
                retrieved_chunks.append(chunk)
        else:
            distances, indices = self.index.search(query_embedding, k)
            retrieved_chunks = [self.chunks[i] for i in indices[0]]
            for chunk, dist in zip(retrieved_chunks, distances[0]):
                chunk['relevance_score'] = float(1 / (1 + dist))
        
        return retrieved_chunks
    
    def ask_sequential(
        self,
        document_metadata: str,
        questions: List[str],
        k: int = 5,
        few_shot_path: Optional[str] = None,
        few_shot_examples: Optional[List[Dict]] = None,
        metadata_dict: Optional[Dict] = None,
    ) -> List[str]:
        """Answer questions sequentially, using previous answers as context."""
        answers = []
        previous_context = ""
        
        
        for idx, question in enumerate(questions):
            # Retrieve relevant chunks for this question
            retrieved_chunks = self.enhanced_retrieval(question, k=k)
            
            # Build context with metadata
            context_parts = []
            for i, chunk in enumerate(retrieved_chunks, 1):
                section = chunk.get('section', 'Unknown Section')
                score = chunk.get('relevance_score', 0)
                context_parts.append(
                    f"[Source {i} - {section} (Relevance: {score:.3f})]\n{chunk['text']}\n"
                )
            context = "\n---\n".join(context_parts)
            
            # Build prompt with previous answers as context
            if previous_context:
                target_prompt = f"""
{document_metadata}

Previous Analysis:
{previous_context}

Context:
{context}

Question: {question}

Building on the previous analysis, please structure your response as a single, clear and concise paragraph following these steps:
1. First, identify the key relevant passages from the context
2. Then, based on the question, extract the relevant information that answers the question
3. Include specific references to the text where appropriate, alongside mentions of other documents in case the provenance of a statement is external to the given document. Use the markdown format of [^number of the section] to reference the input context. When the information is presented from the author, explicitly mention it, e.g. "According to the [author name(s)]". If it is reported from another source from the authors, make it explicit as well.
4. If the context doesn't fully answer the question, acknowledge what information is missing

Answer:"""
            else:
                target_prompt = f"""
{document_metadata}

Context:
{context}

Question: {question}

Please structure your response as a single, clear and concise paragraph following these steps:
1. First, identify the key relevant passages from the context
2. Then, based on the question, extract the relevant information that answers the question
3. Include specific references to the text where appropriate, alongside mentions of other documents in case the provenance of a statement is external to the given document. Use the markdown format of [^number of the section] to reference the input context. When the information is presented from the author, explicitly mention it, e.g. "According to the [author name(s)]". If it is reported from another source from the authors, make it explicit as well.
4. If the context doesn't fully answer the question, acknowledge what information is missing

Answer:"""
            
            # Build chat messages with optional few-shot examples
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are an expert Question-Answering agent that answers literary questions based on an author's publication. "
                        "Given a source and some snippets of it as context given through RAG, solve the task at hand. "
                        "Be precise, avoid metaphors and ambiguity, and keep a clear, explicative and factual style."
                    ),
                }
            ]
            
            # Load examples if a path is provided and in-memory examples not set
            if few_shot_examples is None and few_shot_path:
                try:
                    few_shot_examples = load_few_shot_examples(few_shot_path)
                except Exception as e:
                    print(f"Warning: Failed to load few-shot examples from {few_shot_path}: {e}")
                    few_shot_examples = None
            
            # Add few-shot examples (as user/assistant pairs)
            if few_shot_examples:
                for ex in few_shot_examples:
                    ex_context = ex.get("context", "").strip()
                    ex_question = ex.get("question", "").strip()
                    ex_answer = ex.get("expected_answer", "").strip()
                    if ex_context and ex_question and ex_answer:
                        messages.append({
                            "role": "user",
                            "content": f"Context:\n{ex_context}\n\nQuestion: {ex_question}\nAnswer as per the guidelines.",
                        })
                        messages.append({
                            "role": "assistant",
                            "content": ex_answer,
                        })
            
            # Add the actual target question
            messages.append({"role": "user", "content": target_prompt})
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0.3,
                max_tokens=10000, 
            )
            
            answer = response.choices[0].message.content
            answers.append(answer)
            
            # Update previous context for next question
            previous_context += f"Q{idx+1}: {question}\nA{idx+1}: {answer}\n\n"
            
        return answers

    
    
    def save_qa_results(self, results: List[Dict], filename: str = "rag_document_qa.json", generation_method: str = 'rag_only', document_metadata: Dict = None):
        """Save question-answer results to a file."""
        # Ensure directory exists
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        # Organize results by pseudo-section (here always 'Document-wide' or 'User Provided')
        organized_results = {}
        for result in results:
            section_title = result.get('section_title', 'Document-wide')
            if section_title not in organized_results:
                organized_results[section_title] = {
                    'section_number': result.get('section_number'),
                    'questions_and_answers': []
                }
            organized_results[section_title]['questions_and_answers'].append({
                'question_id': result.get('question_id'),
                'question': result.get('question'),
                'answer': result.get('answer')
            })

        output_data = {
            'metadata': {
                'total_sections': len(organized_results),
                'total_questions': len(results),
                'generation_method': generation_method
            },
            'sections': organized_results
        }
        
        # Add document metadata if provided
        if document_metadata:
            output_data['document_metadata'] = document_metadata

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        print(f"Results saved to {filename}")
    
    def save_index(self, path: str):
        """Save the FAISS index to disk via indexer.py"""
        # Ensure directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)
        save_faiss_index(self.index, path)
        print(f"Index saved to {path}")
    
    def save_metadata(self, path: str):
        """Save chunks and metadata to disk."""
        # Ensure directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data_to_save = {
            'chunks': self.chunks,
            'sections': self.document_sections
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data_to_save, f, ensure_ascii=False, indent=2)
        print(f"Metadata saved to {path}")
    
    def load_index(self, path: str):
        """Load the FAISS index from disk via indexer.py"""
        self.index = load_faiss_index(path)
    
    def load_metadata(self, path: str):
        """Load chunks and metadata from disk."""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.chunks = data.get('chunks', [])
        self.document_sections = data.get('sections', [])
        print(f"Metadata loaded from {path}")
    

# Main execution (fixed process-all mode)
if __name__ == "__main__":
    import openai
    
    ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, os.pardir))
    input_file = os.path.join(BASE_DIR, "input.json")
    documents_base = os.path.join(BASE_DIR, "documents")
    os.makedirs(documents_base, exist_ok=True)
    
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"input.json not found at {input_file}")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        input_data = json.load(f)
    
    if 'files' in input_data and isinstance(input_data['files'], list):
        files_to_process = input_data['files']
        print(f"Found {len(files_to_process)} files to process in input.json")
    else:
        files_to_process = [input_data]
        print("Using legacy single file format from input.json")
    
    # Initialize OpenAI client and pipeline
    openai_client = openai.OpenAI()
    rag = SimpleRAGPipeline(openai_client)
    
    all_results = []
    # Default few-shot examples path in the pipeline folder
    default_few_shot = os.path.join(BASE_DIR, "few_shot_examples.json")
    if not os.path.exists(default_few_shot):
        default_few_shot = None
    
    for file_idx, file_config in enumerate(files_to_process, 1):
        file_id = file_config.get('file_id', f'file_{file_idx}')
        print(f"\n{'='*80}")
        print(f"PROCESSING {file_id.upper()} ({file_idx}/{len(files_to_process)})")
        print(f"{'='*80}")
        
        # Output directory inside pipeline/documents
        output_dir = os.path.join(documents_base, file_id)
        os.makedirs(output_dir, exist_ok=True)
        
        # Config
        user_questions = file_config.get('questions') or None
        doc_meta_dict = file_config.get('document_metadata') or {}
        document_metadata = build_document_metadata_string(doc_meta_dict)
        # Few-shot path: per-file override or default in pipeline
        few_shot_path = file_config.get('few_shot_examples_path') or default_few_shot
        if few_shot_path and not os.path.isabs(few_shot_path):
            few_shot_path = os.path.join(BASE_DIR, few_shot_path)
        # Force source path to pipeline/data/<basename>
        declared_path = file_config.get('file_path') or doc_meta_dict.get('file_path')
        if not declared_path:
            print(f"Error: No file_path specified for {file_id}")
            continue
        filename = os.path.basename(declared_path)
        file_path = os.path.join(PIPE_DATA_DIR, filename)
        
        if not os.path.exists(file_path):
            print(f"Error: Source file not found in pipeline/data: {filename}")
            continue
        
        try:
            index_path = os.path.join(output_dir, "document_index.faiss")
            metadata_path = os.path.join(output_dir, "document_metadata.json")
            
            if os.path.exists(index_path) and os.path.exists(metadata_path):
                print(f"Loading existing index and metadata for {file_id}...")
                rag.load_index(index_path)
                rag.load_metadata(metadata_path)
            else:
                print(f"Processing document: {file_path}")
                with open(file_path, 'r', encoding='utf-8') as f:
                    rag.full_document_text = f.read()
                rag.chunks = rag.smart_chunk_document(rag.full_document_text)
                print(f"Created {len(rag.chunks)} chunks from {len(rag.document_sections)} sections")
                embeddings = rag.create_contextualized_embeddings(rag.chunks)
                rag.create_hybrid_index(embeddings)
                print("Index created successfully")
                rag.save_index(index_path)
                rag.save_metadata(metadata_path)
            
            if not user_questions:
                print(f"Warning: No questions found for {file_id}")
                continue
            
            print("\n" + "="*80)
            print("DOCUMENT-WIDE AUTOMATED QUESTION-ANSWER SESSION")
            print("="*80 + "\n")
            
            answers = rag.ask_sequential(
                document_metadata,
                user_questions,
                k=5,
                few_shot_path=few_shot_path,
                metadata_dict=doc_meta_dict,
            )
            
            results = []
            for idx, (question, answer) in enumerate(zip(user_questions, answers), start=1):
                print(f"\nQ{idx}: {question}")
                print("-" * 40)
                print(f"Answer: {answer}\n")
                results.append({
                    'question': question,
                    'answer': answer,
                    'section_title': 'Document-wide',
                    'section_number': None,
                    'question_id': idx,
                })
            
            output_filename = os.path.join(output_dir, "rag_document_qa.json")
            rag.save_qa_results(results, output_filename, 'rag_only', document_metadata=doc_meta_dict)
            all_results.extend(results)
        except Exception as e:
            print(f"Error processing {file_id}: {str(e)}")
    
    if len(files_to_process) > 1:
        comb_path = os.path.join(BASE_DIR, "rag_qa_combined.json")
        with open(comb_path, 'w', encoding='utf-8') as f:
            json.dump({
                'metadata': {
                    'total_files': len(files_to_process),
                    'total_questions': len(all_results),
                    'generation_method': 'rag_only'
                },
                'results': all_results
            }, f, ensure_ascii=False, indent=2)
        print(f"Combined results saved to {comb_path}")
