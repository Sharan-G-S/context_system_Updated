import numpy as np
from typing import List, Dict, Any, Tuple, Set
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import json
import re
from collections import Counter
from llm import call_llm

# Load sentence transformer for semantic similarity
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')


# ============================================
# DIMENSIONALITY REDUCTION
# ============================================

def reduce_dimensionality(
    embeddings: np.ndarray,
    method: str = "pca",
    target_dim: int = 128
) -> np.ndarray:
    """
    Reduce the dimensionality of embeddings.
    
    Args:
        embeddings: Input embeddings array (n_samples, n_features)
        method: Reduction method - 'pca', 'svd', or 'tsne'
        target_dim: Target dimensionality
        
    Returns:
        Reduced embeddings array
    """
    if embeddings.shape[1] <= target_dim:
        return embeddings
    
    if method == "pca":
        reducer = PCA(n_components=target_dim, random_state=42)
    elif method == "svd":
        reducer = TruncatedSVD(n_components=target_dim, random_state=42)
    elif method == "tsne":
        # t-SNE is more expensive, typically used for visualization
        reducer = TSNE(n_components=min(target_dim, 3), random_state=42)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    reduced = reducer.fit_transform(embeddings)
    return reduced


def compress_embeddings_batch(
    contexts_with_embeddings: List[Dict[str, Any]],
    target_dim: int = 128,
    method: str = "pca"
) -> List[Dict[str, Any]]:
    """
    Compress embeddings for a batch of contexts.
    
    Args:
        contexts_with_embeddings: List of context dicts with 'embedding' key
        target_dim: Target dimensionality
        method: Reduction method
        
    Returns:
        Contexts with compressed embeddings
    """
    if not contexts_with_embeddings:
        return []
    
    # Extract embeddings
    embeddings_list = []
    for ctx in contexts_with_embeddings:
        if "embedding" in ctx and ctx["embedding"] is not None:
            embeddings_list.append(ctx["embedding"])
    
    if not embeddings_list:
        return contexts_with_embeddings
    
    # Convert to numpy array
    embeddings = np.array(embeddings_list)
    
    # Reduce dimensionality
    reduced = reduce_dimensionality(embeddings, method=method, target_dim=target_dim)
    
    # Update contexts
    result = []
    embedding_idx = 0
    for ctx in contexts_with_embeddings:
        ctx_copy = ctx.copy()
        if "embedding" in ctx and ctx["embedding"] is not None:
            ctx_copy["embedding"] = reduced[embedding_idx].tolist()
            ctx_copy["original_dim"] = len(embeddings_list[embedding_idx])
            ctx_copy["reduced_dim"] = target_dim
            embedding_idx += 1
        result.append(ctx_copy)
    
    return result


# ============================================
# SUMMARIZATION
# ============================================

def summarize_extractive(contexts: List[Dict[str, Any]], max_sentences: int = 5) -> str:
    """
    Extractive summarization using TF-IDF and sentence embeddings.
    
    Args:
        contexts: List of context dictionaries
        max_sentences: Maximum number of sentences to extract
        
    Returns:
        Extracted summary
    """
    all_text = "\n".join(c.get("content", "") for c in contexts)
    
    # Split into sentences
    sentences = re.split(r'[.!?]+', all_text)
    sentences = [s.strip() for s in sentences if s.strip() and len(s.strip().split()) > 3]
    
    if not sentences:
        return ""
    
    if len(sentences) <= max_sentences:
        return ". ".join(sentences) + "."
    
    try:
        # Use TF-IDF to score sentences
        vectorizer = TfidfVectorizer(stop_words='english', max_features=100)
        tfidf_matrix = vectorizer.fit_transform(sentences)
        
        # Calculate sentence importance scores
        sentence_scores = []
        for idx, sentence in enumerate(sentences):
            # TF-IDF score
            tfidf_score = tfidf_matrix[idx].sum()
            
            # Length bonus (information density)
            length_bonus = min(len(sentence.split()) / 20.0, 1.0)
            
            # Position bonus (earlier sentences often more important)
            position_bonus = 1.0 / (idx + 1) ** 0.5
            
            total_score = tfidf_score + length_bonus + position_bonus
            sentence_scores.append((total_score, idx, sentence))
        
        # Sort by score and select top sentences
        sentence_scores.sort(reverse=True)
        
        # Take top sentences but maintain original order
        selected = sorted(sentence_scores[:max_sentences], key=lambda x: x[1])
        top_sentences = [sent for _, _, sent in selected]
        
    except Exception:
        # Fallback: use simple word count heuristic
        scored_sentences = []
        for idx, sent in enumerate(sentences):
            score = len(sent.split())
            scored_sentences.append((score, idx, sent))
        
        scored_sentences.sort(reverse=True)
        selected = sorted(scored_sentences[:max_sentences], key=lambda x: x[1])
        top_sentences = [sent for _, _, sent in selected]
    
    return ". ".join(top_sentences) + "."


def summarize_abstractive(contexts: List[Dict[str, Any]], max_length: int = 200) -> str:
    """
    Abstractive-style summarization using TextRank with sentence transformers.
    
    Args:
        contexts: List of context dictionaries
        max_length: Maximum length of summary in words
        
    Returns:
        Generated summary
    """
    all_text = "\n\n".join(c.get("content", "") for c in contexts)
    
    if not all_text.strip():
        return ""
    
    # Split into sentences
    sentences = re.split(r'[.!?]+', all_text)
    sentences = [s.strip() for s in sentences if s.strip() and len(s.strip().split()) > 3]
    
    if not sentences:
        return ""
    
    if len(sentences) <= 3:
        return ". ".join(sentences) + "."
    
    try:
        # Generate sentence embeddings
        embeddings = sentence_model.encode(sentences)
        
        # Calculate similarity matrix
        similarity_matrix = cosine_similarity(embeddings)
        
        # Apply TextRank algorithm (PageRank on sentence similarity graph)
        scores = np.ones(len(sentences))
        damping = 0.85
        
        for _ in range(10):  # 10 iterations
            new_scores = np.ones(len(sentences)) * (1 - damping)
            for i in range(len(sentences)):
                for j in range(len(sentences)):
                    if i != j and similarity_matrix[i][j] > 0:
                        new_scores[i] += damping * similarity_matrix[i][j] * scores[j] / similarity_matrix[j].sum()
            scores = new_scores
        
        # Select sentences until we reach max_length
        ranked_indices = np.argsort(scores)[::-1]
        selected_sentences = []
        word_count = 0
        
        for idx in ranked_indices:
            sentence = sentences[idx]
            words = len(sentence.split())
            if word_count + words <= max_length:
                selected_sentences.append((idx, sentence))
                word_count += words
            if word_count >= max_length * 0.9:  # Allow 10% margin
                break
        
        # Sort by original order
        selected_sentences.sort(key=lambda x: x[0])
        summary = ". ".join([sent for _, sent in selected_sentences])
        
        return summary + "." if summary else ""
        
    except Exception as e:
        # Fallback to extractive
        return summarize_extractive(contexts, max_sentences=min(5, max_length // 30))


def summarize_hierarchical(contexts: List[Dict[str, Any]], levels: int = 2) -> str:
    """
    Hierarchical summarization using progressive extractive compression.
    
    Args:
        contexts: List of context dictionaries
        levels: Number of summarization levels
        
    Returns:
        Final summary
    """
    if not contexts:
        return ""
    
    current_contexts = contexts
    
    for level in range(levels):
        # Calculate target sentence count for this level
        all_text = "\n\n".join(c.get("content", "") for c in current_contexts)
        sentences = re.split(r'[.!?]+', all_text)
        sentence_count = len([s for s in sentences if s.strip()])
        
        # Compress by half each level
        target_sentences = max(3, sentence_count // (2 ** (level + 1)))
        
        # Apply extractive summarization
        summary = summarize_extractive(current_contexts, max_sentences=target_sentences)
        
        # Update contexts for next level
        current_contexts = [{"content": summary}]
    
    return current_contexts[0]["content"] if current_contexts else ""


# ============================================
# SEMANTIC TRANSFORMATION
# ============================================

def extract_key_entities(contexts: List[Dict[str, Any]]) -> List[str]:
    """
    Extract key entities from contexts using LLM.
    
    Args:
        contexts: List of context dictionaries
        
    Returns:
        List of key entities
    """
    text = "\n\n".join(c.get("content", "") for c in contexts)
    
    if not text.strip():
        return []
    
    prompt = f"""
Extract the key entities (people, places, organizations, concepts) from the following text.
Return them as a JSON array of strings.

Text:
{text}

Entities (JSON array):"""

    response = call_llm([
        {"role": "system", "content": "You extract key entities and return JSON."},
        {"role": "user", "content": prompt}
    ])
    
    try:
        entities = json.loads(response)
        if isinstance(entities, list):
            return entities
    except json.JSONDecodeError:
        # Fallback: parse as comma-separated
        return [e.strip() for e in response.split(',')]
    
    return []


def extract_key_facts(contexts: List[Dict[str, Any]]) -> List[str]:
    """
    Extract key facts and relationships from contexts.
    
    Args:
        contexts: List of context dictionaries
        
    Returns:
        List of key facts
    """
    text = "\n\n".join(c.get("content", "") for c in contexts)
    
    if not text.strip():
        return []
    
    prompt = f"""
Extract the most important facts and relationships from the following text.
Return them as a JSON array of strings, with each fact being a clear, standalone statement.

Text:
{text}

Facts (JSON array):"""

    response = call_llm([
        {"role": "system", "content": "You extract key facts and return JSON."},
        {"role": "user", "content": prompt}
    ])
    
    try:
        facts = json.loads(response)
        if isinstance(facts, list):
            return facts
    except json.JSONDecodeError:
        # Fallback: split by newlines
        return [f.strip() for f in response.split('\n') if f.strip()]
    
    return []


def transform_to_qa_pairs(contexts: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """
    Transform contexts into question-answer pairs for better retrieval.
    
    Args:
        contexts: List of context dictionaries
        
    Returns:
        List of QA pair dictionaries with 'question' and 'answer' keys
    """
    text = "\n\n".join(c.get("content", "") for c in contexts)
    
    if not text.strip():
        return []
    
    prompt = f"""
Generate question-answer pairs from the following text.
Create questions that capture the key information and relationships.
Return as JSON array of objects with "question" and "answer" keys.

Text:
{text}

QA Pairs (JSON):"""

    response = call_llm([
        {"role": "system", "content": "You generate QA pairs and return JSON."},
        {"role": "user", "content": prompt}
    ])
    
    try:
        qa_pairs = json.loads(response)
        if isinstance(qa_pairs, list):
            return qa_pairs
    except json.JSONDecodeError:
        return []
    
    return []


def transform_to_structured_knowledge(contexts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Transform unstructured contexts into structured knowledge representation.
    
    Args:
        contexts: List of context dictionaries
        
    Returns:
        Structured knowledge dictionary
    """
    text = "\n\n".join(c.get("content", "") for c in contexts)
    
    if not text.strip():
        return {"entities": [], "facts": [], "relationships": []}
    
    # Extract entities
    entities = extract_key_entities(contexts)
    
    # Extract facts
    facts = extract_key_facts(contexts)
    
    # Extract relationships
    prompt = f"""
Identify key relationships between entities in the following text.
Return as JSON array of objects with "entity1", "relationship", and "entity2" keys.

Text:
{text}

Relationships (JSON):"""

    response = call_llm([
        {"role": "system", "content": "You extract relationships and return JSON."},
        {"role": "user", "content": prompt}
    ])
    
    relationships = []
    try:
        relationships = json.loads(response)
        if not isinstance(relationships, list):
            relationships = []
    except json.JSONDecodeError:
        relationships = []
    
    return {
        "entities": entities,
        "facts": facts,
        "relationships": relationships,
        "original_context_count": len(contexts)
    }


# ============================================
# MAIN COMPRESSION FUNCTION
# ============================================

def compress_context(
    contexts: List[Dict[str, Any]],
    method: str = "abstractive",
    enable_dimensionality_reduction: bool = False,
    enable_semantic_transform: bool = False
) -> Tuple[List[Dict[str, str]], str]:
    """
    Compress contexts using various methods.
    
    Args:
        contexts: List of context dictionaries
        method: Compression method - 'abstractive', 'extractive', 'hierarchical'
        enable_dimensionality_reduction: Apply dimensionality reduction to embeddings
        enable_semantic_transform: Apply semantic transformation
        
    Returns:
        Tuple of (compressed contexts, original text)
    """
    original_text = "\n\n".join(c.get("content", "") for c in contexts)
    
    # Apply dimensionality reduction if requested
    processed_contexts = contexts
    if enable_dimensionality_reduction:
        processed_contexts = compress_embeddings_batch(
            processed_contexts,
            target_dim=128,
            method="pca"
        )
    
    # Apply semantic transformation if requested
    semantic_data = {}
    if enable_semantic_transform:
        semantic_data = transform_to_structured_knowledge(contexts)
    
    # Generate summary based on method
    if method == "extractive":
        summary = summarize_extractive(contexts, max_sentences=5)
    elif method == "hierarchical":
        summary = summarize_hierarchical(contexts, levels=2)
    else:  # abstractive (default)
        summary = summarize_abstractive(contexts, max_length=200)
    
    # Enhance summary with semantic data if available
    if enable_semantic_transform and semantic_data:
        entities_text = ", ".join(semantic_data.get("entities", [])[:10])
        if entities_text:
            summary = f"{summary}\n\nKey entities: {entities_text}"
    
    return (
        [{"role": "system", "content": summary}],
        original_text
    )
