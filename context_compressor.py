from llm import call_llm
import numpy as np
from typing import List, Dict, Any, Tuple
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE
import json


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
    Extractive summarization: select most important sentences.
    
    Args:
        contexts: List of context dictionaries
        max_sentences: Maximum number of sentences to extract
        
    Returns:
        Extracted summary
    """
    all_text = "\n".join(c.get("content", "") for c in contexts)
    sentences = [s.strip() for s in all_text.split('.') if s.strip()]
    
    # Simple heuristic: prioritize longer, information-rich sentences
    scored_sentences = []
    for sent in sentences:
        score = len(sent.split())  # Word count as simple score
        scored_sentences.append((score, sent))
    
    # Sort by score and take top sentences
    scored_sentences.sort(reverse=True)
    top_sentences = [sent for _, sent in scored_sentences[:max_sentences]]
    
    return ". ".join(top_sentences) + "."


def summarize_abstractive(contexts: List[Dict[str, Any]], max_length: int = 200) -> str:
    """
    Abstractive summarization using LLM.
    
    Args:
        contexts: List of context dictionaries
        max_length: Maximum length of summary in words
        
    Returns:
        Generated summary
    """
    original_text = "\n\n".join(c.get("content", "") for c in contexts)
    
    if not original_text.strip():
        return ""
    
    prompt = f"""
Summarize the following context in approximately {max_length} words.
Focus on key facts, entities, and relationships.
Be concise but preserve important details.

Context:
{original_text}

Summary:"""

    summary = call_llm([
        {"role": "system", "content": "You are an expert at creating concise, informative summaries."},
        {"role": "user", "content": prompt}
    ])

    return summary.strip()


def summarize_hierarchical(contexts: List[Dict[str, Any]], levels: int = 2) -> str:
    """
    Hierarchical summarization: summarize in multiple passes.
    
    Args:
        contexts: List of context dictionaries
        levels: Number of summarization levels
        
    Returns:
        Final summary
    """
    if not contexts:
        return ""
    
    current_text = "\n\n".join(c.get("content", "") for c in contexts)
    
    for level in range(levels):
        compression_ratio = 0.5 ** (level + 1)  # Each level compresses by half
        target_words = max(50, int(len(current_text.split()) * compression_ratio))
        
        prompt = f"""
Summarize the following text in approximately {target_words} words.
Level {level + 1} of {levels}.

Text:
{current_text}

Summary:"""

        current_text = call_llm([
            {"role": "system", "content": "You create multi-level summaries."},
            {"role": "user", "content": prompt}
        ])
    
    return current_text.strip()


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
