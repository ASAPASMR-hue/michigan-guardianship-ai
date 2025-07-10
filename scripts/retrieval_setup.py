#!/usr/bin/env python3
"""
retrieval_setup.py - Hybrid Search Setup for Michigan Guardianship AI
Implements query complexity classification and hybrid retrieval with reranking
"""

import os
import sys
import yaml
import pickle
import time
from pathlib import Path
from typing import List, Dict, Tuple
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
import numpy as np
from collections import Counter
import re
from scipy.sparse import hstack

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from scripts.log_step import log_step

# Configuration paths
CONFIG_DIR = Path(__file__).parent.parent / "config"
CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"
CONSTANTS_DIR = Path(__file__).parent.parent / "constants"

class QueryComplexityClassifier:
    """
    Classifies queries into complexity tiers for adaptive retrieval
    with strict latency budgets. Uses trained ML model with fallback.
    """
    def __init__(self):
        self.complexity_tiers = {
            "simple": {
                "examples": ["filing fee?", "what form?", "court address?"],
                "top_k": 5,
                "query_rewrites": 0,
                "rerank_top_k": 3,
                "latency_budget_ms": 800,
                "latency_p95_ms": 1000
            },
            "standard": {
                "examples": ["grandma guardianship", "parent consent"],
                "top_k": 10,
                "query_rewrites": 3,
                "rerank_top_k": 5,
                "latency_budget_ms": 1500,
                "latency_p95_ms": 1800
            },
            "complex": {
                "examples": ["ICWA applies", "emergency + out of state"],
                "top_k": 15,
                "query_rewrites": 5,
                "rerank_top_k": 7,
                "latency_budget_ms": 2000,
                "latency_p95_ms": 2500,
                "fallback_if_slow": {
                    "top_k": 10,
                    "query_rewrites": 3
                }
            }
        }
        
        # Try to load trained model
        self.model_loaded = False
        self.model = None
        self.tfidf = None
        self.label_encoder = None
        self.complexity_keywords = None
        self.confidence = 0.7  # Default confidence
        
        model_path = Path(__file__).parent.parent / "models" / "complexity_classifier.pkl"
        if model_path.exists():
            try:
                with open(model_path, 'rb') as f:
                    model_data = pickle.load(f)
                
                self.model = model_data['classifier']
                self.tfidf = model_data['tfidf']
                self.label_encoder = model_data['label_encoder']
                self.complexity_keywords = model_data['complexity_keywords']
                self.model_loaded = True
                print("Loaded trained complexity classifier")
            except Exception as e:
                print(f"Warning: Could not load trained model: {e}")
                self.model_loaded = False
        
        # Fallback keywords if model not loaded
        if not self.model_loaded:
            self.complex_keywords = ["icwa", "tribal", "emergency", "multi-state", "contested", "cps"]
            self.standard_keywords = ["parent", "grandparent", "terminate", "modify", "move"]
    
    def _extract_features(self, questions):
        """Extract features for ML model (mirrors training code)"""
        # TF-IDF features
        tfidf_features = self.tfidf.transform(questions)
        
        # Keyword features
        keyword_features = []
        for question in questions:
            question_lower = question.lower()
            features = []
            
            # Count keywords for each complexity level
            for tier, keywords in self.complexity_keywords.items():
                count = sum(1 for kw in keywords if kw in question_lower)
                features.append(count)
            
            # Additional features
            features.extend([
                len(question.split()),  # Word count
                question.count('?'),    # Question marks
                question.count('MCL'),  # Legal citations
                question.count('PC'),   # Form references
                'emergency' in question_lower or 'urgent' in question_lower,  # Urgency
                'icwa' in question_lower or 'tribal' in question_lower,       # ICWA
            ])
            
            keyword_features.append(features)
        
        keyword_features = np.array(keyword_features)
        
        # Combine features
        from scipy.sparse import hstack
        combined_features = hstack([tfidf_features, keyword_features])
        
        return combined_features
    
    def classify(self, query: str) -> str:
        """Classify query complexity using trained model or fallback"""
        if self.model_loaded:
            try:
                # Use trained model
                features = self._extract_features([query])
                prediction = self.model.predict(features)[0]
                self.confidence = self.model.predict_proba(features).max()
                tier = self.label_encoder.inverse_transform([prediction])[0]
                return tier
            except Exception as e:
                print(f"Model prediction failed, using fallback: {e}")
        
        # Fallback to keyword-based classification
        query_lower = query.lower()
        self.confidence = 0.7  # Default confidence for keyword-based
        
        # Check for complex keywords
        if any(kw in query_lower for kw in self.complex_keywords):
            self.confidence = 0.8
            return "complex"
        
        # Check for standard keywords or questions
        if any(kw in query_lower for kw in self.standard_keywords) or "?" in query:
            if len(query.split()) > 15:
                self.confidence = 0.75
                return "complex"
            return "standard"
        
        # Short queries are usually simple
        if len(query.split()) <= 5:
            self.confidence = 0.85
            return "simple"
        
        return "standard"
    
    def get_params(self, complexity: str) -> Dict:
        """Get retrieval parameters for given complexity"""
        return self.complexity_tiers.get(complexity, self.complexity_tiers["standard"])

class HybridRetriever:
    """Hybrid search combining vector and lexical search with reranking"""
    
    def __init__(self):
        # Load configurations
        self.load_configs()
        
        # Initialize models
        self.init_models()
        
        # Initialize ChromaDB
        self.init_chromadb()
        
        # Initialize BM25
        self.init_bm25()
        
        # Initialize complexity classifier
        self.classifier = QueryComplexityClassifier()
    
    def load_configs(self):
        """Load configuration files"""
        with open(CONFIG_DIR / "retrieval_pipeline.yaml", "r") as f:
            self.pipeline_config = yaml.safe_load(f)
        
        with open(CONFIG_DIR / "embedding.yaml", "r") as f:
            self.embedding_config = yaml.safe_load(f)
        
        with open(CONSTANTS_DIR / "genesee.yaml", "r") as f:
            self.genesee_constants = yaml.safe_load(f)
    
    def init_models(self):
        """Initialize embedding and reranking models"""
        # Use small model if flag is set
        if os.getenv('USE_SMALL_MODEL', 'false').lower() == 'true':
            print("Using small models for development/testing")
            embed_model = 'all-MiniLM-L6-v2'
            rerank_model = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
        else:
            embed_model = self.embedding_config['primary_model']
            rerank_model = 'BAAI/bge-reranker-v2-m3'
        
        print(f"Loading embedding model: {embed_model}")
        self.embed_model = SentenceTransformer(embed_model)
        
        print(f"Loading reranking model: {rerank_model}")
        # Set HF token for CrossEncoder
        hf_token = os.getenv('HF_TOKEN', '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')
        os.environ['HF_TOKEN'] = hf_token
        
        # CrossEncoder needs trust_remote_code for some models
        self.reranker = CrossEncoder(rerank_model, trust_remote_code=True)
    
    def init_chromadb(self):
        """Initialize ChromaDB connection"""
        self.chroma_client = chromadb.PersistentClient(
            path=str(CHROMA_DIR),
            settings=Settings(anonymized_telemetry=False)
        )
        self.collection = self.chroma_client.get_collection("michigan_guardianship_v2")
        
        # Get all documents for BM25
        print("Loading documents from ChromaDB...")
        results = self.collection.get()
        self.all_docs = results['documents']
        self.all_ids = results['ids']
        self.all_metadata = results['metadatas']
        print(f"Loaded {len(self.all_docs)} documents")
    
    def init_bm25(self):
        """Initialize BM25 for lexical search"""
        print("Initializing BM25...")
        # Tokenize documents
        tokenized_docs = [doc.lower().split() for doc in self.all_docs]
        self.bm25 = BM25Okapi(tokenized_docs)
    
    def generate_query_rewrites(self, query: str, num_rewrites: int) -> List[str]:
        """Generate query variations for better retrieval"""
        rewrites = [query]  # Include original
        
        if num_rewrites == 0:
            return rewrites
        
        # Simple rewrite strategies
        variations = []
        
        # Add question form
        if "?" not in query:
            variations.append(f"What is {query}?")
            variations.append(f"How to {query}?")
        
        # Add context
        variations.append(f"{query} Michigan guardianship")
        variations.append(f"{query} Genesee County")
        
        # Add common variants
        if "fee" in query.lower():
            variations.append("filing cost")
            variations.append("court fees waiver MC 20")
        
        if "form" in query.lower():
            variations.append("PC forms guardianship")
            variations.append("petition documents")
        
        # Return requested number of rewrites
        rewrites.extend(variations[:num_rewrites])
        return rewrites
    
    def hybrid_search(self, query: str, top_k: int = 10, 
                     vector_weight: float = 0.7) -> List[Dict]:
        """Perform hybrid search combining vector and BM25"""
        
        # Vector search
        query_embedding = self.embed_model.encode(query, normalize_embeddings=True)
        vector_results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k * 2,  # Get more for merging
            where={"jurisdiction": "Genesee County"}  # Mandatory filter
        )
        
        # BM25 search
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        
        # Get top BM25 results
        bm25_top_indices = np.argsort(bm25_scores)[-top_k*2:][::-1]
        
        # Combine results
        combined_scores = {}
        
        # Add vector results
        for i, (doc_id, distance) in enumerate(zip(
            vector_results['ids'][0], 
            vector_results['distances'][0]
        )):
            # Convert distance to similarity score (1 - distance for cosine)
            similarity = 1 - distance
            combined_scores[doc_id] = vector_weight * similarity
        
        # Add BM25 scores
        lexical_weight = 1 - vector_weight
        for idx in bm25_top_indices:
            doc_id = self.all_ids[idx]
            # Check Genesee filter
            if self.all_metadata[idx].get('jurisdiction') == 'Genesee County':
                normalized_score = bm25_scores[idx] / (max(bm25_scores) + 1e-6)
                if doc_id in combined_scores:
                    combined_scores[doc_id] += lexical_weight * normalized_score
                else:
                    combined_scores[doc_id] = lexical_weight * normalized_score
        
        # Sort by combined score
        sorted_ids = sorted(combined_scores.keys(), 
                          key=lambda x: combined_scores[x], 
                          reverse=True)[:top_k]
        
        # Get full documents
        results = []
        for doc_id in sorted_ids:
            idx = self.all_ids.index(doc_id)
            results.append({
                'id': doc_id,
                'document': self.all_docs[idx],
                'metadata': self.all_metadata[idx],
                'score': combined_scores[doc_id]
            })
        
        return results
    
    def rerank(self, query: str, documents: List[Dict], top_k: int) -> List[Dict]:
        """Rerank documents using cross-encoder"""
        if not documents:
            return documents
        
        # Prepare pairs for reranking
        pairs = [[query, doc['document']] for doc in documents]
        
        # Get reranking scores
        scores = self.reranker.predict(pairs)
        
        # Add scores to documents
        for i, doc in enumerate(documents):
            doc['rerank_score'] = float(scores[i])
        
        # Sort by rerank score
        reranked = sorted(documents, key=lambda x: x['rerank_score'], reverse=True)
        
        return reranked[:top_k]
    
    def retrieve(self, query: str) -> Tuple[List[Dict], Dict]:
        """Main retrieval function with complexity-aware parameters"""
        # Classify query
        complexity = self.classifier.classify(query)
        confidence = getattr(self.classifier, 'confidence', 0.7)
        params = self.classifier.get_params(complexity)
        
        print(f"\nQuery: '{query}'")
        print(f"Complexity: {complexity} (confidence: {confidence:.2f})")
        print(f"Parameters: top_k={params['top_k']}, rewrites={params['query_rewrites']}")
        
        # Generate query rewrites
        rewrites = self.generate_query_rewrites(query, params['query_rewrites'])
        
        # Perform hybrid search with each rewrite
        all_results = []
        for rewrite in rewrites:
            results = self.hybrid_search(rewrite, params['top_k'])
            all_results.extend(results)
        
        # Deduplicate by ID
        seen_ids = set()
        unique_results = []
        for result in all_results:
            if result['id'] not in seen_ids:
                seen_ids.add(result['id'])
                unique_results.append(result)
        
        # Rerank
        final_results = self.rerank(query, unique_results, params['rerank_top_k'])
        
        # Return results and metadata
        metadata = {
            'complexity': complexity,
            'complexity_confidence': confidence,
            'num_rewrites': len(rewrites),
            'params': params
        }
        
        return final_results, metadata

def test_retrieval():
    """Test the retrieval system with sample queries"""
    retriever = HybridRetriever()
    
    test_queries = [
        # Simple
        "filing fee?",
        "court address",
        "what form",
        
        # Standard  
        "grandparent seeking guardianship",
        "parent wants to terminate guardianship",
        
        # Complex
        "ICWA emergency guardianship out of state parent",
        "native american guardianship"
    ]
    
    print("\n=== Testing Hybrid Retrieval ===")
    
    for query in test_queries:
        results, metadata = retriever.retrieve(query)
        
        print(f"\nTop {len(results)} results:")
        for i, result in enumerate(results):
            print(f"\n{i+1}. Source: {result['metadata']['source']}")
            print(f"   Type: {result['metadata']['doc_type']}")
            print(f"   Combined Score: {result['score']:.3f}")
            print(f"   Rerank Score: {result['rerank_score']:.3f}")
            print(f"   Text: {result['document'][:150]}...")

def main():
    """Main execution function"""
    log_step("Starting retrieval setup", "Initializing hybrid search system", "Per Part A.3")
    
    # Test retrieval
    test_retrieval()
    
    log_step("Retrieval testing complete", "Verified hybrid search with complexity classification", "Quality assurance")
    
    print("\n✓ Retrieval setup completed successfully!")

if __name__ == "__main__":
    main()