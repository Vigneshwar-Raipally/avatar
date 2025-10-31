# rag.py – Tekisho Research Assistant RAG Module with Supabase + Pinecone
import os
import json
import logging
from dotenv import load_dotenv
from typing import Optional, List, Dict, Any
from openai import OpenAI
 
# Load environment variables
load_dotenv()
 
# Environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "tekisho-rag")
PINECONE_ENV = os.getenv("PINECONE_ENV", "us-east1-aws")
 
# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TekishoRAG")
 
# Initialize clients
openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
pinecone_client = None
pinecone_index = None
 
def initialize_pinecone():
    """Initialize Pinecone client and index (lazy import)."""
    global pinecone_client, pinecone_index
   
    try:
        if not PINECONE_API_KEY:
            logger.warning("⚠️ PINECONE_API_KEY not set - RAG features will use fallback responses")
            return False
        
        # Lazy import Pinecone only when needed
        try:
            from pinecone import Pinecone
        except ImportError:
            logger.warning("⚠️ Pinecone library not installed - RAG features will use fallback responses")
            logger.warning("   Install with: pip install pinecone-client")
            return False
           
        pinecone_client = Pinecone(api_key=PINECONE_API_KEY)
        
        # Check if index exists
        existing_indexes = [index.name for index in pinecone_client.list_indexes()]
        if PINECONE_INDEX_NAME not in existing_indexes:
            logger.warning(f"⚠️ Pinecone index '{PINECONE_INDEX_NAME}' not found - RAG features will use fallback responses")
            logger.warning(f"   Available indexes: {existing_indexes}")
            return False
        
        pinecone_index = pinecone_client.Index(PINECONE_INDEX_NAME)
        logger.info(f"✅ Pinecone initialized successfully with index: {PINECONE_INDEX_NAME}")
        return True
    except Exception as e:
        logger.warning(f"⚠️ Pinecone initialization failed - RAG features will use fallback responses: {e}")
        return False
 
def get_embedding(text: str) -> List[float]:
    """Generate OpenAI embedding for text."""
    try:
        if not openai_client:
            logger.warning("OpenAI client not initialized - cannot generate embeddings")
            return []
        
        response = openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        logger.error(f"Failed to generate embedding: {e}")
        return []
 
def query_pinecone(query_text: str, top_k: int = 5, filter_dict: Optional[Dict] = None) -> List[Dict[str, Any]]:
    """Query Pinecone vector database for relevant documents."""
    global pinecone_index
   
    if not pinecone_index:
        if not initialize_pinecone():
            return []
   
    try:
        # Generate embedding for query
        query_embedding = get_embedding(query_text)
        if not query_embedding:
            return []
       
        # Query Pinecone
        query_params = {
            "vector": query_embedding,
            "top_k": top_k,
            "include_metadata": True
        }
       
        if filter_dict:
            query_params["filter"] = filter_dict
       
        results = pinecone_index.query(**query_params)
       
        # Extract relevant documents
        documents = []
        for match in results.get("matches", []):
            documents.append({
                "id": match.get("id"),
                "score": match.get("score"),
                "metadata": match.get("metadata", {}),
                "text": match.get("metadata", {}).get("text", "")
            })
       
        logger.info(f"Retrieved {len(documents)} documents from Pinecone")
        return documents
       
    except Exception as e:
        logger.error(f"Pinecone query failed: {e}")
        return []
 
def format_context_from_documents(documents: List[Dict[str, Any]]) -> str:
    """Format retrieved documents into context string for LLM."""
    if not documents:
        return ""
   
    context_parts = []
    for i, doc in enumerate(documents, 1):
        metadata = doc.get("metadata", {})
        text = doc.get("text", "")
       
        context_parts.append(f"[Document {i}]")
        context_parts.append(text)
        context_parts.append("")  # Empty line for readability
   
    return "\n".join(context_parts)
 
def get_tekisho_solutions(
    challenge: str,
    industry: Optional[str] = None,
    top_k: int = 5
) -> str:
    """
    Get Tekisho AI solutions for a specific business challenge using RAG.
   
    Args:
        challenge: The business challenge or problem
        industry: Optional industry context for more relevant results
        top_k: Number of relevant documents to retrieve
       
    Returns:
        Conversational response with solutions and metrics
    """
    try:
        # Build enhanced query with industry context
        if industry:
            query_text = f"{industry} industry: {challenge}"
            filter_dict = {"industry": industry}
        else:
            query_text = challenge
            filter_dict = None
       
        # Query vector database
        relevant_docs = query_pinecone(query_text, top_k=top_k, filter_dict=filter_dict)
       
        if not relevant_docs:
            logger.warning("No relevant documents found in RAG, using fallback response")
            return generate_fallback_response(challenge, industry)
       
        # Format context for LLM
        context = format_context_from_documents(relevant_docs)
       
        # Generate response using GPT with RAG context
        system_prompt = """You are Aria, an AI solutions expert at Tekisho.
        Use the provided context to answer questions about AI solutions, automation, and business challenges.
        Provide specific metrics, ROI numbers, and implementation timelines when available.
        Be conversational, helpful, and focus on practical business value.
        If the context doesn't contain specific information, acknowledge that and provide general guidance."""
       
        user_prompt = f"""Based on the following context, provide a solution for this challenge:
 
Challenge: {challenge}
{f"Industry: {industry}" if industry else ""}
 
Context from Tekisho knowledge base:
{context}
 
Provide a conversational response that:
1. Addresses the specific challenge
2. Mentions relevant AI solutions or technologies
3. Includes metrics like ROI, cost savings, or productivity improvements (if available in context)
4. Suggests implementation approach or timeline
5. Keeps it natural and conversational (not bullet points)"""
       
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
            max_tokens=300
        )
       
        answer = response.choices[0].message.content.strip()
        logger.info(f"Generated RAG-powered solution for: {challenge}")
        return answer
       
    except Exception as e:
        logger.error(f"RAG solution generation failed: {e}")
        return generate_fallback_response(challenge, industry)
 
def generate_fallback_response(challenge: str, industry: Optional[str] = None) -> str:
    """Generate a fallback response when RAG is unavailable."""
    response = f"I understand you're facing challenges with {challenge}. "
   
    if industry:
        response += f"For businesses in the {industry} industry, "
   
    response += (
        "Tekisho's AI solutions typically deliver significant results. "
        "Our clients see ROI ranging from 150 to 300 percent within 6 to 12 weeks. "
        "Common benefits include cost savings of 25 to 40 percent and productivity improvements of 50 to 80 percent. "
        "I'd be happy to connect you with our solution architects who can provide specific recommendations "
        "tailored to your needs. Would that be helpful?"
    )
   
    return response
 
def add_document_to_knowledge_base(
    text: str,
    metadata: Dict[str, Any],
    doc_id: Optional[str] = None
) -> bool:
    """
    Add a new document to the Pinecone knowledge base.
   
    Args:
        text: Document text content
        metadata: Metadata dictionary (should include relevant fields like industry, solution_type, etc.)
        doc_id: Optional document ID (auto-generated if not provided)
       
    Returns:
        True if successful, False otherwise
    """
    global pinecone_index
   
    if not pinecone_index:
        if not initialize_pinecone():
            return False
   
    try:
        # Generate embedding
        embedding = get_embedding(text)
        if not embedding:
            return False
       
        # Add text to metadata
        metadata["text"] = text
       
        # Generate doc_id if not provided
        if not doc_id:
            import uuid
            doc_id = str(uuid.uuid4())
       
        # Upsert to Pinecone
        pinecone_index.upsert(
            vectors=[{
                "id": doc_id,
                "values": embedding,
                "metadata": metadata
            }]
        )
       
        logger.info(f"Successfully added document {doc_id} to knowledge base")
        return True
       
    except Exception as e:
        logger.error(f"Failed to add document to knowledge base: {e}")
        return False
 
# Initialize on module load
initialize_pinecone()
 
if __name__ == "__main__":
    # Test the RAG system
    print("Testing Tekisho RAG System...")
   
    test_challenge = "improving customer service response times"
    test_industry = "retail"
   
    solution = get_tekisho_solutions(test_challenge, test_industry)
    print(f"\nChallenge: {test_challenge}")
    print(f"Industry: {test_industry}")
    print(f"\nSolution:\n{solution}")