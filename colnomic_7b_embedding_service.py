#!/usr/bin/env python3
"""
ColNomic 7B Embedding Service

FastAPI service that provides ColNomic embeddings using nomic-ai/colnomic-embed-multimodal-7b
with ColQwen2_5_Processor from colpali engine

Based on Nomic AI's state-of-the-art multimodal embedding model that achieves 
62.7 NDCG@5 on Vidore-v2 benchmark.
"""

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import torch
import numpy as np
import uvicorn
import io
import logging
from PIL import Image

# Import ColPali engine components for ColNomic
try:
    from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor
    COLPALI_AVAILABLE = True
except ImportError:
    try:
        # Alternative import path if the above doesn't work
        from colpali_engine.models.colqwen2_5 import ColQwen2_5, ColQwen2_5_Processor
        COLPALI_AVAILABLE = True
    except ImportError:
        COLPALI_AVAILABLE = False
        print("❌ ColPali engine not available. Please install: pip install colpali-engine")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="ColNomic 7B Embedding API", 
    description="API for generating multi-vector embeddings using nomic-ai/colnomic-embed-multimodal-7b",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model and processor
colnomic_model = None
colnomic_processor = None

class EmbeddingResponse(BaseModel):
    embeddings: List[List[List[float]]]  # Multi-vector embeddings
    status: str
    model_info: dict

class QueryEmbeddingResponse(BaseModel):
    embedding: List[List[float]]  # Multi-vector query embedding
    status: str
    model_info: dict

class QueryRequest(BaseModel):
    query: str

@app.on_event("startup")
async def startup_event():
    global colnomic_model, colnomic_processor
    
    if not COLPALI_AVAILABLE:
        logger.error("ColPali engine not available. Cannot load ColNomic model.")
        raise RuntimeError("ColPali engine required for ColNomic model")
    
    try:
        # Load ColNomic 7B model and ColQwen2_5_Processor
        logger.info("Loading ColNomic 7B model and ColQwen2_5_Processor...")
        
        model_name = "nomic-ai/colnomic-embed-multimodal-7b"
        
        # Load the model using ColQwen2_5 architecture
        colnomic_model = ColQwen2_5.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="cuda:0" if torch.cuda.is_available() else "cpu",
            trust_remote_code=True  # Required for Nomic models
        )
        
        # Load the ColQwen2_5_Processor
        colnomic_processor = ColQwen2_5_Processor.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        logger.info(f"ColNomic 7B model loaded successfully on device: {colnomic_model.device}")
        logger.info(f"Model: {model_name}")
        logger.info(f"Processor: ColQwen2_5_Processor")
        logger.info(f"Max visual tokens: {getattr(colnomic_processor, 'max_num_visual_tokens', 'Unknown')}")
        
    except Exception as e:
        logger.error(f"Error loading ColNomic 7B model: {str(e)}")
        logger.error("Make sure you have the latest colpali-engine installed:")
        logger.error("pip install colpali-engine --upgrade")
        raise e

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": colnomic_model is not None,
        "device": str(colnomic_model.device) if colnomic_model else "not loaded",
        "model_name": "nomic-ai/colnomic-embed-multimodal-7b",
        "processor": "ColQwen2_5_Processor",
        "colpali_available": COLPALI_AVAILABLE
    }

@app.post("/embed_images", response_model=EmbeddingResponse)
async def embed_images(files: List[UploadFile] = File(...)):
    """
    Generate ColNomic multi-vector embeddings for uploaded images
    
    Args:
        files: List of image files to embed
        
    Returns:
        EmbeddingResponse with multi-vector embeddings for each image
    """
    if not colnomic_model or not colnomic_processor:
        raise HTTPException(status_code=500, detail="ColNomic model not loaded")
    
    try:
        # Convert uploaded files to PIL Images
        images = []
        for file in files:
            content = await file.read()
            image = Image.open(io.BytesIO(content)).convert("RGB")
            images.append(image)
        
        logger.info(f"Processing {len(images)} images for embedding with ColNomic 7B")
        
        # Process images with ColNomic using ColQwen2_5_Processor
        with torch.no_grad():
            batch_images = colnomic_processor.process_images(images).to(colnomic_model.device)
            image_embeddings = colnomic_model(**batch_images)
        
        # Convert embeddings to list format (multi-vector)
        embeddings_list = []
        for embedding in image_embeddings:
            # Each embedding is a multi-vector representation
            multivector = embedding.cpu().float().numpy().tolist()
            embeddings_list.append(multivector)
        
        logger.info(f"Generated embeddings for {len(embeddings_list)} images")
        logger.info(f"Embedding shape per image: {len(embeddings_list[0])} vectors x {len(embeddings_list[0][0])} dimensions")
        
        return EmbeddingResponse(
            embeddings=embeddings_list,
            status="success",
            model_info={
                "model_name": "nomic-ai/colnomic-embed-multimodal-7b",
                "processor": "ColQwen2_5_Processor",
                "embedding_type": "multi_vector",
                "vector_count_per_image": len(embeddings_list[0]) if embeddings_list else 0,
                "vector_dimension": len(embeddings_list[0][0]) if embeddings_list and embeddings_list[0] else 0,
                "performance_note": "State-of-the-art 62.7 NDCG@5 on Vidore-v2"
            }
        )
        
    except Exception as e:
        logger.error(f"Error processing images: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing images: {str(e)}")

@app.post("/embed_query", response_model=QueryEmbeddingResponse)
async def embed_query(request: QueryRequest):
    """
    Generate ColNomic multi-vector embedding for a text query
    
    Args:
        request: QueryRequest containing the text query
        
    Returns:
        QueryEmbeddingResponse with multi-vector query embedding
    """
    if not colnomic_model or not colnomic_processor:
        raise HTTPException(status_code=500, detail="ColNomic model not loaded")
    
    try:
        query_text = request.query
        logger.info(f"Processing query: '{query_text}' with ColNomic 7B")
        
        # Process query with ColNomic using ColQwen2_5_Processor
        with torch.no_grad():
            batch_query = colnomic_processor.process_queries([query_text]).to(colnomic_model.device)
            query_embedding = colnomic_model(**batch_query)
        
        # Convert to list format (multi-vector)
        multivector_query = query_embedding[0].cpu().float().numpy().tolist()
        
        logger.info(f"Generated query embedding with {len(multivector_query)} vectors x {len(multivector_query[0])} dimensions")
        
        return QueryEmbeddingResponse(
            embedding=multivector_query,
            status="success",
            model_info={
                "model_name": "nomic-ai/colnomic-embed-multimodal-7b",
                "processor": "ColQwen2_5_Processor",
                "embedding_type": "multi_vector",
                "vector_count": len(multivector_query),
                "vector_dimension": len(multivector_query[0]) if multivector_query else 0,
                "performance_note": "State-of-the-art 62.7 NDCG@5 on Vidore-v2"
            }
        )
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.get("/model_info")
async def get_model_info():
    """Get information about the loaded ColNomic model"""
    if not colnomic_model:
        return {"status": "Model not loaded"}
    
    return {
        "model_loaded": True,
        "device": str(colnomic_model.device),
        "model_name": "nomic-ai/colnomic-embed-multimodal-7b",
        "model_type": "ColNomic (ColQwen2.5 architecture)",
        "processor": "ColQwen2_5_Processor",
        "embedding_type": "multi_vector",
        "torch_dtype": str(colnomic_model.dtype) if hasattr(colnomic_model, 'dtype') else "unknown",
        "max_visual_tokens": getattr(colnomic_processor, 'max_num_visual_tokens', 'Unknown') if colnomic_processor else 'Unknown',
        "performance_benchmark": "62.7 NDCG@5 on Vidore-v2",
        "advantages": [
            "State-of-the-art performance on visual document retrieval",
            "Improved hard negative mining",
            "Better sampling from same source",
            "Enhanced multi-vector late interaction"
        ]
    }

if __name__ == "__main__":
    uvicorn.run(
        "colnomic_7b_embedding_service:app",
        host="0.0.0.0",
        port=8001,  # Different port to avoid conflicts
        reload=False,  # Set to False in production
        log_level="info"
    ) 