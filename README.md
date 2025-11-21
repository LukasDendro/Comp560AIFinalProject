# AI-Augmented Root Problem Discovery System

A proof-of-concept system that uses semantic analysis and large language models to help innovators discover root user problems from product review data.

## Motivation

Traditional design thinking begins with discovering root problems—the fundamental user needs underlying surface-level complaints. This process typically requires extensive qualitative research, interviews, and careful analysis. This project explores whether AI can augment this discovery process by analyzing large volumes of user reviews to surface deeper insights.

**Example transformation:**
- Surface complaint: "Earbuds slip out during jogging"
- Root problem: "Users need confidence their device stays secure without conscious attention during physical activity"

## How It Works

The system implements a three-stage pipeline combining semantic search with LLM-powered synthesis:

### 1. Query Embedding
User queries about problem spaces are embedded using OpenAI's `text-embedding-3-small` model, transforming natural language into a high-dimensional semantic representation.

### 2. Cluster-Aware Retrieval
Rather than searching all reviews, the system uses hierarchical clustering to identify the most relevant semantic regions:
- K-means clustering (100 clusters) groups similar reviews in embedding space
- Query is compared against cluster centers to identify top-M most relevant clusters
- K-NN search retrieves the 20 most similar reviews from selected clusters only
- This provides ~40x computational speedup while maintaining retrieval quality

### 3. Root Problem Synthesis
Retrieved reviews are processed by GPT-4o mini using structured few-shot prompting to generate 2-3 root problem hypotheses. Each includes:
- A clear problem statement
- Supporting evidence from actual user reviews
- Explanation of why it represents a root (not surface) problem

## Results

Evaluated across 15 diverse test queries spanning physical activity, battery life, connectivity, comfort, sound quality, and more:

- **Average retrieval similarity**: 0.51 (reviews highly relevant to queries)
- **Search efficiency**: 40x faster than brute-force with cluster-aware retrieval
- **Output quality**: Consistent structured root problems with evidence-backed insights
- **Coverage**: System surfaces patterns across multiple products and user contexts

The system successfully transforms specific user complaints into broader, actionable problem statements suitable for innovation work.

## Project Structure

```
.
├── src/
│   ├── embeddings.py      # Embedding generation and clustering
│   ├── retrieval.py       # K-NN similarity search with clustering
│   ├── synthesis.py       # LLM-based root problem synthesis
│   └── pipeline.py        # End-to-end integration
├── tests/
│   ├── test_queries.json  # 15 evaluation queries
│   └── run_evaluation.py  # Evaluation harness
├── scripts/
│   ├── clean_earphones_subspace.py    # Data preprocessing
│   └── split_products_into_csv.py     # Product data splitting
├── data/
│   ├── processed/products/  # Individual product review CSVs
│   └── embeddings/          # Generated embeddings and clusters
├── discover.py            # Command-line interface
└── requirements.txt       # Python dependencies
```

## Technical Implementation

**Core Technologies:**
- OpenAI API (text-embedding-3-small, GPT-4o mini)
- scikit-learn (K-means clustering, cosine similarity)
- pandas (data processing)
- Python 3.8+

**Key Design Decisions:**
- Cluster-aware retrieval balances speed and quality for production feasibility
- Few-shot prompting ensures consistent, structured LLM outputs
- Evidence extraction maintains traceability to source reviews
- Modular architecture separates embedding, retrieval, and synthesis concerns

## Limitations & Future Work

- Dataset limited to Bluetooth earphones/headphones reviews
- Root problem quality depends on LLM prompt engineering
- No validation against human expert problem identification
- Could benefit from multi-stage retrieval refinement

---

**Course Project**: COMP 560 - AI Final Project  
