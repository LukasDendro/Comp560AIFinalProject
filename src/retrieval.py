"""K-NN Retrieval System"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Dict
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
import json


class ReviewRetriever:
    """K-NN based retrieval system for finding similar reviews"""
    
    def __init__(
        self, 
        embeddings_path: str,
        metadata_path: str,
        config_path: str,
        api_key: str = None,
        use_clustering: bool = True
    ):
        """Initialize the retrieval system"""
        self.client = OpenAI(api_key=api_key)
        self.use_clustering = use_clustering
        
        # Load embeddings
        print(f"Loading embeddings from {embeddings_path}")
        self.embeddings = np.load(embeddings_path)
        
        # Load metadata
        print(f"Loading metadata from {metadata_path}")
        self.metadata = pd.read_csv(metadata_path)
        
        # Load config
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.model = self.config['model']
        
        print(f"Loaded {len(self.embeddings)} embeddings")
        print(f"   Model: {self.model}")
        print(f"   Dimension: {self.embeddings.shape[1]}")
        
        # Load clustering data if enabled
        self.cluster_labels = None
        self.cluster_centers = None
        self.num_clusters = None
        
        if self.use_clustering:
            embeddings_dir = Path(embeddings_path).parent
            cluster_labels_path = embeddings_dir / "cluster_labels.npy"
            cluster_centers_path = embeddings_dir / "cluster_centers.npy"
            
            if cluster_labels_path.exists() and cluster_centers_path.exists():
                print(f"\nLoading clustering data...")
                self.cluster_labels = np.load(cluster_labels_path)
                self.cluster_centers = np.load(cluster_centers_path)
                self.num_clusters = self.config.get('num_clusters', len(self.cluster_centers))
                
                print(f"Cluster-aware retrieval enabled")
                print(f"   Number of clusters: {self.num_clusters}")
                print(f"   Cluster centers shape: {self.cluster_centers.shape}")
                print(f"   Cluster labels shape: {self.cluster_labels.shape}")
                
                # Show cluster distribution
                unique, counts = np.unique(self.cluster_labels, return_counts=True)
                print(f"   Cluster sizes: min={counts.min()}, max={counts.max()}, mean={counts.mean():.1f}")
            else:
                print(f"\nClustering files not found. Falling back to brute-force retrieval.")
                self.use_clustering = False
    
    def embed_query(self, query: str) -> np.ndarray:
        """Generate embedding for a query string"""
        response = self.client.embeddings.create(
            model=self.model,
            input=query
        )
        return np.array(response.data[0].embedding)
    
    def _find_top_clusters(
        self, 
        query_embedding: np.ndarray, 
        M: int = 5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Find top-M most similar clusters to the query"""
        # Ensure query_embedding is 2D
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Compute similarities to all cluster centers
        cluster_similarities = cosine_similarity(query_embedding, self.cluster_centers)[0]
        
        # Get top-M cluster indices
        top_cluster_indices = np.argsort(cluster_similarities)[-M:][::-1]
        top_cluster_scores = cluster_similarities[top_cluster_indices]
        
        return top_cluster_indices, top_cluster_scores
    
    def _clustered_retrieval(
        self, 
        query: str, 
        k: int = 20, 
        M: int = 5
    ) -> Tuple[pd.DataFrame, np.ndarray, Dict]:
        """Retrieve reviews using cluster-aware hierarchical search"""
        # Embed query
        query_embedding = self.embed_query(query)
        query_embedding_2d = query_embedding.reshape(1, -1)
        
        # Step 1: Find top-M clusters
        top_clusters, cluster_scores = self._find_top_clusters(query_embedding, M=M)
        
        print(f"Selected top {M} clusters:")
        for i, (cluster_id, score) in enumerate(zip(top_clusters, cluster_scores)):
            cluster_size = np.sum(self.cluster_labels == cluster_id)
            print(f"   Cluster {cluster_id}: {cluster_size} reviews (similarity: {score:.3f})")
        
        # Step 2: Get all review indices from selected clusters
        selected_indices = []
        for cluster_id in top_clusters:
            cluster_indices = np.where(self.cluster_labels == cluster_id)[0]
            selected_indices.extend(cluster_indices)
        
        selected_indices = np.array(selected_indices)
        total_candidates = len(selected_indices)
        print(f"Searching {total_candidates} reviews from {M} clusters (vs {len(self.embeddings)} total)")
        
        # Step 3: Compute similarities only for reviews in selected clusters
        selected_embeddings = self.embeddings[selected_indices]
        similarities = cosine_similarity(query_embedding_2d, selected_embeddings)[0]
        
        # Step 4: Get top-K reviews from the selected clusters
        # If k > total_candidates, return all candidates
        k_actual = min(k, total_candidates)
        top_k_local_indices = np.argsort(similarities)[-k_actual:][::-1]
        top_k_global_indices = selected_indices[top_k_local_indices]
        top_k_scores = similarities[top_k_local_indices]
        
        # Step 5: Get corresponding reviews with cluster information
        top_k_reviews = self.metadata.iloc[top_k_global_indices].copy()
        top_k_reviews['similarity_score'] = top_k_scores
        top_k_reviews['cluster_id'] = self.cluster_labels[top_k_global_indices]
        
        # Step 6: Build cluster metadata for pipeline display
        cluster_metadata = {
            'num_clusters_searched': M,
            'total_clusters': self.num_clusters,
            'searched_clusters': [],
            'total_reviews_searched': total_candidates,
            'total_reviews': len(self.embeddings)
        }
        
        # Add details for each searched cluster
        for cluster_id, score in zip(top_clusters, cluster_scores):
            cluster_size = int(np.sum(self.cluster_labels == cluster_id))
            reviews_retrieved = int(np.sum(top_k_reviews['cluster_id'] == cluster_id))
            cluster_metadata['searched_clusters'].append({
                'cluster_id': int(cluster_id),
                'similarity': float(score),
                'total_reviews': cluster_size,
                'reviews_retrieved': reviews_retrieved
            })
        
        return top_k_reviews, top_k_scores, cluster_metadata
    
    def retrieve_similar_reviews(
        self, 
        query: str, 
        k: int = 20,
        M: int = 5
    ) -> Tuple[pd.DataFrame, np.ndarray, Dict]:
        """Retrieve top-K most similar reviews to query"""
        print(f"\nQuery: '{query}'")
        print(f"Retrieving top {k} similar reviews...")
        
        cluster_metadata = None
        
        # Use cluster-aware retrieval if enabled
        if self.use_clustering:
            print(f"Using cluster-aware retrieval (searching top {M} clusters)")
            top_k_reviews, top_k_scores, cluster_metadata = self._clustered_retrieval(query, k=k, M=M)
        else:
            print(f"Using brute-force retrieval (searching all {len(self.embeddings)} reviews)")
            # Embed query
            query_embedding = self.embed_query(query)
            query_embedding = query_embedding.reshape(1, -1)
            
            # Calculate cosine similarities
            similarities = cosine_similarity(query_embedding, self.embeddings)[0]
            
            # Get top-K indices
            top_k_indices = np.argsort(similarities)[-k:][::-1]
            top_k_scores = similarities[top_k_indices]
            
            # Get corresponding reviews
            top_k_reviews = self.metadata.iloc[top_k_indices].copy()
            top_k_reviews['similarity_score'] = top_k_scores
        
        print(f"Retrieved {len(top_k_reviews)} reviews")
        print(f"   Similarity range: {top_k_scores.min():.3f} - {top_k_scores.max():.3f}")
        print(f"   Mean similarity: {top_k_scores.mean():.3f}")
        
        return top_k_reviews, top_k_scores, cluster_metadata
    
    def get_retrieval_summary(
        self, 
        retrieved_reviews: pd.DataFrame
    ) -> Dict:
        """Generate summary statistics for retrieved reviews"""
        return {
            "num_reviews": len(retrieved_reviews),
            "unique_products": retrieved_reviews['product_name'].nunique(),
            "products": retrieved_reviews['product_name'].value_counts().to_dict(),
            "avg_similarity": retrieved_reviews['similarity_score'].mean(),
            "min_similarity": retrieved_reviews['similarity_score'].min(),
            "max_similarity": retrieved_reviews['similarity_score'].max(),
            "avg_rating": retrieved_reviews['rating'].mean(),
            "rating_distribution": retrieved_reviews['rating'].value_counts().to_dict()
        }
    
    def get_cluster_retrieval_stats(
        self,
        retrieved_reviews: pd.DataFrame,
        query_embedding: np.ndarray
    ) -> Dict:
        """Generate cluster-level statistics for retrieved reviews"""
        if not self.use_clustering or 'cluster_id' not in retrieved_reviews.columns:
            return {"error": "Clustering not enabled or cluster_id not available"}
        
        # Ensure query_embedding is 2D
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Get unique clusters in results
        unique_clusters = retrieved_reviews['cluster_id'].unique()
        
        # Calculate cluster similarities
        cluster_similarities = {}
        for cluster_id in unique_clusters:
            cluster_center = self.cluster_centers[cluster_id]
            similarity = cosine_similarity(query_embedding, cluster_center.reshape(1, -1))[0][0]
            cluster_similarities[int(cluster_id)] = float(similarity)
        
        # Distribution of reviews across clusters
        cluster_distribution = retrieved_reviews['cluster_id'].value_counts().to_dict()
        cluster_distribution = {int(k): int(v) for k, v in cluster_distribution.items()}
        
        # Average similarity per cluster
        cluster_avg_similarity = {}
        for cluster_id in unique_clusters:
            cluster_reviews = retrieved_reviews[retrieved_reviews['cluster_id'] == cluster_id]
            cluster_avg_similarity[int(cluster_id)] = float(cluster_reviews['similarity_score'].mean())
        
        return {
            "num_clusters_represented": len(unique_clusters),
            "cluster_distribution": cluster_distribution,
            "cluster_center_similarities": cluster_similarities,
            "cluster_avg_review_similarity": cluster_avg_similarity,
            "total_clusters_searched": len(unique_clusters)
        }
    
    def format_reviews_for_llm(
        self, 
        retrieved_reviews: pd.DataFrame,
        max_reviews: int = 20
    ) -> str:
        """Format retrieved reviews for LLM consumption"""
        reviews_subset = retrieved_reviews.head(max_reviews)
        
        formatted_reviews = []
        for idx, row in reviews_subset.iterrows():
            review_text = f"""
Review #{idx + 1} (Similarity: {row['similarity_score']:.3f})
Product: {row['product_name']}
Rating: {row['rating']}/5
Review: {row['review_text']}
---"""
            formatted_reviews.append(review_text)
        
        return "\n".join(formatted_reviews)

