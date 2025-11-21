"""Embedding Generation Pipeline"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple
import json
from openai import OpenAI
from tqdm import tqdm
import time
from sklearn.cluster import KMeans


class EmbeddingGenerator:
    """Generate and manage embeddings for product reviews using OpenAI API"""
    
    def __init__(self, api_key: str = None, model: str = "text-embedding-3-small"):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.embedding_dim = 1536
        
    def load_all_reviews(
        self, 
        products_dir: str, 
        max_per_product: int = 400,
        min_word_count: int = 15,
        stratify_by_rating: bool = True
    ) -> pd.DataFrame:
        """Load and sample product review CSVs into a balanced DataFrame"""
        products_path = Path(products_dir)
        all_reviews = []
        
        print(f"Loading and sampling reviews from {products_dir}")
        print(f"Settings: max_per_product={max_per_product}, min_words={min_word_count}")
        
        total_before = 0
        total_after = 0
        
        for csv_file in sorted(products_path.glob("*.csv")):
            df = pd.read_csv(csv_file)
            original_count = len(df)
            total_before += original_count
            
            # Calculate word count for filtering
            df['word_count'] = df['review_text'].astype(str).str.split().str.len()
            
            # Filter by minimum word count
            df_filtered = df[df['word_count'] >= min_word_count].copy()
            
            # Sample reviews
            if len(df_filtered) <= max_per_product:
                # Use all if fewer than max
                sampled_df = df_filtered
            else:
                if stratify_by_rating:
                    # Stratified sampling: maintain rating distribution
                    sampled_df = df_filtered.groupby('rating', group_keys=False).apply(
                        lambda x: x.sample(
                            n=min(len(x), max(1, int(max_per_product * len(x) / len(df_filtered)))),
                            random_state=42
                        ),
                        include_groups=False
                    )
                    # Only sample again if we got more than max_per_product
                    if len(sampled_df) > max_per_product:
                        sampled_df = sampled_df.sample(n=max_per_product, random_state=42)
                else:
                    # Random sampling
                    sampled_df = df_filtered.sample(n=max_per_product, random_state=42)
            
            # Drop the word_count column before adding to final dataset
            sampled_df = sampled_df.drop(columns=['word_count'])
            
            all_reviews.append(sampled_df)
            total_after += len(sampled_df)
            
            print(f"{csv_file.name:40s} {original_count:5d} -> {len(df_filtered):5d} (filtered) -> {len(sampled_df):5d} (sampled)")
        
        combined_df = pd.concat(all_reviews, ignore_index=True)
        
        print(f"\nSampling complete!")
        print(f"Total before: {total_before:,} reviews")
        print(f"Total after:  {len(combined_df):,} reviews ({len(combined_df)/total_before*100:.1f}% of original)")
        
        return combined_df
    
    def generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for a single text using OpenAI API"""
        # Clean the text
        text = str(text).strip()
        if not text:
            return np.zeros(self.embedding_dim)
        
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=text
            )
            embedding = response.data[0].embedding
            return np.array(embedding)
        
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return np.zeros(self.embedding_dim)
    
    def generate_batch_embeddings(
        self, 
        texts, 
        batch_size: int = 100,
        delay: float = 0.1
    ) -> np.ndarray:
        """Generate embeddings for a batch of texts with rate limiting"""
        embeddings = []
        
        print(f"Generating embeddings for {len(texts)} reviews...")
        
        for i in tqdm(range(0, len(texts), batch_size)):
            batch = texts[i:i + batch_size]
            
            # Clean batch
            batch = [str(text).strip() for text in batch]
            batch = [text if text else " " for text in batch]  # OpenAI requires non-empty
            
            try:
                response = self.client.embeddings.create(
                    model=self.model,
                    input=batch
                )
                
                batch_embeddings = [item.embedding for item in response.data]
                embeddings.extend(batch_embeddings)
                
                # Rate limiting
                time.sleep(delay)
                
            except Exception as e:
                print(f"\nError in batch {i//batch_size}: {e}")
                embeddings.extend([np.zeros(self.embedding_dim).tolist()] * len(batch))
        
        return np.array(embeddings)
    
    def cluster_embeddings(
        self, 
        embeddings: np.ndarray, 
        n_clusters: int = 25,
        random_state: int = 42
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Cluster embeddings using K-Means algorithm"""
        print(f"\nClustering {len(embeddings)} embeddings into {n_clusters} clusters...")
        
        # Initialize and fit K-Means
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            n_init=10,
            max_iter=300,
            verbose=0
        )
        
        # Fit and predict
        cluster_labels = kmeans.fit_predict(embeddings)
        cluster_centers = kmeans.cluster_centers_
        
        print(f"Clustering complete!")
        print(f"Algorithm: K-Means, Clusters: {n_clusters}, Inertia: {kmeans.inertia_:.2f}")
        
        return cluster_labels, cluster_centers
    
    def get_cluster_statistics(self, cluster_labels: np.ndarray) -> Dict:
        """Compute and display clustering statistics"""
        unique_clusters, counts = np.unique(cluster_labels, return_counts=True)
        n_clusters = len(unique_clusters)
        
        stats = {
            "num_clusters": n_clusters,
            "min_size": int(counts.min()),
            "max_size": int(counts.max()),
            "mean_size": float(counts.mean()),
            "std_size": float(counts.std()),
            "total_reviews": int(len(cluster_labels))
        }
        
        print(f"\nCluster Statistics:")
        print(f"Number of clusters: {stats['num_clusters']}")
        print(f"Total reviews: {stats['total_reviews']:,}")
        print(f"Cluster size - Min: {stats['min_size']}, Max: {stats['max_size']}, Mean: {stats['mean_size']:.1f}, Std: {stats['std_size']:.1f}")
        
        return stats
    
    def generate_and_save_embeddings(
        self,
        reviews_df: pd.DataFrame,
        output_path: str,
        batch_size: int = 100,
        n_clusters: int = 25
    ) -> Dict:
        """Generate embeddings for all reviews, cluster them, and save to disk"""
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Generate embeddings
        review_texts = reviews_df['review_text'].tolist()
        embeddings = self.generate_batch_embeddings(review_texts, batch_size=batch_size)
        
        # Save embeddings
        embeddings_file = output_path / "review_embeddings.npy"
        np.save(embeddings_file, embeddings)
        print(f"\nSaved embeddings to {embeddings_file}")
        
        # Cluster embeddings
        cluster_labels, cluster_centers = self.cluster_embeddings(
            embeddings, 
            n_clusters=n_clusters,
            random_state=42
        )
        
        # Save cluster labels
        cluster_labels_file = output_path / "cluster_labels.npy"
        np.save(cluster_labels_file, cluster_labels)
        print(f"Saved cluster labels to {cluster_labels_file}")
        
        # Save cluster centers
        cluster_centers_file = output_path / "cluster_centers.npy"
        np.save(cluster_centers_file, cluster_centers)
        print(f"Saved cluster centers to {cluster_centers_file}")
        
        # Get and display cluster statistics
        cluster_stats = self.get_cluster_statistics(cluster_labels)
        
        # Save review metadata
        metadata_df = reviews_df[['review_id', 'product_id', 'product_name', 'review_text', 'rating']].copy()
        metadata_file = output_path / "review_metadata.csv"
        metadata_df.to_csv(metadata_file, index=False)
        print(f"\nSaved metadata to {metadata_file}")
        
        # Save configuration
        config = {
            "model": self.model,
            "embedding_dim": self.embedding_dim,
            "num_reviews": len(reviews_df),
            "num_clusters": n_clusters,
            "clustering_algorithm": "KMeans",
            "cluster_stats": cluster_stats,
            "generated_at": pd.Timestamp.now().isoformat()
        }
        config_file = output_path / "embedding_config.json"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"Saved config to {config_file}")
        
        return config
