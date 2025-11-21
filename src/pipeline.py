"""End-to-End Root Problem Discovery Pipeline"""

import os
from pathlib import Path
from typing import Dict, Optional
import json
import pandas as pd
from datetime import datetime

from retrieval import ReviewRetriever
from synthesis import RootProblemSynthesizer


class RootProblemDiscoveryPipeline:
    """Complete pipeline for root problem discovery from user query"""
    
    def __init__(
        self,
        embeddings_dir: str = "data/embeddings",
        api_key: Optional[str] = None
    ):
        """Initialize the complete pipeline"""
        embeddings_path = Path(embeddings_dir)
        
        print("Initializing Root Problem Discovery Pipeline...")
        
        # Initialize retriever
        self.retriever = ReviewRetriever(
            embeddings_path=embeddings_path / "review_embeddings.npy",
            metadata_path=embeddings_path / "review_metadata.csv",
            config_path=embeddings_path / "embedding_config.json",
            api_key=api_key
        )
        
        # Initialize synthesizer
        self.synthesizer = RootProblemSynthesizer(api_key=api_key)
        
        print("Pipeline ready!\n")
    
    def discover_root_problems(
        self,
        query: str,
        k: int = 20,
        num_reviews_for_llm: int = 20
    ) -> Dict:
        """Run complete pipeline: retrieve similar reviews and synthesize root problems"""
        print("\n" + "="*80)
        print("ROOT PROBLEM DISCOVERY PIPELINE")
        print("="*80)
        print(f"\nQuery: {query}")
        print(f"Settings: K={k}, LLM Reviews={num_reviews_for_llm}")
        
        # Stage 1: Retrieve similar reviews
        print("\n" + "─"*80)
        print("STAGE 1: SEMANTIC RETRIEVAL")
        print("─"*80)
        
        similar_reviews, similarity_scores, cluster_metadata = self.retriever.retrieve_similar_reviews(
            query=query,
            k=k,
            M=5  # Search top 5 clusters
        )
        
        retrieval_summary = self.retriever.get_retrieval_summary(similar_reviews)
        
        # Stage 2: Synthesize root problems
        print("\n" + "─"*80)
        print("STAGE 2: ROOT PROBLEM SYNTHESIS")
        print("─"*80)
        
        root_problems_result = self.synthesizer.generate_root_problems(
            query=query,
            retrieved_reviews=similar_reviews,
            num_reviews=num_reviews_for_llm
        )
        
        # Compile complete result
        result = {
            "query": query,
            "timestamp": datetime.now().isoformat(),
            "settings": {
                "k": k,
                "num_reviews_for_llm": num_reviews_for_llm
            },
            "retrieval": {
                "summary": retrieval_summary,
                "top_reviews": similar_reviews.to_dict('records'),
                "cluster_info": cluster_metadata  # Will be None if clustering not enabled
            },
            "root_problems": root_problems_result
        }
        
        return result
    
    def display_results(self, result: Dict) -> str:
        """Display results in human-readable format"""
        output = []
        output.append("\n" + "="*80)
        output.append("ROOT PROBLEM DISCOVERY RESULTS")
        output.append("="*80)
        output.append(f"\nQuery: {result['query']}")
        output.append(f"Timestamp: {result['timestamp']}")
        
        # Retrieval summary
        output.append("\n" + "─"*80)
        output.append("RETRIEVAL SUMMARY")
        output.append("─"*80)
        
        summary = result['retrieval']['summary']
        output.append(f"\nRetrieved: {summary['num_reviews']} reviews from {summary['unique_products']} products")
        output.append(f"Similarity: {summary['min_similarity']:.3f} - {summary['max_similarity']:.3f} (avg: {summary['avg_similarity']:.3f})")
        output.append(f"Average rating: {summary['avg_rating']:.1f}/5")
        
        output.append("\nTop products represented:")
        for product, count in list(summary['products'].items())[:5]:
            output.append(f"   • {product}: {count} reviews")
        
        # Cluster analysis (if available)
        cluster_info = result['retrieval'].get('cluster_info')
        if cluster_info is not None:
            output.append("\n" + "─"*80)
            output.append("CLUSTER ANALYSIS")
            output.append("─"*80)
            
            num_searched = cluster_info['num_clusters_searched']
            total_clusters = cluster_info['total_clusters']
            pct_searched = (num_searched / total_clusters * 100) if total_clusters > 0 else 0
            
            output.append(f"\nSearched {num_searched} of {total_clusters} clusters ({pct_searched:.0f}% of dataset)")
            output.append("\nTop Clusters Selected:")
            
            for cluster in cluster_info['searched_clusters']:
                cluster_id = cluster['cluster_id']
                similarity = cluster['similarity']
                contributed = cluster['reviews_retrieved']
                output.append(f"   • Cluster {cluster_id}: similarity {similarity:.3f}, contributed {contributed} reviews")
            
            total_reviews = cluster_info['total_reviews']
            searched_reviews = cluster_info['total_reviews_searched']
            speedup = total_reviews / searched_reviews if searched_reviews > 0 else 1
            
            output.append(f"\nSearch efficiency: ~{searched_reviews} reviews searched instead of {total_reviews} ({speedup:.1f}x speedup)")
        
        # Root problems
        output.append("\n" + "─"*80)
        output.append("DISCOVERED ROOT PROBLEMS")
        output.append("─"*80)
        
        root_problems = result['root_problems'].get('root_problems', [])
        
        if not root_problems:
            output.append("\nNo root problems generated")
            if 'error' in result['root_problems']:
                output.append(f"Error: {result['root_problems']['error']}")
        else:
            for i, problem in enumerate(root_problems, 1):
                output.append(f"\n{'┌' + '─'*78 + '┐'}")
                output.append(f"│ ROOT PROBLEM #{i}: {problem.get('title', 'Untitled'):<62} │")
                output.append(f"{'└' + '─'*78 + '┘'}")
                
                output.append(f"\nDescription:")
                output.append(f"   {problem.get('description', 'No description')}")
                
                output.append(f"\nSupporting Evidence:")
                for j, evidence in enumerate(problem.get('evidence', []), 1):
                    # Wrap long evidence text
                    evidence_lines = [evidence[i:i+70] for i in range(0, len(evidence), 70)]
                    output.append(f"   {j}. {evidence_lines[0]}")
                    for line in evidence_lines[1:]:
                        output.append(f"      {line}")
                
                output.append(f"\nWhy This is a Root Problem:")
                output.append(f"   {problem.get('why_root', 'No explanation')}")
                
                if i < len(root_problems):
                    output.append("")
        
        output.append("\n" + "="*80)
        
        return "\n".join(output)
    
    def save_results(
        self,
        result: Dict,
        output_dir: str = "output",
        filename: Optional[str] = None
    ):
        """Save results to files (JSON + formatted text)"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Generate filename if not provided
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"root_problems_{timestamp}"
        
        # Save JSON
        json_file = output_path / f"{filename}.json"
        with open(json_file, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"Saved JSON to {json_file}")
        
        # Save formatted text
        text_file = output_path / f"{filename}.txt"
        formatted_output = self.display_results(result)
        with open(text_file, 'w') as f:
            f.write(formatted_output)
        print(f"Saved formatted output to {text_file}")
