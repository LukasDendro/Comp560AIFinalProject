"""
Evaluation Script
Run test queries and evaluate system performance
"""

import json
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pipeline import RootProblemDiscoveryPipeline


def load_test_queries(queries_file: str = "tests/test_queries.json") -> dict:
    """Load test queries from JSON file"""
    with open(queries_file, 'r') as f:
        return json.load(f)


def run_evaluation(
    pipeline: RootProblemDiscoveryPipeline,
    test_queries: dict,
    output_dir: str = "output/evaluation",
    filter_query_ids: list = None
):
    """
    Run all test queries and save results
    
    Args:
        pipeline: Initialized pipeline
        test_queries: Dict with test queries
        output_dir: Directory to save evaluation results
        filter_query_ids: Optional list of query IDs to run (runs all if None)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Filter queries if specified
    queries_to_run = test_queries['test_queries']
    if filter_query_ids is not None:
        queries_to_run = [q for q in queries_to_run if q['id'] in filter_query_ids]
        print(f"Filtering to {len(filter_query_ids)} specific queries: {filter_query_ids}")
    
    print("="*80)
    print("EVALUATION: ROOT PROBLEM DISCOVERY SYSTEM")
    print("="*80)
    print(f"\nRunning {len(queries_to_run)} test queries...")
    print(f"Output directory: {output_dir}\n")
    
    results = []
    
    for i, test_case in enumerate(queries_to_run, 1):
        print(f"\n{'='*80}")
        print(f"TEST QUERY {i}/{len(queries_to_run)}")
        print(f"{'='*80}")
        print(f"Category: {test_case['category']}")
        print(f"Query: {test_case['query']}\n")
        
        try:
            # Run discovery
            result = pipeline.discover_root_problems(
                query=test_case['query'],
                k=20,
                num_reviews_for_llm=20
            )
            
            # Add test case metadata
            result['test_case'] = test_case
            
            # Display results
            formatted_output = pipeline.display_results(result)
            print(formatted_output)
            
            # Save individual result
            filename = f"query_{test_case['id']:02d}_{test_case['category'].replace(' ', '_').lower()}"
            pipeline.save_results(result, output_dir=output_dir, filename=filename)
            
            # Collect summary stats
            retrieval_summary = result['retrieval']['summary']
            root_problems = result['root_problems'].get('root_problems', [])
            
            results.append({
                'query_id': test_case['id'],
                'category': test_case['category'],
                'query': test_case['query'],
                'num_root_problems': len(root_problems),
                'avg_similarity': retrieval_summary['avg_similarity'],
                'min_similarity': retrieval_summary['min_similarity'],
                'max_similarity': retrieval_summary['max_similarity'],
                'num_products': retrieval_summary['unique_products'],
                'avg_rating': retrieval_summary['avg_rating']
            })
            
            print(f"\nQuery {i} complete\n")
            
        except Exception as e:
            print(f"\nError processing query {i}: {e}\n")
            results.append({
                'query_id': test_case['id'],
                'category': test_case['category'],
                'query': test_case['query'],
                'error': str(e)
            })
    
    # Save summary
    summary_df = pd.DataFrame(results)
    summary_file = output_path / f"evaluation_summary_{timestamp}.csv"
    summary_df.to_csv(summary_file, index=False)
    print(f"\nSaved evaluation summary to {summary_file}")
    
    # Print overall statistics
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)
    
    if 'avg_similarity' in summary_df.columns:
        print(f"\nTotal queries: {len(summary_df)}")
        print(f"Average similarity across all queries: {summary_df['avg_similarity'].mean():.3f}")
        print(f"Similarity range: {summary_df['min_similarity'].min():.3f} - {summary_df['max_similarity'].max():.3f}")
        print(f"Average root problems per query: {summary_df['num_root_problems'].mean():.1f}")
        
        print("\nBy Category:")
        category_stats = summary_df.groupby('category').agg({
            'avg_similarity': 'mean',
            'num_root_problems': 'mean'
        }).round(3)
        print(category_stats)
    
    print("\nEvaluation complete!")
    

def main():
    """Run complete evaluation"""
    import dotenv
    dotenv.load_dotenv()
    
    # Load test queries
    test_queries = load_test_queries()
    
    # Initialize pipeline
    print("Initializing pipeline...\n")
    pipeline = RootProblemDiscoveryPipeline()
    
    # Top 3 queries by similarity score:
    # Query 4: Comfort & Wearability (0.554)
    # Query 13: Call Quality & Communication (0.551)
    # Query 2: Battery & Usage Patterns (0.527)
    top_queries = [4, 13, 2]
    
    # Run evaluation (filter to top 3 queries)
    run_evaluation(pipeline, test_queries, filter_query_ids=top_queries)


if __name__ == "__main__":
    main()

