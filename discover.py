#!/usr/bin/env python3
"""
Command-line interface for Root Problem Discovery System
"""

import argparse
import sys
from pathlib import Path
import dotenv

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from pipeline import RootProblemDiscoveryPipeline


def main():
    parser = argparse.ArgumentParser(
        description="Discover root problems from product reviews using AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive query
  python discover.py

  # Direct query
  python discover.py --query "What problems do users face with battery life?"

  # Adjust number of reviews
  python discover.py --query "connectivity issues" --k 30

  # Save with custom filename
  python discover.py --query "comfort problems" --output my_results
        """
    )
    
    parser.add_argument(
        '--query', '-q',
        type=str,
        help='Query about the problem space (if not provided, will prompt interactively)'
    )
    
    parser.add_argument(
        '--k',
        type=int,
        default=20,
        help='Number of similar reviews to retrieve (default: 20)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Custom output filename (without extension)'
    )
    
    parser.add_argument(
        '--embeddings-dir',
        type=str,
        default='data/embeddings',
        help='Directory containing embeddings (default: data/embeddings)'
    )
    
    args = parser.parse_args()
    
    # Load environment variables
    dotenv.load_dotenv()
    
    # Get query
    if args.query:
        query = args.query
    else:
        print("="*80)
        print("ROOT PROBLEM DISCOVERY SYSTEM")
        print("="*80)
        print("\nEnter your query about the problem space you're exploring.")
        print("Example: 'What problems do users face with battery life?'\n")
        query = input("Query: ").strip()
        
        if not query:
            print("Error: Query cannot be empty")
            sys.exit(1)
    
    try:
        # Initialize pipeline
        print("\nInitializing pipeline...")
        pipeline = RootProblemDiscoveryPipeline(embeddings_dir=args.embeddings_dir)
        
        # Run discovery
        result = pipeline.discover_root_problems(
            query=query,
            k=args.k,
            num_reviews_for_llm=args.k
        )
        
        # Display results
        formatted_output = pipeline.display_results(result)
        print(formatted_output)
        
        # Save results
        pipeline.save_results(result, filename=args.output)
        
        print("\nDiscovery complete!")
        
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nMake sure you've generated embeddings first:")
        print("  python src/embeddings.py")
        sys.exit(1)
        
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

