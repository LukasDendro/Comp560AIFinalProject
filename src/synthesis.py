"""Root Problem Synthesis Engine"""

import pandas as pd
from typing import Dict
from openai import OpenAI
import json


class RootProblemSynthesizer:
    """Generate root problem suggestions from retrieved reviews using LLM"""
    
    def __init__(self, api_key: str = None, model: str = "gpt-5-mini-2025-08-07"):
        self.client = OpenAI(api_key=api_key)
        self.model = model
    
    def build_few_shot_prompt(
        self,
        query: str,
        retrieved_reviews: pd.DataFrame,
        num_reviews: int = 20
    ) -> str:
        """Build few-shot prompt for root problem discovery"""
        # Select top reviews
        reviews_subset = retrieved_reviews.head(num_reviews)
        
        # Format reviews
        formatted_reviews = []
        for idx, row in reviews_subset.iterrows():
            formatted_reviews.append(
                f"Review {idx + 1} [{row['product_name']}, Rating: {row['rating']}/5, Similarity: {row['similarity_score']:.3f}]:\n"
                f"\"{row['review_text']}\"\n"
            )
        
        reviews_text = "\n".join(formatted_reviews)
        
        # Build prompt with few-shot examples
        prompt = f"""You are an expert design thinking researcher specializing in uncovering root problems from user feedback. Your task is to analyze product reviews and identify CORE ROOT PROBLEMS - the deep, underlying needs that users may not explicitly state.

IMPORTANT DISTINCTIONS:
- Surface Problem: The immediate complaint (e.g., "earbuds slip out")
- Intermediate Problem: The direct consequence (e.g., "uncomfortable during movement")  
- Root Problem: The fundamental unmet need (e.g., "users need confidence their device will stay secure without conscious attention during varied physical activities")

FEW-SHOT EXAMPLES:

Example 1:
Surface complaints: "Battery dies too quickly", "only 2 hours of use", "constantly charging"
→ Root Problem: "Users need their device to match their lifestyle rhythm without forcing behavioral adaptations around recharging"

Example 2:
Surface complaints: "Bluetooth keeps disconnecting", "pairing is annoying", "have to reconnect constantly"
→ Root Problem: "Users expect seamless, invisible connectivity that doesn't interrupt their experience or require technical troubleshooting"

Example 3:
Surface complaints: "Uncomfortable after 30 minutes", "ear pain", "too much pressure"
→ Root Problem: "Users want to forget they're wearing a device - seeking zero cognitive load and physical awareness during extended use"

---

DESIGNER'S QUERY:
"{query}"

---

RETRIEVED USER REVIEWS (semantically similar to query):
{reviews_text}

---

YOUR TASK:
Analyze these reviews and identify 2 ROOT PROBLEMS. For each root problem:

1. Go beyond surface complaints to identify the fundamental unmet need
2. Consider patterns across multiple reviews
3. Think about what users REALLY care about at a deeper level
4. Frame problems in terms of user goals, not product features
5. Provide specific supporting evidence from the reviews

OUTPUT FORMAT (JSON):
{{
  "root_problems": [
    {{
      "title": "Brief title of root problem",
      "description": "2-3 sentence explanation of the core unmet need",
      "evidence": [
        "Quote or paraphrase from Review X that supports this",
        "Quote or paraphrase from Review Y that supports this",
        "Quote or paraphrase from Review Z that supports this"
      ],
      "why_root": "1-2 sentences explaining why this is a ROOT problem, not just a surface complaint"
    }}
  ]
}}

Generate 2 root problems now:"""
        
        return prompt
    
    def generate_root_problems(
        self,
        query: str,
        retrieved_reviews: pd.DataFrame,
        num_reviews: int = 20
    ) -> Dict:
        """Generate root problem suggestions using LLM"""
        print(f"Generating root problem suggestions with {self.model}...")
        
        # Build prompt
        prompt = self.build_few_shot_prompt(query, retrieved_reviews, num_reviews)
        
        # Call LLM - different endpoints for reasoning models vs chat models
        try:
            # Try responses API first (for o1-preview, o1-mini, gpt-5-pro, etc.)
            # These models don't support temperature or response_format
            try:
                # Prepend system instructions to user prompt for reasoning models
                full_prompt = "You are an expert design thinking researcher who excels at uncovering deep, root problems from user feedback.\n\n" + prompt
                
                # Responses API uses 'input' parameter instead of 'messages'
                response = self.client.responses.create(
                    model=self.model,
                    input=full_prompt
                )
                # Extract content from responses API
                # The output is a list of content blocks (reasoning + text response)
                content = ""
                if hasattr(response, 'output') and isinstance(response.output, list):
                    # Iterate through output blocks to find the text response
                    for block in response.output:
                        # Skip reasoning blocks, look for text/message blocks
                        if hasattr(block, 'type'):
                            if block.type == 'message' or block.type == 'text':
                                # Found the actual response
                                if hasattr(block, 'content') and block.content:
                                    if isinstance(block.content, list) and len(block.content) > 0:
                                        # Content is a list of text items
                                        first_content = block.content[0]
                                        if hasattr(first_content, 'text'):
                                            content = first_content.text
                                        else:
                                            content = str(first_content)
                                    else:
                                        content = str(block.content)
                                    break
                                elif hasattr(block, 'text'):
                                    content = block.text
                                    break
                    
                    # If no message block found, check if there's encrypted_content to decrypt
                    if not content:
                        for block in response.output:
                            if hasattr(block, 'encrypted_content') and block.encrypted_content:
                                print("Response contains encrypted content - may need decryption")
                                
                elif hasattr(response, 'output'):
                    # Try other output formats
                    if hasattr(response.output, 'content'):
                        content = response.output.content
                    else:
                        content = str(response.output)
                else:
                    # Fallback to choices format (similar to chat completions)
                    content = response.choices[0].message.content
                
            except Exception as responses_error:
                # Fallback to chat completions API (for GPT-4o, GPT-4o-mini, etc.)
                print(f"   Trying chat completions API (responses API failed: {responses_error})")
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert design thinking researcher who excels at uncovering deep, root problems from user feedback."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]
                    # Removed temperature and response_format - focusing on quality over format
                )
                # Extract content from chat completions API
                content = response.choices[0].message.content
            
            # Try to parse as JSON
            try:
                result = json.loads(content)
            except json.JSONDecodeError:
                # If not valid JSON, try to extract JSON from markdown code blocks
                import re
                json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', content, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group(1))
                else:
                    # Last resort: try to find any JSON object in the text
                    json_match = re.search(r'\{.*"root_problems".*\}', content, re.DOTALL)
                    if json_match:
                        result = json.loads(json_match.group(0))
                    else:
                        raise ValueError(f"Could not parse JSON from response: {content[:200]}...")
            
            print(f"Generated {len(result.get('root_problems', []))} root problem suggestions")
            
            return result
            
        except Exception as e:
            print(f"Error generating root problems: {e}")
            return {
                "root_problems": [],
                "error": str(e)
            }
    
    def format_output(self, root_problems_result: Dict, query: str) -> str:
        """Format root problems as readable text output"""
        output = []
        output.append("=" * 80)
        output.append("ROOT PROBLEM DISCOVERY RESULTS")
        output.append("=" * 80)
        output.append(f"\nQuery: {query}\n")
        
        if "error" in root_problems_result:
            output.append(f"Error: {root_problems_result['error']}")
            return "\n".join(output)
        
        root_problems = root_problems_result.get("root_problems", [])
        
        for i, problem in enumerate(root_problems, 1):
            output.append(f"\n{'─' * 80}")
            output.append(f"ROOT PROBLEM #{i}: {problem.get('title', 'Untitled')}")
            output.append(f"{'─' * 80}")
            output.append(f"\n{problem.get('description', 'No description')}")
            
            output.append("\nSUPPORTING EVIDENCE:")
            for j, evidence in enumerate(problem.get('evidence', []), 1):
                output.append(f"   {j}. {evidence}")
            
            output.append(f"\nWHY THIS IS A ROOT PROBLEM:")
            output.append(f"   {problem.get('why_root', 'No explanation')}")
        
        output.append("\n" + "=" * 80)
        
        return "\n".join(output)
