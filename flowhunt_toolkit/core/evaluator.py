"""Flow evaluation logic using LLM as a judge."""

import pandas as pd
from typing import Dict, Any, List, Optional
from pathlib import Path
import json

from .client import FlowHuntClient


class FlowEvaluator:
    """Evaluates FlowHunt flows using LLM as a judge methodology."""
    
    # Default public FlowHunt flow ID for LLM judge (placeholder)
    DEFAULT_JUDGE_FLOW_ID = "public-llm-judge-flow-id"
    
    def __init__(self, client: FlowHuntClient, judge_flow_id: Optional[str] = None):
        """Initialize flow evaluator.
        
        Args:
            client: FlowHunt API client
            judge_flow_id: Custom LLM judge flow ID. Uses default if not provided.
        """
        self.client = client
        self.judge_flow_id = judge_flow_id or self.DEFAULT_JUDGE_FLOW_ID
    
    def load_evaluation_data(self, csv_file: Path) -> pd.DataFrame:
        """Load evaluation data from CSV file.
        
        Args:
            csv_file: Path to CSV file with 'question' and 'expected_answer' columns
            
        Returns:
            DataFrame with evaluation data
            
        Raises:
            ValueError: If required columns are missing
        """
        df = pd.read_csv(csv_file)
        
        required_columns = ['question', 'expected_answer']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(f"CSV file missing required columns: {missing_columns}")
        
        return df
    
    def evaluate_flow(self, 
                     flow_id: str, 
                     evaluation_data: pd.DataFrame,
                     batch_size: int = 10) -> List[Dict[str, Any]]:
        """Evaluate a flow using LLM as a judge.
        
        Args:
            flow_id: FlowHunt flow ID to evaluate
            evaluation_data: DataFrame with questions and expected answers
            batch_size: Number of questions to process in each batch
            
        Returns:
            List of evaluation results
        """
        results = []
        
        for i in range(0, len(evaluation_data), batch_size):
            batch = evaluation_data.iloc[i:i+batch_size]
            batch_results = self._evaluate_batch(flow_id, batch)
            results.extend(batch_results)
        
        return results
    
    def _evaluate_batch(self, flow_id: str, batch: pd.DataFrame) -> List[Dict[str, Any]]:
        """Evaluate a batch of questions.
        
        Args:
            flow_id: FlowHunt flow ID to evaluate
            batch: DataFrame batch with questions and expected answers
            
        Returns:
            List of evaluation results for the batch
        """
        batch_results = []
        
        for _, row in batch.iterrows():
            question = str(row['question'])
            expected_answer = str(row['expected_answer'])
            
            try:
                # Execute the target flow with the question
                flow_inputs = {"input": question, "question": question}  # Try both common input names
                flow_output = self.client.execute_flow(flow_id, flow_inputs)
                
                # Extract the actual answer from flow output
                actual_answer = self._extract_answer_from_output(flow_output)
                
                # Use LLM judge to evaluate the answer
                judge_result = self._judge_answer(question, expected_answer, actual_answer)
                
                result = {
                    "question": question,
                    "expected_answer": expected_answer,
                    "actual_answer": actual_answer,
                    "judge_score": judge_result.get("score", 0),
                    "judge_reasoning": judge_result.get("reasoning", "No reasoning provided"),
                    "flow_id": flow_id,
                    "judge_flow_id": self.judge_flow_id,
                    "flow_output_raw": flow_output  # Include raw output for debugging
                }
                
            except Exception as e:
                # Handle errors gracefully
                result = {
                    "question": question,
                    "expected_answer": expected_answer,
                    "actual_answer": f"ERROR: {str(e)}",
                    "judge_score": 0,
                    "judge_reasoning": f"Failed to execute flow: {str(e)}",
                    "flow_id": flow_id,
                    "judge_flow_id": self.judge_flow_id,
                    "error": str(e)
                }
            
            batch_results.append(result)
        
        return batch_results
    
    def _extract_answer_from_output(self, flow_output: Dict[str, Any]) -> str:
        """Extract the answer from flow execution output.
        
        Args:
            flow_output: Raw output from flow execution
            
        Returns:
            Extracted answer as string
        """
        # Try common output field names
        possible_fields = ['output', 'result', 'answer', 'response', 'text', 'content']
        
        for field in possible_fields:
            if field in flow_output:
                value = flow_output[field]
                if isinstance(value, str):
                    return value
                elif isinstance(value, dict):
                    # If it's a dict, try to extract text from common sub-fields
                    for sub_field in ['text', 'content', 'message', 'value']:
                        if sub_field in value:
                            return str(value[sub_field])
                else:
                    return str(value)
        
        # If no standard field found, return the entire output as string
        return str(flow_output)
    
    def _judge_answer(self, question: str, expected_answer: str, actual_answer: str) -> Dict[str, Any]:
        """Use LLM judge to evaluate an answer.
        
        Args:
            question: The original question
            expected_answer: Expected answer
            actual_answer: Actual answer from the flow
            
        Returns:
            Judge evaluation result
        """
        judge_prompt = self._create_judge_prompt(question, expected_answer, actual_answer)
        
        try:
            # Execute judge flow with the prompt
            judge_inputs = {"prompt": judge_prompt, "input": judge_prompt}
            judge_result = self.client.execute_flow(self.judge_flow_id, judge_inputs)
            
            # Extract and parse the judge output
            judge_output = self._extract_answer_from_output(judge_result)
            parsed_result = self._parse_judge_output(judge_output)
            
            return parsed_result
            
        except Exception as e:
            # Fallback to simple heuristic evaluation if judge flow fails
            return self._fallback_evaluation(question, expected_answer, actual_answer, str(e))
    
    def _create_judge_prompt(self, question: str, expected_answer: str, actual_answer: str) -> str:
        """Create prompt for LLM judge.
        
        Args:
            question: The original question
            expected_answer: Expected answer
            actual_answer: Actual answer from the flow
            
        Returns:
            Formatted prompt for the judge
        """
        prompt = f"""
You are an expert evaluator tasked with judging the quality of an AI system's answer.

Question: {question}

Expected Answer: {expected_answer}

Actual Answer: {actual_answer}

Please evaluate the actual answer against the expected answer and provide:
1. A score from 0-10 (where 10 is perfect match)
2. Detailed reasoning for your score

Consider factors like:
- Factual accuracy
- Completeness
- Relevance
- Clarity

Format your response as JSON:
{{
    "score": <number>,
    "reasoning": "<detailed explanation>"
}}
"""
        return prompt.strip()
    
    def _parse_judge_output(self, judge_output: str) -> Dict[str, Any]:
        """Parse the judge output to extract score and reasoning.
        
        Args:
            judge_output: Raw output from judge flow
            
        Returns:
            Parsed score and reasoning
        """
        try:
            # Try to parse as JSON first
            import json
            parsed = json.loads(judge_output)
            if isinstance(parsed, dict) and 'score' in parsed:
                return {
                    'score': float(parsed.get('score', 0)),
                    'reasoning': parsed.get('reasoning', 'No reasoning provided')
                }
        except (json.JSONDecodeError, ValueError):
            pass
        
        # Fallback: try to extract score and reasoning from text
        score = 0
        reasoning = judge_output
        
        # Look for score patterns
        import re
        score_patterns = [
            r'score[:\s]+([0-9\.]+)',
            r'rating[:\s]+([0-9\.]+)',
            r'([0-9\.]+)\s*(?:out of|/|\s)\s*10',
            r'([0-9\.]+)\s*points?'
        ]
        
        for pattern in score_patterns:
            match = re.search(pattern, judge_output.lower())
            if match:
                try:
                    score = float(match.group(1))
                    break
                except ValueError:
                    continue
        
        return {
            'score': score,
            'reasoning': reasoning
        }
    
    def _fallback_evaluation(self, question: str, expected_answer: str, actual_answer: str, error: str) -> Dict[str, Any]:
        """Provide fallback evaluation when judge flow fails.
        
        Args:
            question: The original question
            expected_answer: Expected answer
            actual_answer: Actual answer from the flow
            error: Error message from judge flow
            
        Returns:
            Fallback evaluation result
        """
        # Simple heuristic evaluation
        if "ERROR:" in actual_answer:
            score = 0
            reasoning = f"Flow execution failed: {actual_answer}"
        else:
            # Basic similarity check
            expected_lower = expected_answer.lower().strip()
            actual_lower = actual_answer.lower().strip()
            
            if expected_lower == actual_lower:
                score = 10
                reasoning = "Exact match with expected answer"
            elif expected_lower in actual_lower or actual_lower in expected_lower:
                score = 7
                reasoning = "Partial match with expected answer"
            else:
                # Check for common words
                expected_words = set(expected_lower.split())
                actual_words = set(actual_lower.split())
                common_words = expected_words.intersection(actual_words)
                
                if len(common_words) > 0:
                    similarity_ratio = len(common_words) / max(len(expected_words), len(actual_words))
                    score = similarity_ratio * 5  # Scale to 0-5
                    reasoning = f"Some common words found. Judge flow error: {error}"
                else:
                    score = 1
                    reasoning = f"No obvious similarity. Judge flow error: {error}"
        
        return {
            'score': score,
            'reasoning': reasoning
        }
    
    def save_results(self, results: List[Dict[str, Any]], output_file: Path):
        """Save evaluation results to file.
        
        Args:
            results: List of evaluation results
            output_file: Path to output file
        """
        if output_file.suffix.lower() == '.json':
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
        elif output_file.suffix.lower() == '.csv':
            df = pd.DataFrame(results)
            df.to_csv(output_file, index=False)
        else:
            raise ValueError(f"Unsupported output format: {output_file.suffix}")
    
    def calculate_summary_stats(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate summary statistics from evaluation results.
        
        Args:
            results: List of evaluation results
            
        Returns:
            Summary statistics
        """
        # TODO: Implement summary statistics calculation
        return {
            "total_questions": len(results),
            "average_score": "TODO: Calculate from judge scores",
            "score_distribution": "TODO: Calculate score distribution",
            "pass_rate": "TODO: Calculate based on threshold"
        }
