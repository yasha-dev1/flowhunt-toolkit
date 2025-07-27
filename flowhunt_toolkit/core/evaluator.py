"""Flow evaluation logic using LLM as a judge."""

import pandas as pd
from typing import Dict, Any, List, Optional
from pathlib import Path
import json

from .client import FlowHuntClient


class FlowEvaluator:
    """Evaluates FlowHunt flows using LLM as a judge methodology."""
    
    # Default public FlowHunt flow ID for LLM judge
    DEFAULT_JUDGE_FLOW_ID = "f1b96a9a-7327-45b1-93ad-c6e28212a891"
    
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
            csv_file: Path to CSV file with 'flow_input' and 'expected_output' columns
            
        Returns:
            DataFrame with evaluation data
            
        Raises:
            ValueError: If required columns are missing
        """
        df = pd.read_csv(csv_file)
        
        required_columns = ['flow_input', 'expected_output']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(f"CSV file missing required columns: {missing_columns}")
        
        return df

    def evaluate_batch(self, flow_id: str, batch: pd.DataFrame) -> List[Dict[str, Any]]:
        """Evaluate a batch of questions.
        
        Args:
            flow_id: FlowHunt flow ID to evaluate
            batch: DataFrame batch with questions and expected answers
            
        Returns:
            List of evaluation results for the batch
        """
        batch_results = []
        
        for _, row in batch.iterrows():
            question = str(row['flow_input'])
            expected_answer = str(row['expected_output'])
            
            try:
                # Execute the target flow with the question
                flow_inputs = {"input": question, "question": question}  # Try both common input names
                actual_answer = self.client.execute_flow(flow_id, flow_inputs)

                # Use LLM judge to evaluate the answer
                judge_result = self._judge_answer(expected_answer, actual_answer)
                
                result = {
                    "question": question,
                    "expected_answer": expected_answer,
                    "actual_answer": actual_answer,
                    "judge_score": judge_result.get("total_rating", 0),
                    "judge_reasoning": judge_result.get("reasoning", "No reasoning provided"),
                    "judge_correctness": judge_result.get("correctness", "Unknown"),
                    "flow_id": flow_id,
                    "judge_flow_id": self.judge_flow_id,
                }
                
            except Exception as e:
                # Handle errors gracefully
                result = {
                    "question": question,
                    "expected_answer": expected_answer,
                    "actual_answer": f"ERROR: {str(e)}",
                    "judge_score": 0,
                    "judge_reasoning": f"Failed to execute flow: {str(e)}",
                    "judge_correctness": "Error",
                    "flow_id": flow_id,
                    "judge_flow_id": self.judge_flow_id,
                    "error": str(e)
                }
            
            batch_results.append(result)
        
        return batch_results

    def _judge_answer(self, expected_answer: str, actual_answer: str) -> Dict[str, Any]:
        """Use LLM judge to evaluate an answer.
        
        Args:
            expected_answer: Expected answer
            actual_answer: Actual answer from the flow
            
        Returns:
            Judge evaluation result
        """
        judge_inputs = {"target_response": expected_answer, "actual_response": actual_answer}
        judge_answer = self.client.execute_flow(self.judge_flow_id, judge_inputs)
        return json.loads(judge_answer)

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
        """Calculate comprehensive summary statistics from evaluation results.
        
        Args:
            results: List of evaluation results
            
        Returns:
            Summary statistics including accuracy, score distribution, and error rates
        """
        if not results:
            return {
                "total_questions": 0,
                "average_score": 0.0,
                "median_score": 0.0,
                "std_score": 0.0,
                "pass_rate": 0.0,
                "accuracy": 0.0,
                "error_rate": 0.0,
            }
        
        # Convert results to DataFrame for easier analysis
        df = pd.DataFrame(results)
        
        # Extract numeric scores, handling errors and non-numeric values
        scores = []
        errors = 0
        correct_answers = 0
        
        for result in results:
            score = result.get('judge_score', 0)
            correctness = result.get('judge_correctness', 'Unknown')
            if isinstance(score, (int, float)):
                scores.append(float(score))
            else:
                scores.append(0.0)

            if correctness.lower() == 'correct':
                correct_answers += 1
            
            if correctness.lower() == 'incorrect':
                errors += 1
        
        scores_series = pd.Series(scores)
        
        # Calculate basic statistics
        total_questions = len(results)
        average_score = scores_series.mean()
        median_score = scores_series.median()
        std_score = scores_series.std()
        
        # Calculate rates
        pass_rate = (correct_answers / total_questions) * 100 if total_questions > 0 else 0
        accuracy = pass_rate  # Same as pass rate in this context
        error_rate = (errors / total_questions) * 100 if total_questions > 0 else 0

        return {
            "total_questions": total_questions,
            "average_score": average_score,
            "median_score": median_score,
            "std_score": std_score,
            "pass_rate": pass_rate,
            "accuracy": accuracy,
            "error_rate": error_rate,
            "min_score": scores_series.min(),
            "max_score": scores_series.max(),
            "q25_score": scores_series.quantile(0.25),
            "q75_score": scores_series.quantile(0.75)
        }
