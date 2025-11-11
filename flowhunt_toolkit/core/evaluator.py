"""Flow evaluation logic using LLM as a judge."""

import pandas as pd
from typing import Dict, Any, List, Optional
from pathlib import Path
import json
import time
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.panel import Panel
from rich.table import Table
from rich import box

from .client import FlowHuntClient
from .report_generator import EvaluationReportGenerator


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
        self.report_generator = EvaluationReportGenerator()
    
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

                # judge_result should have score, correctness and reasoning at least
                judge_score = judge_result.pop('total_rating', 0)
                judge_correctness = judge_result.pop('correctness', 'Unknown')
                judge_reasoning = judge_result.pop('reasoning', 'No reasoning provided')

                
                result = {
                    "question": question,
                    "expected_answer": expected_answer,
                    "actual_answer": actual_answer,
                    "flow_id": flow_id,
                    "judge_flow_id": self.judge_flow_id,
                    "judge_score": judge_score,
                    "judge_correctness": judge_correctness,
                    "judge_reasoning": judge_reasoning,
                    **judge_result,  # Include judge results
                }
                
            except Exception as e:
                # Handle errors gracefully
                result = {
                    "question": question,
                    "expected_answer": expected_answer,
                    "actual_answer": f"ERROR: {str(e)}",
                    "judge_score": 0,
                    "judge_correctness": "Error",
                    "judge_reasoning": f"Error during evaluation: {str(e)}",
                    "flow_id": flow_id,
                    "judge_flow_id": self.judge_flow_id,
                    "error": str(e)
                }
            
            batch_results.append(result)
        
        return batch_results

    def evaluate_parallel(self, flow_id: str, evaluation_data: pd.DataFrame,
                         max_parallel: int = 4, check_interval: int = 2) -> List[Dict[str, Any]]:
        """Evaluate questions in parallel with controlled parallelism.

        Args:
            flow_id: FlowHunt flow ID to evaluate
            evaluation_data: DataFrame with 'flow_input' and 'expected_output' columns
            max_parallel: Maximum number of parallel evaluations
            check_interval: Seconds between result checks

        Returns:
            List of all evaluation results
        """
        console = Console()
        workspace_id = self.client.get_workspace_id()

        # Prepare evaluation tasks
        eval_tasks = []
        for _, row in evaluation_data.iterrows():
            eval_tasks.append({
                'question': str(row['flow_input']),
                'expected_answer': str(row['expected_output'])
            })

        with self.client.flowhunt.ApiClient(self.client.configuration) as api_client:
            api_instance = self.client.flowhunt.FlowsApi(api_client)

            # Track tasks
            pending_flow_tasks = {}  # flow_id -> {process_id, question, expected_answer}
            pending_judge_tasks = {}  # judge_flow_id -> {process_id, question, expected_answer, actual_answer}
            completed_results = []
            failed_results = []
            tasks_queue = eval_tasks.copy()

            # Create header
            header_panel = Panel(
                f"[bold cyan]🎯 PARALLEL FLOW EVALUATION[/bold cyan]\n"
                f"[dim]Evaluating {len(eval_tasks)} questions with max {max_parallel} parallel workers[/dim]\n"
                f"[dim]Flow ID: {flow_id[:8]}...{flow_id[-8:] if len(flow_id) > 16 else flow_id}[/dim]\n"
                f"[dim]Judge Flow ID: {self.judge_flow_id[:8]}...{self.judge_flow_id[-8:] if len(self.judge_flow_id) > 16 else self.judge_flow_id}[/dim]",
                box=box.ROUNDED,
                border_style="cyan"
            )
            console.print(header_panel)

            # Create progress display
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(complete_style="green", finished_style="bold green"),
                TaskProgressColumn(),
                TextColumn("•"),
                TimeElapsedColumn(),
                TextColumn("•"),
                TimeRemainingColumn(),
                console=console,
                transient=False
            ) as progress:

                # Create three progress bars
                schedule_task = progress.add_task(
                    "[blue]Scheduling evaluations...",
                    total=len(eval_tasks)
                )

                flow_task = progress.add_task(
                    "[yellow]Running flows...",
                    total=len(eval_tasks)
                )

                judge_task = progress.add_task(
                    "[magenta]Judging results...",
                    total=len(eval_tasks)
                )

                # Statistics tracking
                start_time = time.time()
                total_scheduled = 0
                total_flow_completed = 0
                total_judged = 0

                while tasks_queue or pending_flow_tasks or pending_judge_tasks:
                    # Schedule new flow evaluations up to max_parallel limit
                    current_running = len(pending_flow_tasks) + len(pending_judge_tasks)
                    while current_running < max_parallel and tasks_queue:
                        task = tasks_queue.pop(0)

                        try:
                            # Execute target flow
                            flow_inputs = {"input": task['question'], "question": task['question']}
                            flow_invoke_request = self.client.flowhunt.FlowInvokeRequest(
                                variables={},
                                human_input=task['question']
                            )

                            response = api_instance.invoke_flow_singleton(
                                flow_id=flow_id,
                                workspace_id=workspace_id,
                                flow_invoke_request=flow_invoke_request
                            )

                            pending_flow_tasks[response.id] = {
                                'question': task['question'],
                                'expected_answer': task['expected_answer']
                            }
                            total_scheduled += 1
                            progress.advance(schedule_task, 1)
                            current_running += 1

                        except Exception as e:
                            console.print(f"[red]✗[/red] Failed to schedule flow: {str(e)[:60]}...")
                            failed_results.append({
                                "question": task['question'],
                                "expected_answer": task['expected_answer'],
                                "actual_answer": f"ERROR: {str(e)}",
                                "judge_score": 0,
                                "judge_correctness": "Error",
                                "judge_reasoning": f"Error scheduling flow: {str(e)}",
                                "flow_id": flow_id,
                                "judge_flow_id": self.judge_flow_id,
                                "error": str(e)
                            })
                            total_scheduled += 1
                            progress.advance(schedule_task, 1)
                            progress.advance(flow_task, 1)
                            progress.advance(judge_task, 1)

                    # Update progress descriptions
                    progress.update(
                        schedule_task,
                        description=f"[blue]Scheduling[/blue] │ {total_scheduled}/{len(eval_tasks)} scheduled"
                    )

                    progress.update(
                        flow_task,
                        description=f"[yellow]Running flows[/yellow] │ [green]{total_flow_completed} ✓[/green] [yellow]{len(pending_flow_tasks)} ⚡[/yellow]"
                    )

                    progress.update(
                        judge_task,
                        description=f"[magenta]Judging[/magenta] │ [green]{total_judged} ✓[/green] [red]{len(failed_results)} ✗[/red] [yellow]{len(pending_judge_tasks)} ⚡[/yellow]"
                    )

                    # Check for completed flow tasks
                    if pending_flow_tasks or pending_judge_tasks:
                        time.sleep(check_interval)

                        # Check completed flows
                        flow_process_ids = list(pending_flow_tasks.keys())
                        for process_id in flow_process_ids:
                            task_data = pending_flow_tasks[process_id]

                            try:
                                is_ready, content = self.client.get_flow_results(flow_id, process_id)

                                if is_ready:
                                    del pending_flow_tasks[process_id]
                                    total_flow_completed += 1
                                    progress.advance(flow_task, 1)

                                    if content and content != "NOCONTENT":
                                        # Schedule judge evaluation
                                        try:
                                            judge_inputs = {
                                                "target_response": task_data['expected_answer'],
                                                "actual_response": content
                                            }
                                            judge_invoke_request = self.client.flowhunt.FlowInvokeRequest(
                                                variables=judge_inputs,
                                                human_input=""
                                            )

                                            judge_response = api_instance.invoke_flow_singleton(
                                                flow_id=self.judge_flow_id,
                                                workspace_id=workspace_id,
                                                flow_invoke_request=judge_invoke_request
                                            )

                                            pending_judge_tasks[judge_response.id] = {
                                                'question': task_data['question'],
                                                'expected_answer': task_data['expected_answer'],
                                                'actual_answer': content
                                            }

                                        except Exception as e:
                                            console.print(f"[red]✗[/red] Failed to schedule judge: {str(e)[:60]}...")
                                            failed_results.append({
                                                "question": task_data['question'],
                                                "expected_answer": task_data['expected_answer'],
                                                "actual_answer": content,
                                                "judge_score": 0,
                                                "judge_correctness": "Error",
                                                "judge_reasoning": f"Error scheduling judge: {str(e)}",
                                                "flow_id": flow_id,
                                                "judge_flow_id": self.judge_flow_id,
                                                "error": str(e)
                                            })
                                            progress.advance(judge_task, 1)
                                    else:
                                        failed_results.append({
                                            "question": task_data['question'],
                                            "expected_answer": task_data['expected_answer'],
                                            "actual_answer": "ERROR: No content returned",
                                            "judge_score": 0,
                                            "judge_correctness": "Error",
                                            "judge_reasoning": "No content returned from flow",
                                            "flow_id": flow_id,
                                            "judge_flow_id": self.judge_flow_id,
                                            "error": "No content"
                                        })
                                        progress.advance(judge_task, 1)

                            except Exception as e:
                                console.print(f"[red]✗[/red] Flow processing error: {str(e)[:60]}...")
                                del pending_flow_tasks[process_id]
                                total_flow_completed += 1
                                progress.advance(flow_task, 1)
                                failed_results.append({
                                    "question": task_data['question'],
                                    "expected_answer": task_data['expected_answer'],
                                    "actual_answer": f"ERROR: {str(e)}",
                                    "judge_score": 0,
                                    "judge_correctness": "Error",
                                    "judge_reasoning": f"Error during flow execution: {str(e)}",
                                    "flow_id": flow_id,
                                    "judge_flow_id": self.judge_flow_id,
                                    "error": str(e)
                                })
                                progress.advance(judge_task, 1)

                        # Check completed judge tasks
                        judge_process_ids = list(pending_judge_tasks.keys())
                        for process_id in judge_process_ids:
                            task_data = pending_judge_tasks[process_id]

                            try:
                                is_ready, content = self.client.get_flow_results(self.judge_flow_id, process_id)

                                if is_ready:
                                    del pending_judge_tasks[process_id]
                                    total_judged += 1
                                    progress.advance(judge_task, 1)

                                    if content and content != "NOCONTENT":
                                        try:
                                            judge_result = json.loads(content)
                                            judge_score = judge_result.pop('total_rating', 0)
                                            judge_correctness = judge_result.pop('correctness', 'Unknown')
                                            judge_reasoning = judge_result.pop('reasoning', 'No reasoning provided')

                                            completed_results.append({
                                                "question": task_data['question'],
                                                "expected_answer": task_data['expected_answer'],
                                                "actual_answer": task_data['actual_answer'],
                                                "flow_id": flow_id,
                                                "judge_flow_id": self.judge_flow_id,
                                                "judge_score": judge_score,
                                                "judge_correctness": judge_correctness,
                                                "judge_reasoning": judge_reasoning,
                                                **judge_result
                                            })
                                        except Exception as e:
                                            console.print(f"[red]✗[/red] Failed to parse judge result: {str(e)[:60]}...")
                                            failed_results.append({
                                                "question": task_data['question'],
                                                "expected_answer": task_data['expected_answer'],
                                                "actual_answer": task_data['actual_answer'],
                                                "judge_score": 0,
                                                "judge_correctness": "Error",
                                                "judge_reasoning": f"Error parsing judge result: {str(e)}",
                                                "flow_id": flow_id,
                                                "judge_flow_id": self.judge_flow_id,
                                                "error": str(e)
                                            })
                                    else:
                                        failed_results.append({
                                            "question": task_data['question'],
                                            "expected_answer": task_data['expected_answer'],
                                            "actual_answer": task_data['actual_answer'],
                                            "judge_score": 0,
                                            "judge_correctness": "Error",
                                            "judge_reasoning": "No content returned from judge",
                                            "flow_id": flow_id,
                                            "judge_flow_id": self.judge_flow_id,
                                            "error": "No judge content"
                                        })

                            except Exception as e:
                                console.print(f"[red]✗[/red] Judge processing error: {str(e)[:60]}...")
                                del pending_judge_tasks[process_id]
                                total_judged += 1
                                progress.advance(judge_task, 1)
                                failed_results.append({
                                    "question": task_data['question'],
                                    "expected_answer": task_data['expected_answer'],
                                    "actual_answer": task_data['actual_answer'],
                                    "judge_score": 0,
                                    "judge_correctness": "Error",
                                    "judge_reasoning": f"Error during judge evaluation: {str(e)}",
                                    "flow_id": flow_id,
                                    "judge_flow_id": self.judge_flow_id,
                                    "error": str(e)
                                })

            # Combine all results
            all_results = completed_results + failed_results

            # Final summary
            total_time = time.time() - start_time
            success_rate = (len(completed_results) / len(eval_tasks) * 100) if eval_tasks else 0

            # Create summary table
            summary_table = Table(
                title="[bold]📊 Evaluation Summary[/bold]",
                show_header=True,
                header_style="bold magenta",
                box=box.ROUNDED,
                border_style="cyan"
            )
            summary_table.add_column("Metric", style="cyan", justify="left")
            summary_table.add_column("Count", style="white", justify="right")
            summary_table.add_column("Percentage", style="green", justify="right")

            summary_table.add_row("✅ Completed", str(len(completed_results)), f"{success_rate:.1f}%")
            summary_table.add_row("❌ Failed", str(len(failed_results)), f"{(len(failed_results)/len(eval_tasks)*100):.1f}%" if eval_tasks else "0%")
            summary_table.add_row("📊 Total", str(len(eval_tasks)), "100%")

            console.print(summary_table)

            # Performance metrics
            throughput = len(eval_tasks) / total_time if total_time > 0 else 0
            perf_panel = Panel(
                f"[bold cyan]⚡ Performance Metrics[/bold cyan]\n"
                f"[green]⏱️ Total time:[/green] {total_time:.1f} seconds\n"
                f"[green]🚀 Throughput:[/green] {throughput:.2f} evaluations/second\n"
                f"[green]🔥 Max parallel:[/green] {max_parallel} workers\n"
                f"[green]🎯 Success rate:[/green] {success_rate:.1f}%",
                box=box.ROUNDED,
                border_style="green"
            )
            console.print(perf_panel)

        return all_results

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
        
        Analyzes all columns in the results, providing:
        - Basic statistics for numerical columns (mean, median, std, quartiles, etc.)
        - Value counts and distributions for categorical columns
        - Legacy compatibility with existing statistics
        
        Args:
            results: List of evaluation results
            
        Returns:
            Dictionary with overall_stats (legacy compatibility) and column_stats (detailed analysis)
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
                "column_stats": {},
                "overall_stats": {}
            }
        
        # Convert results to DataFrame for comprehensive analysis
        df = pd.DataFrame(results)
        
        # Fixed columns that should not be analyzed as dynamic columns
        fixed_columns = {'question', 'expected_answer', 'actual_answer', 'flow_id', 'judge_flow_id'}
        
        # Calculate legacy statistics for backward compatibility
        legacy_stats = self._calculate_legacy_stats(df)
        
        # Analyze all columns dynamically
        column_stats = {}
        for column in df.columns:
            if column not in fixed_columns:
                column_stats[column] = self._analyze_column(df[column], column)
        
        # Combine results with legacy compatibility
        result = {
            **legacy_stats,  # Legacy stats at top level for backward compatibility
            "column_stats": column_stats,
            "overall_stats": legacy_stats
        }
        
        return result
    
    def _calculate_legacy_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate legacy statistics for backward compatibility.
        
        Args:
            df: DataFrame with evaluation results
            
        Returns:
            Dictionary with legacy statistics
        """
        total_questions = len(df)
        
        # Handle judge_score with proper error handling
        if 'judge_score' in df.columns:
            # Convert to numeric, coercing errors to NaN
            scores = pd.to_numeric(df['judge_score'], errors='coerce').fillna(0.0)
            average_score = scores.mean()
            median_score = scores.median()
            std_score = scores.std()
            min_score = scores.min()
            max_score = scores.max()
            q25_score = scores.quantile(0.25)
            q75_score = scores.quantile(0.75)
        else:
            average_score = median_score = std_score = 0.0
            min_score = max_score = q25_score = q75_score = 0.0
        
        # Handle judge_correctness
        if 'judge_correctness' in df.columns:
            correctness_counts = df['judge_correctness'].str.lower().value_counts()
            correct_answers = correctness_counts.get('correct', 0)
            incorrect_answers = correctness_counts.get('incorrect', 0)
        else:
            correct_answers = incorrect_answers = 0
        
        # Calculate rates
        pass_rate = (correct_answers / total_questions) * 100 if total_questions > 0 else 0
        accuracy = pass_rate  # Same as pass rate in this context
        error_rate = (incorrect_answers / total_questions) * 100 if total_questions > 0 else 0
        
        return {
            "total_questions": total_questions,
            "average_score": average_score,
            "median_score": median_score,
            "std_score": std_score,
            "pass_rate": pass_rate,
            "accuracy": accuracy,
            "error_rate": error_rate,
            "min_score": min_score,
            "max_score": max_score,
            "q25_score": q25_score,
            "q75_score": q75_score
        }
    
    def _analyze_column(self, series: pd.Series, column_name: str) -> Dict[str, Any]:
        """Analyze a single column and return appropriate statistics.
        
        Args:
            series: Pandas Series to analyze
            column_name: Name of the column
            
        Returns:
            Dictionary with column analysis including type and statistics
        """
        # Remove null values for analysis
        clean_series = series.dropna()
        
        if len(clean_series) == 0:
            return {
                "type": "empty",
                "count": 0,
                "null_count": len(series)
            }
        
        # Try to convert to numeric
        numeric_series = pd.to_numeric(clean_series, errors='coerce')
        numeric_count = numeric_series.notna().sum()
        numeric_percentage = (numeric_count / len(clean_series)) * 100
        
        # Determine if column is primarily numerical (>80% numeric values)
        if numeric_percentage > 80 and numeric_count > 0:
            # Numerical column analysis
            numeric_clean = numeric_series.dropna()
            return {
                "type": "numerical",
                "count": len(clean_series),
                "null_count": len(series) - len(clean_series),
                "numeric_count": numeric_count,
                "mean": numeric_clean.mean(),
                "median": numeric_clean.median(),
                "std": numeric_clean.std(),
                "min": numeric_clean.min(),
                "max": numeric_clean.max(),
                "q25": numeric_clean.quantile(0.25),
                "q75": numeric_clean.quantile(0.75),
                "visualization_type": "histogram"
            }
        else:
            # Categorical column analysis
            value_counts = clean_series.value_counts()
            total_count = len(clean_series)
            
            # Limit to top 20 categories for performance
            top_categories = value_counts.head(20)
            
            return {
                "type": "categorical",
                "count": total_count,
                "null_count": len(series) - len(clean_series),
                "unique_count": len(value_counts),
                "top_categories": top_categories.to_dict(),
                "top_categories_percentage": (top_categories / total_count * 100).round(2).to_dict(),
                "most_common": value_counts.index[0] if len(value_counts) > 0 else None,
                "most_common_count": value_counts.iloc[0] if len(value_counts) > 0 else 0,
                "visualization_type": "bar_chart"
            }
    
    def generate_html_report(self, results: List[Dict[str, Any]], output_dir: Path, 
                           flow_id: str) -> Path:
        """Generate interactive HTML report with visualizations.
        
        Args:
            results: List of evaluation results
            output_dir: Directory to save the report
            flow_id: Flow ID for naming the report
            
        Returns:
            Path to generated HTML report
        """
        # Calculate summary statistics
        summary_stats = self.calculate_summary_stats(results)
        summary_stats['accuracy'] = summary_stats['accuracy'] / 100
        summary_stats['error_rate'] = summary_stats['error_rate'] / 100
        
        # Generate report filename
        from datetime import datetime
        date_str = datetime.now().strftime("%Y%m%d")
        report_name = f"eval_report_{date_str}-{flow_id}.html"
        report_path = output_dir / report_name
        
        # Generate HTML report
        return self.report_generator.generate_html_report(results, report_path, summary_stats)
