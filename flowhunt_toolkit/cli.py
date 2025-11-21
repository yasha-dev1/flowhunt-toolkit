#!/usr/bin/env python3
"""Main CLI module for Flowhunt Toolkit."""

import click
import sys
from typing import Optional
from pathlib import Path
import time
from datetime import datetime
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn
from rich.console import Console

from . import __version__
from .core.client import FlowHuntClient
from .core.evaluator import FlowEvaluator
from .core.liveagent_client import LiveAgentClient
from .utils.logger import Logger
from .utils.pdf_processor import PDFProcessor
from .utils.docx_processor import DOCXProcessor
from .utils.web_processor import WebProcessor
import pandas as pd


@click.group()
@click.version_option(version=__version__)
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
@click.pass_context
def main(ctx, verbose):
    """Flowhunt Toolkit - A CLI tool for FlowHunt Flow Engineers.
    
    This toolkit provides advanced utilities for FlowHunt flow development,
    evaluation, and management.
    """
    ctx.ensure_object(dict)
    ctx.obj['verbose'] = verbose
    
    if verbose:
        click.echo(f"Flowhunt Toolkit v{__version__}")


@main.command()
@click.argument('csv_file', type=click.Path(exists=True))
@click.argument('flow_id', type=str)
@click.option('--judge-flow-id', type=str, help='Custom LLM judge flow ID (uses default public flow if not specified)')
@click.option('--output-dir', '-o', type=click.Path(), help='Output Directory for evaluation results (default: eval_output)')
@click.option('--batch-size', type=int, default=10, help='Batch size for processing (default: 10)')
@click.option('--max-parallel', type=int, default=4, help='Maximum number of parallel evaluations (default: 4)')
@click.option('--check-interval', type=int, default=2, help='Seconds between result checks (default: 2)')
@click.pass_context
def evaluate(ctx, csv_file, flow_id, judge_flow_id, output_dir, batch_size, max_parallel, check_interval):
    """Evaluate a flow using LLM as a judge.
    
    This command takes a CSV file with 'flow_input' and 'expected_output' columns
    and evaluates the specified flow's performance using an LLM judge.
    
    CSV_FILE: Path to CSV file with flow_input and expected_output columns
    FLOW_ID: The FlowHunt flow ID to evaluate
    """
    verbose = ctx.obj.get('verbose', False)
    
    # Set default output directory if not specified
    if not output_dir:
        output_dir = "eval_output"
    
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if verbose:
        click.echo(f"Evaluating flow {flow_id} with CSV: {csv_file}")
        click.echo(f"Judge flow ID: {judge_flow_id or 'default public flow'}")
        click.echo(f"Batch size: {batch_size}")
        click.echo(f"Max parallel: {max_parallel}")
        click.echo(f"Check interval: {check_interval}s")
        click.echo(f"Output Directory: {output_path.absolute()}")
    
    try:
        # Initialize FlowHunt client
        try:
            client = FlowHuntClient.from_config_file()
        except FileNotFoundError:
            click.echo("Error: No FlowHunt configuration found. Please run 'flowhunt auth' first.", err=True)
            sys.exit(1)
        except Exception as e:
            click.echo(f"Error: Failed to initialize FlowHunt client: {str(e)}", err=True)
            sys.exit(1)
        
        # Initialize evaluator
        evaluator = FlowEvaluator(client, judge_flow_id)
        
        # Load evaluation data
        try:
            evaluation_data = evaluator.load_evaluation_data(Path(csv_file))
            click.echo(f"Loaded {len(evaluation_data)} questions from {csv_file}")
        except Exception as e:
            click.echo(f"Error: Failed to load CSV file: {str(e)}", err=True)
            sys.exit(1)
        
        # Run evaluation with parallel execution
        click.echo(f"Starting evaluation with {max_parallel} parallel workers...")
        results = evaluator.evaluate_parallel(
            flow_id=flow_id,
            evaluation_data=evaluation_data,
            max_parallel=max_parallel,
            check_interval=check_interval
        )
        
        # Calculate summary statistics
        summary = evaluator.calculate_summary_stats(results)
        
        # Display results
        click.echo("\nEvaluation completed!")
        click.echo(f"Total questions: {summary['total_questions']}")
        click.echo(f"Average score: {summary['average_score']:.2f}/10")
        click.echo(f"Median score: {summary['median_score']:.2f}/10")
        click.echo(f"Standard deviation: {summary['std_score']:.2f}")
        click.echo(f"Pass rate (≥7): {summary['pass_rate']:.1f}%")
        click.echo(f"Accuracy: {summary['accuracy']:.1f}%")
        click.echo(f"Error rate: {summary['error_rate']:.1f}%")
        
        # Save results as CSV with proper naming
        from datetime import datetime
        current_date = datetime.now().strftime("%Y%m%d")
        csv_filename = f"eval_results_{current_date}-{flow_id}.csv"
        csv_path = output_path / csv_filename
        
        try:
            evaluator.save_results(results, csv_path)
            click.echo(f"\nResults saved to {csv_path}")
        except Exception as e:
            click.echo(f"Warning: Failed to save results: {str(e)}", err=True)
        
        # Generate HTML report with visualizations
        try:
            click.echo("\nGenerating interactive HTML report...")
            report_path = evaluator.generate_html_report(results, output_path, flow_id)
            click.echo(f"HTML report saved to {report_path}")
            click.echo("\n🎉 Evaluation complete! Open the HTML report to view interactive visualizations.")
        except Exception as e:
            click.echo(f"Warning: Failed to generate HTML report: {str(e)}", err=True)
            if verbose:
                import traceback
                click.echo(f"Traceback: {traceback.format_exc()}", err=True)
        
    except KeyboardInterrupt:
        click.echo("\nEvaluation interrupted by user.", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        sys.exit(1)


@main.group()
@click.pass_context
def flows(ctx):
    """Manage FlowHunt flows.
    
    This command group provides utilities for listing and inspecting flows.
    """
    pass


@flows.command()
@click.argument('flow_id', type=str)
@click.option('--format', 'output_format', type=click.Choice(['json', 'yaml', 'table']), default='table', help='Output format')
@click.pass_context
def inspect(ctx, flow_id, output_format):
    """Inspect a FlowHunt flow's configuration and metadata.
    
    FLOW_ID: The FlowHunt flow ID to inspect
    """
    verbose = ctx.obj.get('verbose', False)
    
    if verbose:
        click.echo(f"Inspecting flow {flow_id} in {output_format} format")
    
    try:
        # Initialize FlowHunt client
        try:
            client = FlowHuntClient.from_config_file()
        except FileNotFoundError:
            click.echo("Error: No FlowHunt configuration found. Please run 'flowhunt auth' first.", err=True)
            sys.exit(1)
        except Exception as e:
            click.echo(f"Error: Failed to initialize FlowHunt client: {str(e)}", err=True)
            sys.exit(1)
        
        # Get flow information
        try:
            flow_info = client.get_flow_info(flow_id)
        except Exception as e:
            click.echo(f"Error: Failed to get flow information: {str(e)}", err=True)
            sys.exit(1)
        
        # Format and display output
        if output_format == 'json':
            import json
            from datetime import datetime
            
            def json_serializer(obj):
                """JSON serializer for objects not serializable by default json code"""
                if isinstance(obj, datetime):
                    return obj.isoformat()
                raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
            
            click.echo(json.dumps(flow_info, indent=2, default=json_serializer))
        elif output_format == 'yaml':
            try:
                import yaml
                from datetime import datetime
                
                def yaml_representer(dumper, data):
                    if isinstance(data, datetime):
                        return dumper.represent_scalar('tag:yaml.org,2002:str', data.isoformat())
                    return dumper.represent_data(data)
                
                yaml.add_representer(datetime, yaml_representer)
                click.echo(yaml.dump(flow_info, default_flow_style=False))
            except ImportError:
                click.echo("Error: PyYAML not installed. Install with: pip install pyyaml", err=True)
                sys.exit(1)
        else:  # table format
            click.echo(f"Flow ID: {flow_id}")
            click.echo(f"Name: {flow_info.get('name', 'N/A')}")
            
            # Handle description with proper formatting
            description = flow_info.get('description', 'N/A')
            if description and description != 'N/A':
                # Wrap long descriptions
                import textwrap
                wrapped_desc = textwrap.fill(description, width=80, initial_indent='Description: ', subsequent_indent='             ')
                click.echo(wrapped_desc)
            else:
                click.echo(f"Description: {description}")
            
            # Show actual fields from the response
            click.echo(f"Flow Type: {flow_info.get('flow_type', 'N/A')}")
            click.echo(f"Components: {flow_info.get('component_count', 'N/A')}")
            click.echo(f"Cache Enabled: {flow_info.get('enable_cache', 'N/A')}")
            
            # Handle executed_at date field
            if 'executed_at' in flow_info and flow_info['executed_at']:
                date_val = flow_info['executed_at']
                if hasattr(date_val, 'strftime'):
                    executed = date_val.strftime('%Y-%m-%d %H:%M:%S')
                else:
                    # Handle ISO string format
                    try:
                        from datetime import datetime
                        if isinstance(date_val, str):
                            dt = datetime.fromisoformat(date_val.replace('Z', '+00:00'))
                            executed = dt.strftime('%Y-%m-%d %H:%M:%S')
                        else:
                            executed = str(date_val)
                    except:
                        executed = str(date_val)
                click.echo(f"Last Executed: {executed}")
            else:
                click.echo("Last Executed: N/A")
            
            # Handle last_modified date field
            if 'last_modified' in flow_info and flow_info['last_modified']:
                date_val = flow_info['last_modified']
                if hasattr(date_val, 'strftime'):
                    modified = date_val.strftime('%Y-%m-%d %H:%M:%S')
                else:
                    try:
                        from datetime import datetime
                        if isinstance(date_val, str):
                            dt = datetime.fromisoformat(date_val.replace('Z', '+00:00'))
                            modified = dt.strftime('%Y-%m-%d %H:%M:%S')
                        else:
                            modified = str(date_val)
                    except:
                        modified = str(date_val)
                click.echo(f"Last Modified: {modified}")
            else:
                click.echo("Last Modified: N/A")
            
            # Show category ID if available
            if 'category_id' in flow_info and flow_info['category_id']:
                click.echo(f"Category ID: {flow_info['category_id']}")
        
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        sys.exit(1)


@main.command()
@click.argument('input_file', type=click.Path(exists=True))
@click.argument('flow_id', type=str)
@click.option('--output-dir', '-o', type=click.Path(), help='Output directory for generated files (for CSV with flow_input/filename columns)')
@click.option('--output-file', type=click.Path(), help='Output file for results (for JSON or other formats)')
@click.option("--format", type=click.Choice(['json', 'csv']), default='csv', help='Output format for results (default: csv)')
@click.option('--overwrite', is_flag=True, help='Overwrite existing files')
@click.option('--check-interval', type=int, default=2, help='Seconds between result checks (default: 2)')
@click.option('--max-parallel', type=int, default=50, help='Maximum number of tasks to schedule in parallel (default: 50)')
@click.option('--sequential', is_flag=True, help='Force sequential execution instead of default parallel processing')
@click.pass_context
def batch_run(ctx, input_file, flow_id, output_dir, output_file, format, overwrite, check_interval, max_parallel, sequential):
    """Run a flow in batch mode with multiple inputs.
    
    Supported input formats:
    - CSV: Columns supported: flow_input (mandatory), filename (optional), flow_variable (optional JSON)
    - JSON: Array of objects or single object with flow data
    - TXT: Each line becomes a flow_input
    
    When CSV 'filename' column is present, --output-dir must be specified.
    
    If no --output-file is specified, results are automatically saved as:
    {current-date}-{flow-id}-output.{format} in the current directory.
    
    INPUT_FILE: Path to file containing input data (CSV, JSON, or TXT)
    FLOW_ID: The FlowHunt flow ID to execute
    """
    verbose = ctx.obj.get('verbose', False)
    logger = Logger(verbose=verbose)
    
    # Validate that output_dir and output_file are not both specified
    if output_dir and output_file:
        logger.error("--output-dir and --output-file cannot be used together. Choose one.")
        sys.exit(1)
    
    # Log command start with configuration
    config_args = {
        'input_file': input_file,
        'flow_id': flow_id,
        'output_dir': output_dir,
        'output_file': output_file,
        'format': format,
        'overwrite': overwrite,
        'check_interval': f"{check_interval}s",
        'max_parallel': max_parallel,
        'mode': 'sequential' if sequential else 'parallel (default)'
    }
    logger.command_start('batch-run', config_args)
    
    # Add tip about check interval if it seems high
    if check_interval > 5 and not sequential:
        logger.warning(f"Check interval is {check_interval}s - consider using --check-interval=2 for faster feedback")

    start_time = time.time()

    try:
        # Initialize FlowHunt client
        logger.progress_start("Initializing FlowHunt client...")
        try:
            client = FlowHuntClient.from_config_file()
            logger.progress_done("FlowHunt client initialized")
        except FileNotFoundError:
            logger.error("No FlowHunt configuration found. Please run 'flowhunt auth' first.")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Failed to initialize FlowHunt client: {str(e)}")
            sys.exit(1)
        
        input_path = Path(input_file)
        
        # Check if this is a CSV file with the expected format for batch processing
        if input_path.suffix.lower() == '.csv':
            logger.progress_start("Analyzing CSV file format...")
            try:
                # Validate CSV format and columns
                import csv
                with open(input_path, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    fieldnames = reader.fieldnames
                    
                    # Check if CSV has headers
                    if not fieldnames:
                        logger.error("CSV file must have headers.")
                        sys.exit(1)
                    
                    # Define allowed columns
                    mandatory_columns = {'flow_input'}
                    optional_columns = {'filename', 'flow_variable'}
                    allowed_columns = mandatory_columns | optional_columns
                    
                    # Check for mandatory columns
                    missing_mandatory = mandatory_columns - set(fieldnames)
                    if missing_mandatory:
                        logger.error(f"CSV file must contain the following mandatory column(s): {', '.join(missing_mandatory)}")
                        sys.exit(1)
                    
                    # Check for invalid columns
                    invalid_columns = set(fieldnames) - allowed_columns
                    if invalid_columns:
                        logger.error(f"CSV file contains invalid column(s): {', '.join(invalid_columns)}")
                        logger.info(f"Allowed columns are: {', '.join(sorted(allowed_columns))}")
                        sys.exit(1)
                    
                    # Check if filename column exists and output_dir is required
                    has_filename = 'filename' in fieldnames
                    has_flow_variable = 'flow_variable' in fieldnames
                    
                    if has_filename and not output_dir:
                        logger.error("--output-dir is required when CSV file contains 'filename' column.")
                        sys.exit(1)
                    
                    # Check if output_dir is specified without filename column
                    if not has_filename and output_dir:
                        logger.error("--output-dir only makes sense when CSV file contains 'filename' column.")
                        sys.exit(1)
                
                logger.progress_done("CSV format validated")
                
                # If we reach here, CSV validation passed
                # Use parallel processing by default unless --sequential is specified
                if not sequential:
                    # Parallel execution is now the default
                    if not (has_filename or has_flow_variable):
                        logger.info("🚀 Using parallel batch processing (default mode)")
                        logger.info("💡 Results will be saved to output file. Use --sequential for one-by-one processing.")
                        # For parallel without filename column, we need to set output_dir to None
                        # Results will be aggregated in memory and saved to output_file
                        output_dir_for_batch = None
                        force_parallel = True  # Signal to aggregate results
                    else:
                        output_dir_for_batch = output_dir
                        force_parallel = False
                        if has_filename:
                            logger.info("Using parallel batch processing with individual file outputs")
                        else:
                            logger.info("Using parallel batch processing with flow variables")
                    
                    logger.progress_start("Starting parallel batch execution...")
                    stats = client.batch_execute_from_csv(
                        csv_file=str(input_path),
                        flow_id=flow_id,
                        output_dir=output_dir_for_batch,
                        overwrite=overwrite,
                        check_interval=check_interval,
                        max_parallel=max_parallel,
                        force_parallel=force_parallel
                    )
                    
                    duration = time.time() - start_time
                    logger.progress_done("Batch execution completed", duration)
                    
                    # Display results
                    logger.stats_table("Execution Results", {
                        "Total inputs": stats['total'],
                        "Completed successfully": stats['completed'],
                        "Failed": stats['failed'],
                        "Skipped (files already exist)": stats['skipped']
                    })
                    
                    # If force_parallel without filename, results are in stats['results']
                    if force_parallel and not has_filename and 'results' in stats:
                        # Generate automatic output file if not specified
                        if not output_file:
                            current_date = datetime.now().strftime("%Y-%m-%d")
                            output_file = f"{current_date}-{flow_id}-output.{format}"
                            logger.info(f"Saving aggregated results to: {output_file}")

                        # Save aggregated results
                        logger.progress_start(f"Saving results to {output_file}...")
                        try:
                            output_path = Path(output_file)
                            
                            if format == 'csv' or output_path.suffix.lower() == '.csv':
                                import pandas as pd
                                df = pd.DataFrame(stats['results'])
                                df.to_csv(output_path, index=False)
                            else:
                                import json
                                with open(output_path, 'w') as f:
                                    json.dump(stats['results'], f, indent=2)
                            
                            logger.progress_done(f"Results saved to {output_file}")
                        except Exception as e:
                            logger.warning(f"Failed to save results: {str(e)}")
                    
                    return
                else:
                    # User explicitly requested sequential processing
                    logger.info("📝 Using sequential processing (--sequential flag specified)")
                    logger.warning("Sequential processing is slower. Remove --sequential flag for parallel execution.")

                    # Read topics from CSV for sequential processing
                    topics = []
                    with open(input_path, 'r', encoding='utf-8') as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            flow_input = row.get('flow_input', '').strip()
                            if flow_input:
                                topic = {'flow_input': flow_input}
                                if 'filename' in row and row['filename']:
                                    topic['filename'] = row['filename'].strip()
                                if 'flow_variable' in row and row['flow_variable']:
                                    import json
                                    try:
                                        topic['flow_variable'] = json.loads(row['flow_variable'].strip())
                                    except json.JSONDecodeError:
                                        pass
                                topics.append(topic)

                    logger.progress_start("Starting sequential batch execution...")

                    from rich.console import Console
                    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn

                    console = Console()
                    completed = 0
                    failed = 0
                    skipped = 0

                    with Progress(
                        SpinnerColumn(),
                        TextColumn("[progress.description]{task.description}"),
                        BarColumn(complete_style="yellow", finished_style="bold yellow"),
                        TaskProgressColumn(),
                        TextColumn("•"),
                        TimeRemainingColumn(),
                        console=console,
                        transient=False
                    ) as progress:

                        task = progress.add_task(
                            "[yellow]Sequential processing...",
                            total=len(topics)
                        )

                        for topic in topics:
                            try:
                                # Check if file already exists (skip if not overwriting)
                                if has_filename and output_dir and not overwrite and 'filename' in topic:
                                    file_path = Path(output_dir) / topic['filename']
                                    if file_path.exists():
                                        skipped += 1
                                        progress.advance(task, 1)
                                        continue

                                # Build variables dictionary
                                variables = {}
                                if 'filename' in topic:
                                    variables['filename'] = topic['filename']
                                if 'flow_variable' in topic and isinstance(topic['flow_variable'], dict):
                                    variables.update(topic['flow_variable'])

                                # Execute flow
                                result = client.execute_flow(flow_id, variables=variables, human_input=topic['flow_input'])

                                if result:
                                    # Save to file if filename column exists and output_dir is specified
                                    if has_filename and output_dir and 'filename' in topic:
                                        output_dir_path = Path(output_dir)
                                        output_dir_path.mkdir(parents=True, exist_ok=True)
                                        file_path = output_dir_path / topic['filename']
                                        file_path.parent.mkdir(parents=True, exist_ok=True)
                                        with open(file_path, 'w', encoding='utf-8') as f:
                                            f.write(result)
                                    completed += 1
                                else:
                                    failed += 1

                                # Update progress
                                progress.update(
                                    task,
                                    description=f"[yellow]Sequential[/yellow] │ [green]{completed} ✓[/green] [red]{failed} ✗[/red] [dim]{skipped} skipped[/dim]"
                                )

                            except Exception as e:
                                failed += 1
                                if verbose:
                                    console.print(f"[red]✗[/red] Failed: {str(e)[:60]}...")

                            progress.advance(task, 1)

                    duration = time.time() - start_time
                    logger.progress_done("Sequential execution completed", duration)

                    # Display summary
                    logger.stats_table("Execution Results", {
                        "Total inputs": len(topics),
                        "Completed successfully": completed,
                        "Failed": failed,
                        "Skipped (files already exist)": skipped
                    })

                    return

            except Exception as e:
                if "CSV file must have headers" in str(e) or "CSV file must contain" in str(e) or "CSV file contains invalid" in str(e) or "--output-dir is required" in str(e):
                    # Re-raise validation errors
                    raise
                logger.debug(f"Could not use optimized batch processing: {e}")
                logger.warning("Falling back to standard sequential processing...")
        
        # Standard processing for other formats or CSV without expected columns
        logger.progress_start("Loading input data...")
        inputs_list = []
        try:
            if input_path.suffix.lower() == '.csv':
                import pandas as pd
                data = pd.read_csv(input_path)
                inputs_list = data.to_dict('records')
            elif input_path.suffix.lower() == '.json':
                import json
                with open(input_path, 'r') as f:
                    file_content = json.load(f)
                    if isinstance(file_content, list):
                        inputs_list = file_content
                    else:
                        inputs_list = [file_content]
            elif input_path.suffix.lower() == '.txt':
                # TXT file support - each line is a flow_input
                with open(input_path, 'r', encoding='utf-8') as f:
                    lines = [line.strip() for line in f.readlines() if line.strip()]
                    inputs_list = [{'flow_input': line} for line in lines]
            else:
                logger.error("Unsupported file format. Use .csv, .json, or .txt files.")
                sys.exit(1)
                
            logger.progress_done(f"Loaded {len(inputs_list)} inputs from {input_file}")
            
        except Exception as e:
            logger.error(f"Failed to load input file: {str(e)}")
            sys.exit(1)
        
        # Generate automatic output file if not specified
        if not output_file:
            current_date = datetime.now().strftime("%Y-%m-%d")
            output_file = f"{current_date}-{flow_id}-output.{format}"
            logger.info(f"Auto-generating output file: {output_file}")
        
        # Check if user wants sequential or parallel (default)
        if sequential:
            # Sequential processing
            logger.info("📝 Using sequential processing (--sequential flag specified)")
            logger.warning("Sequential processing is slower. Remove --sequential flag for parallel execution.")
            logger.progress_start("Starting sequential batch execution...")
            
            results = []
            successful = 0
            failed = 0
            console = Console()
            
            # Use Rich progress bar for sequential processing
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(complete_style="yellow", finished_style="bold yellow"),
                TaskProgressColumn(),
                TextColumn("•"),
                TimeRemainingColumn(),
                console=console,
                transient=False
            ) as progress:
                
                task = progress.add_task(
                    "[yellow]Sequential processing...", 
                    total=len(inputs_list)
                )
                
                for i, inputs in enumerate(inputs_list):
                    try:
                        # Convert inputs to variables and human_input format
                        if isinstance(inputs, dict):
                            # Use 'flow_input' key if present, otherwise use 'human_input' key, otherwise use all as variables
                            if 'flow_input' in inputs:
                                human_input = inputs.pop('flow_input', '')
                                variables = inputs
                            else:
                                human_input = inputs.pop('human_input', '')
                                variables = inputs
                        else:
                            human_input = str(inputs)
                            variables = {}
                        
                        # Update progress description with current task info
                        progress.update(
                            task, 
                            description=f"[yellow]Sequential[/yellow] │ [green]{successful} ✓[/green] [red]{failed} ✗[/red] │ {human_input[:30]}{'...' if len(human_input) > 30 else ''}"
                        )
                        
                        result = client.execute_flow(flow_id, variables=variables, human_input=human_input)
                        results.append({
                            'input_index': i,
                            'flow_input': human_input,
                            'variables': variables,
                            'result': result,
                            'status': 'success'
                        })
                        successful += 1
                        
                        if verbose:
                            console.print(f"[green]✓[/green] Flow {i+1}/{len(inputs_list)} completed successfully")
                            
                    except Exception as e:
                        results.append({
                            'input_index': i,
                            'flow_input': human_input if 'human_input' in locals() else str(inputs),
                            'variables': variables if 'variables' in locals() else {},
                            'result': None,
                            'status': 'error',
                            'error': str(e)
                        })
                        failed += 1
                        
                        if verbose:
                            console.print(f"[red]✗[/red] Flow {i+1}/{len(inputs_list)} failed: {str(e)}")
                    
                    progress.advance(task, 1)
            
            duration = time.time() - start_time
            logger.progress_done("Sequential execution completed", duration)
            
            # Display summary
            logger.stats_table("Execution Results", {
                "Total inputs": len(results),
                "Successful": successful,
                "Failed": failed
            })
        else:
            # Parallel processing (DEFAULT)
            logger.info("🚀 Using parallel batch processing (default mode)")
            logger.progress_start("Starting parallel batch execution...")
            
            # Convert inputs_list to topics format for batch processing
            topics = []
            for i, inputs in enumerate(inputs_list):
                if isinstance(inputs, dict):
                    if 'flow_input' in inputs:
                        topic = {'flow_input': inputs['flow_input']}
                        # Add other fields as flow_variable
                        remaining = {k: v for k, v in inputs.items() if k != 'flow_input'}
                        if remaining:
                            topic['flow_variable'] = remaining
                    else:
                        # Use all as flow_variable
                        topic = {'flow_input': str(inputs), 'flow_variable': inputs}
                else:
                    topic = {'flow_input': str(inputs)}
                topics.append(topic)
            
            # Create temporary CSV for batch processing
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as temp_csv:
                import csv as csv_module
                fieldnames = ['flow_input']
                if any('flow_variable' in t for t in topics):
                    fieldnames.append('flow_variable')
                
                writer = csv_module.DictWriter(temp_csv, fieldnames=fieldnames)
                writer.writeheader()
                
                for topic in topics:
                    row = {'flow_input': topic['flow_input']}
                    if 'flow_variable' in topic:
                        import json
                        row['flow_variable'] = json.dumps(topic['flow_variable'])
                    writer.writerow(row)
                
                temp_csv_path = temp_csv.name
            
            # Run parallel batch processing
            stats = client.batch_execute_from_csv(
                csv_file=temp_csv_path,
                flow_id=flow_id,
                output_dir=None,  # No output dir for aggregated results
                overwrite=overwrite,
                check_interval=check_interval,
                max_parallel=max_parallel,
                force_parallel=True  # Force parallel for aggregated results
            )
            
            # Clean up temp file
            Path(temp_csv_path).unlink()
            
            duration = time.time() - start_time
            logger.progress_done("Parallel execution completed", duration)
            
            # Display results
            logger.stats_table("Execution Results", {
                "Total inputs": stats['total'],
                "Completed successfully": stats['completed'],
                "Failed": stats['failed']
            })
            
            # Get results from stats
            results = stats.get('results', [])
        
        # Save results if output file specified or auto-generated
        if output_file and results:
            logger.progress_start(f"Saving results to {output_file}...")
            try:
                output_path = Path(output_file)
                
                if format == 'csv' or output_path.suffix.lower() == '.csv':
                    import pandas as pd
                    # Flatten results for CSV
                    flattened = []
                    for r in results:
                        row = {
                            'input_index': r['input_index'],
                            'flow_input': r['flow_input'],
                            'status': r['status']
                        }
                        # Add variables as separate columns
                        if r['variables']:
                            for key, value in r['variables'].items():
                                row[f'var_{key}'] = str(value)
                        
                        if r['status'] == 'success':
                            row['result'] = str(r['result'])
                        else:
                            row['error'] = r.get('error', '')
                        flattened.append(row)
                    
                    df = pd.DataFrame(flattened)
                    df.to_csv(output_path, index=False)
                else:
                    # JSON format
                    import json
                    with open(output_path, 'w') as f:
                        json.dump(results, f, indent=2)
                
                logger.progress_done(f"Results saved to {output_file}")
            except Exception as e:
                logger.warning(f"Failed to save results: {str(e)}")

    except KeyboardInterrupt:
        logger.error("Batch execution interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Batch execution failed: {str(e)}")
        sys.exit(1)


@main.command()
@click.argument('csv_file', type=click.Path(exists=True))
@click.argument('flow_id', type=str)
@click.option('--output-file', type=click.Path(), help='Output CSV file for results (default: auto-generated)')
@click.option('--col-variable-name', type=str, default='col_name', help='Variable name for column headers (default: col_name)')
@click.option('--check-interval', type=int, default=2, help='Seconds between result checks (default: 2)')
@click.option('--max-parallel', type=int, default=50, help='Maximum number of tasks to schedule in parallel (default: 50)')
@click.pass_context
def batch_run_matrix(ctx, csv_file, flow_id, output_file, col_variable_name, check_interval, max_parallel):
    """Run a flow for each cell in a CSV matrix (first column = input source).

    This command processes each row in a CSV where the FIRST COLUMN contains the
    input value, and each subsequent column represents a different processing variant.
    For each cell in columns 2+, the first column's value is sent as flow_input,
    and the column name is passed as a flow variable.

    The output CSV preserves the first column as-is and replaces other cells with
    flow results. Rows with empty first column are skipped. Failed executions will
    show error messages in their cells.

    Example CSV input:
        Input,Variant1,Variant2,Variant3
        What is AI?,"","",""
        Python features?,"","",""

    For each row, the first column value is sent to all other columns:
    - Row 1, Variant1: flow_input="What is AI?", variables={"col_name": "Variant1"}
    - Row 1, Variant2: flow_input="What is AI?", variables={"col_name": "Variant2"}
    - Row 1, Variant3: flow_input="What is AI?", variables={"col_name": "Variant3"}
    - (Same pattern for Row 2 with "Python features?")

    CSV must have at least 2 columns (first = input, rest = processing columns).
    Total executions = (non-empty rows) × (columns - 1)

    CSV_FILE: Path to CSV file (first column = input source, rest = processing columns)
    FLOW_ID: The FlowHunt flow ID to execute for each cell
    """
    verbose = ctx.obj.get('verbose', False)
    logger = Logger(verbose=verbose)

    # Generate automatic output file if not specified
    if not output_file:
        current_date = datetime.now().strftime("%Y-%m-%d")
        output_file = f"{current_date}-{flow_id}-matrix-output.csv"
        logger.info(f"Auto-generating output file: {output_file}")

    # Log command start with configuration
    config_args = {
        'csv_file': csv_file,
        'flow_id': flow_id,
        'output_file': output_file,
        'col_variable_name': col_variable_name,
        'check_interval': f"{check_interval}s",
        'max_parallel': max_parallel
    }
    logger.command_start('batch-run-matrix', config_args)

    start_time = time.time()

    try:
        # Initialize FlowHunt client
        logger.progress_start("Initializing FlowHunt client...")
        try:
            client = FlowHuntClient.from_config_file()
            logger.progress_done("FlowHunt client initialized")
        except FileNotFoundError:
            logger.error("No FlowHunt configuration found. Please run 'flowhunt auth' first.")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Failed to initialize FlowHunt client: {str(e)}")
            sys.exit(1)

        # Execute matrix batch processing
        logger.info(f"🔢 Starting CSV matrix batch processing...")
        logger.info(f"📊 Column variable name: '{col_variable_name}'")
        logger.info(f"⚡ Max parallel: {max_parallel} workers")

        try:
            stats = client.batch_execute_matrix_from_csv(
                csv_file=csv_file,
                flow_id=flow_id,
                output_file=output_file,
                col_variable_name=col_variable_name,
                check_interval=check_interval,
                max_parallel=max_parallel
            )

            duration = time.time() - start_time
            logger.progress_done("Matrix batch execution completed", duration)

            # Display results
            logger.stats_table("Execution Results", {
                "Total cells processed": stats['total'],
                "Completed successfully": stats['completed'],
                "Failed": stats['failed'],
                "Output file": output_file
            })

            logger.info(f"✅ Results saved to {output_file}")

        except Exception as e:
            logger.error(f"Matrix batch execution failed: {str(e)}")
            sys.exit(1)

    except KeyboardInterrupt:
        logger.error("Matrix batch execution interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        sys.exit(1)


@main.command()
@click.option('--base-url', default='https://api.flowhunt.io', help='FlowHunt API base URL (protocol + domain)')
@click.pass_context
def auth(ctx, base_url):
    """Authenticate with FlowHunt API.

    This command will prompt for API credentials and store them securely.
    You can optionally specify a custom base URL for the API endpoint.
    """
    verbose = ctx.obj.get('verbose', False)

    if verbose:
        click.echo("Setting up FlowHunt API authentication")
        click.echo(f"Using API endpoint: {base_url}")

    # Prompt for API key
    api_key = click.prompt("Enter your FlowHunt API key", hide_input=True, type=str)

    try:
        # Test the credentials by creating a client and making a simple API call
        client = FlowHuntClient(api_key=api_key, base_url=base_url)

        click.echo("Testing API connection...")
        # Try to list flows to verify the credentials work
        flows = client.list_flows()

        # Save configuration
        client.save_config()

        click.echo("✓ Authentication successful!")
        click.echo(f"✓ Configuration saved to {Path.home() / '.flowhunt' / 'config.json'}")
        click.echo(f"✓ API endpoint: {base_url}")
        click.echo(f"✓ Found {len(flows)} flows in your account")

    except Exception as e:
        click.echo(f"✗ Authentication failed: {str(e)}", err=True)
        sys.exit(1)


@flows.command()
@click.option('--format', 'output_format', type=click.Choice(['json', 'table']), default='table', help='Output format')
@click.pass_context
def list(ctx, output_format):
    """List available FlowHunt flows.
    """
    verbose = ctx.obj.get('verbose', False)
    
    if verbose:
        click.echo(f"Listing flows in {output_format} format")
    
    try:
        # Initialize FlowHunt client
        try:
            client = FlowHuntClient.from_config_file()
        except FileNotFoundError:
            click.echo("Error: No FlowHunt configuration found. Please run 'flowhunt auth' first.", err=True)
            sys.exit(1)
        except Exception as e:
            click.echo(f"Error: Failed to initialize FlowHunt client: {str(e)}", err=True)
            sys.exit(1)
        
        # Get flows list
        try:
            flows = client.list_flows()
        except Exception as e:
            click.echo(f"Error: Failed to list flows: {str(e)}", err=True)
            sys.exit(1)
        
        if not flows:
            click.echo("No flows found in your account.")
            return
        
        # Format and display output
        if output_format == 'json':
            import json
            from datetime import datetime
            
            def json_serializer(obj):
                """JSON serializer for objects not serializable by default json code"""
                if isinstance(obj, datetime):
                    return obj.isoformat()
                raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
            
            click.echo(json.dumps(flows, indent=2, default=json_serializer))
        else:  # table format
            click.echo(f"Found {len(flows)} flows:\n")
            
            # First, let's examine the actual structure by printing one flow in verbose mode
            if verbose and flows:
                click.echo("Debug: First flow structure:")
                import json
                from datetime import datetime
                
                def json_serializer(obj):
                    if isinstance(obj, datetime):
                        return obj.isoformat()
                    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
                
                click.echo(json.dumps(flows[0], indent=2, default=json_serializer))
                click.echo("\n" + "="*50 + "\n")
            
            # Table header - show relevant information from the actual response
            click.echo(f"{'ID':<36} {'Name':<25} {'Components':<10} {'Last Executed':<20}")
            click.echo("-" * 100)
            
            # Table rows
            for flow in flows:
                # Don't truncate the flow ID - show full UUID
                flow_id = str(flow.get('id', 'N/A'))
                name = str(flow.get('name', 'N/A'))[:24]
                component_count = str(flow.get('component_count', 'N/A'))[:9]
                
                # Handle executed_at date field
                executed = 'N/A'
                if 'executed_at' in flow and flow['executed_at']:
                    date_val = flow['executed_at']
                    if hasattr(date_val, 'strftime'):
                        executed = date_val.strftime('%Y-%m-%d %H:%M')
                    else:
                        # Handle ISO string format
                        try:
                            from datetime import datetime
                            if isinstance(date_val, str):
                                # Parse ISO format string
                                dt = datetime.fromisoformat(date_val.replace('Z', '+00:00'))
                                executed = dt.strftime('%Y-%m-%d %H:%M')
                            else:
                                executed = str(date_val)[:19]
                        except:
                            executed = str(date_val)[:19]
                
                click.echo(f"{flow_id:<36} {name:<25} {component_count:<10} {executed:<20}")
        
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        sys.exit(1)


@main.group()
@click.pass_context
def index(ctx):
    """Index external data sources into FlowHunt.
    
    This command group provides utilities for indexing data from various
    external sources by processing them through FlowHunt flows.
    """
    pass


@index.command()
@click.argument('pdf_path', type=click.Path(exists=True))
@click.argument('index_flow_id', type=str)
@click.option('--max-tokens', type=int, default=4000, help='Maximum tokens per chunk (default: 4000)')
@click.option('--output-csv', type=click.Path(), help='Path to save processing results CSV')
@click.option('--sequential', is_flag=True, help='Process chunks sequentially, waiting for each to complete before starting the next')
@click.pass_context
def pdf(ctx, pdf_path, index_flow_id, max_tokens, output_csv, sequential):
    """Index a PDF file by processing text chunks through a FlowHunt flow.

    This command extracts text from a PDF file, chunks it based on token count,
    and processes each chunk through a FlowHunt flow for indexing.

    PDF_PATH: Path to the PDF file to process
    INDEX_FLOW_ID: FlowHunt flow ID to use for processing chunks
    """
    verbose = ctx.obj.get('verbose', False)
    logger = Logger(verbose=verbose)

    pdf_file = Path(pdf_path)

    # Generate default CSV filename if not provided
    if not output_csv:
        current_date = datetime.now().strftime("%Y%m%d")
        output_csv = f"pdf_index_{current_date}_{pdf_file.stem}.csv"

    output_path = Path(output_csv)

    # Log command start
    config_args = {
        'pdf_path': str(pdf_file),
        'index_flow_id': index_flow_id,
        'max_tokens': max_tokens,
        'output_csv': str(output_path),
        'sequential': sequential
    }
    logger.command_start('index pdf', config_args)

    start_time = time.time()

    try:
        # Initialize FlowHunt client
        logger.progress_start("Initializing FlowHunt client...")
        try:
            flowhunt_client = FlowHuntClient.from_config_file()
            logger.progress_done("FlowHunt client initialized")
        except FileNotFoundError:
            logger.error("No FlowHunt configuration found. Please run 'flowhunt auth' first.")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Failed to initialize FlowHunt client: {str(e)}")
            sys.exit(1)

        # Initialize PDF processor
        logger.progress_start("Initializing PDF processor...")
        try:
            pdf_processor = PDFProcessor(max_tokens=max_tokens)
            logger.progress_done("PDF processor initialized")
        except Exception as e:
            logger.error(f"Failed to initialize PDF processor: {str(e)}")
            sys.exit(1)

        # Extract and chunk PDF text
        logger.progress_start(f"Processing PDF: {pdf_file.name}...")
        try:
            chunks = pdf_processor.process_pdf(pdf_file, max_tokens)
            logger.progress_done(f"PDF processed into {len(chunks)} chunks")
        except Exception as e:
            logger.error(f"Failed to process PDF: {str(e)}")
            sys.exit(1)

        if not chunks:
            logger.info("No text chunks extracted from PDF")
            return

        # Display chunk statistics
        total_tokens = sum(token_count for _, token_count in chunks)
        avg_tokens = total_tokens // len(chunks) if chunks else 0
        logger.stats_table("PDF Processing Summary", {
            "Total chunks": len(chunks),
            "Total tokens": total_tokens,
            "Average tokens per chunk": avg_tokens,
            "Max tokens per chunk": max_tokens
        })

        # Process chunks through FlowHunt flow
        logger.info(f"Starting to index {len(chunks)} chunks through flow...")

        successful = 0
        failed = 0
        results_data = []
        console = Console()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(complete_style="green", finished_style="bold green"),
            TaskProgressColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
            console=console,
            transient=False
        ) as progress:

            task = progress.add_task(
                "[green]Indexing chunks...",
                total=len(chunks)
            )

            for chunk_idx, (chunk_text, token_count) in enumerate(chunks, 1):
                try:
                    # Update progress description
                    chunk_preview = chunk_text[:50].replace('\n', ' ')
                    progress.update(
                        task,
                        description=f"[green]Indexing[/green] │ [green]{successful} ✓[/green] [red]{failed} ✗[/red] │ Chunk {chunk_idx}/{len(chunks)}: {chunk_preview}..."
                    )

                    # Invoke FlowHunt flow
                    process_id = flowhunt_client.invoke_flow(
                        flow_id=index_flow_id,
                        human_input=chunk_text,
                        variables={
                            'chunk_index': chunk_idx,
                            'total_chunks': len(chunks),
                            'token_count': token_count,
                            'pdf_filename': pdf_file.name,
                            'source': 'pdf'
                        },
                        singleton=False,
                    )

                    # If sequential mode, poll until completion
                    if sequential:
                        if verbose:
                            console.print(f"[blue]⏳[/blue] Waiting for chunk {chunk_idx}/{len(chunks)} to complete (process: {process_id})...")

                        poll_interval = 2  # seconds
                        max_wait_time = 300  # 5 minutes timeout
                        elapsed_time = 0

                        while elapsed_time < max_wait_time:
                            is_ready, result = flowhunt_client.get_flow_results(index_flow_id, process_id)

                            if is_ready:
                                if result and result != "NOCONTENT":
                                    # Success
                                    results_data.append({
                                        'chunk_index': chunk_idx,
                                        'token_count': token_count,
                                        'chunk_preview': chunk_text[:100].replace('\n', ' '),
                                        'flow_process_id': process_id,
                                        'status': 'success',
                                        'result': result[:200] + ('...' if len(result) > 200 else ''),
                                        'indexed_at': datetime.now().isoformat()
                                    })
                                    successful += 1

                                    if verbose:
                                        console.print(f"[green]✓[/green] Chunk {chunk_idx}/{len(chunks)} completed successfully ({token_count} tokens)")
                                else:
                                    # Completed but no content
                                    results_data.append({
                                        'chunk_index': chunk_idx,
                                        'token_count': token_count,
                                        'chunk_preview': chunk_text[:100].replace('\n', ' '),
                                        'flow_process_id': process_id,
                                        'status': 'failed',
                                        'error': 'No content returned',
                                        'indexed_at': datetime.now().isoformat()
                                    })
                                    failed += 1

                                    if verbose:
                                        console.print(f"[red]✗[/red] Chunk {chunk_idx}/{len(chunks)} returned no content")

                                break

                            # Wait before next poll
                            time.sleep(poll_interval)
                            elapsed_time += poll_interval

                        # Check if timed out
                        if elapsed_time >= max_wait_time:
                            results_data.append({
                                'chunk_index': chunk_idx,
                                'token_count': token_count,
                                'chunk_preview': chunk_text[:100].replace('\n', ' '),
                                'flow_process_id': process_id,
                                'status': 'failed',
                                'error': 'Timeout waiting for completion',
                                'indexed_at': datetime.now().isoformat()
                            })
                            failed += 1

                            if verbose:
                                console.print(f"[red]✗[/red] Chunk {chunk_idx}/{len(chunks)} timed out")

                    else:
                        # Non-sequential mode - just add to results data immediately
                        results_data.append({
                            'chunk_index': chunk_idx,
                            'token_count': token_count,
                            'chunk_preview': chunk_text[:100].replace('\n', ' '),
                            'flow_process_id': process_id,
                            'status': 'success',
                            'indexed_at': datetime.now().isoformat()
                        })

                        successful += 1

                        if verbose:
                            console.print(f"[green]✓[/green] Indexed chunk {chunk_idx}/{len(chunks)} ({token_count} tokens, process: {process_id})")

                        # Rate limiting to be nice to the API
                        time.sleep(0.5)

                except Exception as e:
                    failed += 1

                    if verbose:
                        console.print(f"[red]✗[/red] Failed to index chunk {chunk_idx}/{len(chunks)}: {str(e)}")

                    # Still add to results with error status
                    results_data.append({
                        'chunk_index': chunk_idx,
                        'token_count': token_count,
                        'chunk_preview': chunk_text[:100].replace('\n', ' '),
                        'flow_process_id': f"ERROR: {str(e)}",
                        'status': 'failed',
                        'indexed_at': datetime.now().isoformat()
                    })

                progress.advance(task, 1)

        # Save results to CSV
        if results_data:
            logger.progress_start(f"Saving results to {output_csv}...")
            try:
                df = pd.DataFrame(results_data)
                df.to_csv(output_path, index=False)
                logger.progress_done(f"Results saved to {output_csv}")
            except Exception as e:
                logger.warning(f"Failed to save results: {str(e)}")

        duration = time.time() - start_time
        logger.progress_done("PDF indexing completed", duration)

        # Display summary
        logger.stats_table("Indexing Results", {
            "Total chunks": len(chunks),
            "Successfully indexed": successful,
            "Failed": failed,
            "Results saved to": str(output_path)
        })

        logger.info(f"✅ PDF indexing complete! Results saved to {output_path}")

    except KeyboardInterrupt:
        logger.error("Indexing interrupted by user")
        if results_data:
            df = pd.DataFrame(results_data)
            df.to_csv(output_path, index=False)
            logger.info(f"Partial results saved to {output_path}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Indexing failed: {str(e)}")
        if 'results_data' in locals() and results_data:
            df = pd.DataFrame(results_data)
            df.to_csv(output_path, index=False)
            logger.info(f"Partial results saved to {output_path}")
        sys.exit(1)


@index.command()
@click.argument('docx_path', type=click.Path(exists=True))
@click.argument('index_flow_id', type=str)
@click.option('--max-tokens', type=int, default=4000, help='Maximum tokens per chunk (default: 4000)')
@click.option('--output-csv', type=click.Path(), help='Path to save processing results CSV')
@click.pass_context
def docx(ctx, docx_path, index_flow_id, max_tokens, output_csv):
    """Index a DOCX file by processing text chunks through a FlowHunt flow.

    This command extracts text from a DOCX file, chunks it based on token count,
    and processes each chunk through a FlowHunt flow for indexing.

    DOCX_PATH: Path to the DOCX file to process
    INDEX_FLOW_ID: FlowHunt flow ID to use for processing chunks
    """
    verbose = ctx.obj.get('verbose', False)
    logger = Logger(verbose=verbose)

    docx_file = Path(docx_path)

    # Generate default CSV filename if not provided
    if not output_csv:
        current_date = datetime.now().strftime("%Y%m%d")
        output_csv = f"docx_index_{current_date}_{docx_file.stem}.csv"

    output_path = Path(output_csv)

    # Log command start
    config_args = {
        'docx_path': str(docx_file),
        'index_flow_id': index_flow_id,
        'max_tokens': max_tokens,
        'output_csv': str(output_path)
    }
    logger.command_start('index docx', config_args)

    start_time = time.time()

    try:
        # Initialize FlowHunt client
        logger.progress_start("Initializing FlowHunt client...")
        try:
            flowhunt_client = FlowHuntClient.from_config_file()
            logger.progress_done("FlowHunt client initialized")
        except FileNotFoundError:
            logger.error("No FlowHunt configuration found. Please run 'flowhunt auth' first.")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Failed to initialize FlowHunt client: {str(e)}")
            sys.exit(1)

        # Initialize DOCX processor
        logger.progress_start("Initializing DOCX processor...")
        try:
            docx_processor = DOCXProcessor(max_tokens=max_tokens)
            logger.progress_done("DOCX processor initialized")
        except Exception as e:
            logger.error(f"Failed to initialize DOCX processor: {str(e)}")
            sys.exit(1)

        # Extract and chunk DOCX text
        logger.progress_start(f"Processing DOCX: {docx_file.name}...")
        try:
            chunks = docx_processor.process_docx(docx_file, max_tokens)
            logger.progress_done(f"DOCX processed into {len(chunks)} chunks")
        except Exception as e:
            logger.error(f"Failed to process DOCX: {str(e)}")
            sys.exit(1)

        if not chunks:
            logger.info("No text chunks extracted from DOCX")
            return

        # Display chunk statistics
        total_tokens = sum(token_count for _, token_count in chunks)
        avg_tokens = total_tokens // len(chunks) if chunks else 0
        logger.stats_table("DOCX Processing Summary", {
            "Total chunks": len(chunks),
            "Total tokens": total_tokens,
            "Average tokens per chunk": avg_tokens,
            "Max tokens per chunk": max_tokens
        })

        # Process chunks through FlowHunt flow
        logger.info(f"Starting to index {len(chunks)} chunks through flow...")

        successful = 0
        failed = 0
        results_data = []
        console = Console()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(complete_style="green", finished_style="bold green"),
            TaskProgressColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
            console=console,
            transient=False
        ) as progress:

            task = progress.add_task(
                "[green]Indexing chunks...",
                total=len(chunks)
            )

            for chunk_idx, (chunk_text, token_count) in enumerate(chunks, 1):
                try:
                    # Update progress description
                    chunk_preview = chunk_text[:50].replace('\n', ' ')
                    progress.update(
                        task,
                        description=f"[green]Indexing[/green] │ [green]{successful} ✓[/green] [red]{failed} ✗[/red] │ Chunk {chunk_idx}/{len(chunks)}: {chunk_preview}..."
                    )

                    # Invoke FlowHunt flow
                    process_id = flowhunt_client.invoke_flow(
                        flow_id=index_flow_id,
                        human_input=chunk_text,
                        variables={
                            'chunk_index': chunk_idx,
                            'total_chunks': len(chunks),
                            'token_count': token_count,
                            'docx_filename': docx_file.name,
                            'source': 'docx'
                        },
                        singleton=False,
                    )

                    # Add to results data
                    results_data.append({
                        'chunk_index': chunk_idx,
                        'token_count': token_count,
                        'chunk_preview': chunk_text[:100].replace('\n', ' '),
                        'flow_process_id': process_id,
                        'status': 'success',
                        'indexed_at': datetime.now().isoformat()
                    })

                    successful += 1

                    if verbose:
                        console.print(f"[green]✓[/green] Indexed chunk {chunk_idx}/{len(chunks)} ({token_count} tokens, process: {process_id})")

                    # Rate limiting to be nice to the API
                    time.sleep(0.5)

                except Exception as e:
                    failed += 1

                    if verbose:
                        console.print(f"[red]✗[/red] Failed to index chunk {chunk_idx}/{len(chunks)}: {str(e)}")

                    # Still add to results with error status
                    results_data.append({
                        'chunk_index': chunk_idx,
                        'token_count': token_count,
                        'chunk_preview': chunk_text[:100].replace('\n', ' '),
                        'flow_process_id': f"ERROR: {str(e)}",
                        'status': 'failed',
                        'indexed_at': datetime.now().isoformat()
                    })

                progress.advance(task, 1)

        # Save results to CSV
        if results_data:
            logger.progress_start(f"Saving results to {output_csv}...")
            try:
                df = pd.DataFrame(results_data)
                df.to_csv(output_path, index=False)
                logger.progress_done(f"Results saved to {output_csv}")
            except Exception as e:
                logger.warning(f"Failed to save results: {str(e)}")

        duration = time.time() - start_time
        logger.progress_done("DOCX indexing completed", duration)

        # Display summary
        logger.stats_table("Indexing Results", {
            "Total chunks": len(chunks),
            "Successfully indexed": successful,
            "Failed": failed,
            "Results saved to": str(output_path)
        })

        logger.info(f"✅ DOCX indexing complete! Results saved to {output_path}")

    except KeyboardInterrupt:
        logger.error("Indexing interrupted by user")
        if results_data:
            df = pd.DataFrame(results_data)
            df.to_csv(output_path, index=False)
            logger.info(f"Partial results saved to {output_path}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Indexing failed: {str(e)}")
        if 'results_data' in locals() and results_data:
            df = pd.DataFrame(results_data)
            df.to_csv(output_path, index=False)
            logger.info(f"Partial results saved to {output_path}")
        sys.exit(1)


@index.command()
@click.argument('url', type=str)
@click.argument('index_flow_id', type=str)
@click.option('--max-tokens', type=int, default=4000, help='Maximum tokens per chunk (default: 4000)')
@click.option('--output-csv', type=click.Path(), help='Path to save processing results CSV')
@click.pass_context
def url(ctx, url, index_flow_id, max_tokens, output_csv):
    """Index a web page by processing text chunks through a FlowHunt flow.

    This command fetches content from a URL, converts it to markdown,
    chunks it based on token count, and processes each chunk through a FlowHunt flow.

    URL: The web page URL to process
    INDEX_FLOW_ID: FlowHunt flow ID to use for processing chunks
    """
    verbose = ctx.obj.get('verbose', False)
    logger = Logger(verbose=verbose)

    # Generate default CSV filename if not provided
    if not output_csv:
        current_date = datetime.now().strftime("%Y%m%d")
        # Create safe filename from URL
        from urllib.parse import urlparse
        parsed = urlparse(url)
        safe_name = parsed.netloc.replace('.', '_')
        output_csv = f"url_index_{current_date}_{safe_name}.csv"

    output_path = Path(output_csv)

    # Log command start
    config_args = {
        'url': url,
        'index_flow_id': index_flow_id,
        'max_tokens': max_tokens,
        'output_csv': str(output_path)
    }
    logger.command_start('index url', config_args)

    start_time = time.time()

    try:
        # Initialize FlowHunt client
        logger.progress_start("Initializing FlowHunt client...")
        try:
            flowhunt_client = FlowHuntClient.from_config_file()
            logger.progress_done("FlowHunt client initialized")
        except FileNotFoundError:
            logger.error("No FlowHunt configuration found. Please run 'flowhunt auth' first.")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Failed to initialize FlowHunt client: {str(e)}")
            sys.exit(1)

        # Initialize web processor
        logger.progress_start("Initializing web processor...")
        try:
            web_processor = WebProcessor(max_tokens=max_tokens)
            logger.progress_done("Web processor initialized")
        except Exception as e:
            logger.error(f"Failed to initialize web processor: {str(e)}")
            sys.exit(1)

        # Fetch and chunk URL content
        logger.progress_start(f"Fetching and processing URL: {url}...")
        try:
            chunks = web_processor.process_url(url, max_tokens)
            logger.progress_done(f"URL processed into {len(chunks)} chunks")
        except Exception as e:
            logger.error(f"Failed to process URL: {str(e)}")
            sys.exit(1)

        if not chunks:
            logger.info("No text chunks extracted from URL")
            return

        # Display chunk statistics
        total_tokens = sum(token_count for _, token_count in chunks)
        avg_tokens = total_tokens // len(chunks) if chunks else 0
        logger.stats_table("URL Processing Summary", {
            "Total chunks": len(chunks),
            "Total tokens": total_tokens,
            "Average tokens per chunk": avg_tokens,
            "Max tokens per chunk": max_tokens
        })

        # Process chunks through FlowHunt flow
        logger.info(f"Starting to index {len(chunks)} chunks through flow...")

        successful = 0
        failed = 0
        results_data = []
        console = Console()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(complete_style="green", finished_style="bold green"),
            TaskProgressColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
            console=console,
            transient=False
        ) as progress:

            task = progress.add_task(
                "[green]Indexing chunks...",
                total=len(chunks)
            )

            for chunk_idx, (chunk_text, token_count) in enumerate(chunks, 1):
                try:
                    # Update progress description
                    chunk_preview = chunk_text[:50].replace('\n', ' ')
                    progress.update(
                        task,
                        description=f"[green]Indexing[/green] │ [green]{successful} ✓[/green] [red]{failed} ✗[/red] │ Chunk {chunk_idx}/{len(chunks)}: {chunk_preview}..."
                    )

                    # Invoke FlowHunt flow
                    process_id = flowhunt_client.invoke_flow(
                        flow_id=index_flow_id,
                        human_input=chunk_text,
                        variables={
                            'chunk_index': chunk_idx,
                            'total_chunks': len(chunks),
                            'token_count': token_count,
                            'url': url,
                            'source': 'url'
                        },
                        singleton=False,
                    )

                    # Add to results data
                    results_data.append({
                        'chunk_index': chunk_idx,
                        'token_count': token_count,
                        'chunk_preview': chunk_text[:100].replace('\n', ' '),
                        'flow_process_id': process_id,
                        'status': 'success',
                        'indexed_at': datetime.now().isoformat()
                    })

                    successful += 1

                    if verbose:
                        console.print(f"[green]✓[/green] Indexed chunk {chunk_idx}/{len(chunks)} ({token_count} tokens, process: {process_id})")

                    # Rate limiting to be nice to the API
                    time.sleep(0.5)

                except Exception as e:
                    failed += 1

                    if verbose:
                        console.print(f"[red]✗[/red] Failed to index chunk {chunk_idx}/{len(chunks)}: {str(e)}")

                    # Still add to results with error status
                    results_data.append({
                        'chunk_index': chunk_idx,
                        'token_count': token_count,
                        'chunk_preview': chunk_text[:100].replace('\n', ' '),
                        'flow_process_id': f"ERROR: {str(e)}",
                        'status': 'failed',
                        'indexed_at': datetime.now().isoformat()
                    })

                progress.advance(task, 1)

        # Save results to CSV
        if results_data:
            logger.progress_start(f"Saving results to {output_csv}...")
            try:
                df = pd.DataFrame(results_data)
                df.to_csv(output_path, index=False)
                logger.progress_done(f"Results saved to {output_csv}")
            except Exception as e:
                logger.warning(f"Failed to save results: {str(e)}")

        duration = time.time() - start_time
        logger.progress_done("URL indexing completed", duration)

        # Display summary
        logger.stats_table("Indexing Results", {
            "Total chunks": len(chunks),
            "Successfully indexed": successful,
            "Failed": failed,
            "Results saved to": str(output_path)
        })

        logger.info(f"✅ URL indexing complete! Results saved to {output_path}")

    except KeyboardInterrupt:
        logger.error("Indexing interrupted by user")
        if results_data:
            df = pd.DataFrame(results_data)
            df.to_csv(output_path, index=False)
            logger.info(f"Partial results saved to {output_path}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Indexing failed: {str(e)}")
        if 'results_data' in locals() and results_data:
            df = pd.DataFrame(results_data)
            df.to_csv(output_path, index=False)
            logger.info(f"Partial results saved to {output_path}")
        sys.exit(1)


@index.command()
@click.argument('sitemap_url', type=str)
@click.argument('index_flow_id', type=str)
@click.option('--max-tokens', type=int, default=4000, help='Maximum tokens per chunk (default: 4000)')
@click.option('--limit', type=int, help='Maximum number of URLs to process from sitemap')
@click.option('--output-csv', type=click.Path(), help='Path to save processing results CSV')
@click.pass_context
def sitemap(ctx, sitemap_url, index_flow_id, max_tokens, limit, output_csv):
    """Index all URLs from a sitemap by processing through a FlowHunt flow.

    This command fetches all URLs from a sitemap.xml, processes each page's content,
    chunks it based on token count, and processes each chunk through a FlowHunt flow.

    SITEMAP_URL: URL of the sitemap.xml file
    INDEX_FLOW_ID: FlowHunt flow ID to use for processing chunks
    """
    verbose = ctx.obj.get('verbose', False)
    logger = Logger(verbose=verbose)

    # Generate default CSV filename if not provided
    if not output_csv:
        current_date = datetime.now().strftime("%Y%m%d")
        from urllib.parse import urlparse
        parsed = urlparse(sitemap_url)
        safe_name = parsed.netloc.replace('.', '_')
        output_csv = f"sitemap_index_{current_date}_{safe_name}.csv"

    output_path = Path(output_csv)

    # Log command start
    config_args = {
        'sitemap_url': sitemap_url,
        'index_flow_id': index_flow_id,
        'max_tokens': max_tokens,
        'limit': limit or 'No limit',
        'output_csv': str(output_path)
    }
    logger.command_start('index sitemap', config_args)

    start_time = time.time()

    try:
        # Initialize FlowHunt client
        logger.progress_start("Initializing FlowHunt client...")
        try:
            flowhunt_client = FlowHuntClient.from_config_file()
            logger.progress_done("FlowHunt client initialized")
        except FileNotFoundError:
            logger.error("No FlowHunt configuration found. Please run 'flowhunt auth' first.")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Failed to initialize FlowHunt client: {str(e)}")
            sys.exit(1)

        # Initialize web processor
        logger.progress_start("Initializing web processor...")
        try:
            web_processor = WebProcessor(max_tokens=max_tokens)
            logger.progress_done("Web processor initialized")
        except Exception as e:
            logger.error(f"Failed to initialize web processor: {str(e)}")
            sys.exit(1)

        # Parse sitemap
        logger.progress_start(f"Parsing sitemap: {sitemap_url}...")
        try:
            urls = web_processor.parse_sitemap(sitemap_url)
            if limit:
                urls = urls[:limit]
            logger.progress_done(f"Found {len(urls)} URLs in sitemap")
        except Exception as e:
            logger.error(f"Failed to parse sitemap: {str(e)}")
            sys.exit(1)

        if not urls:
            logger.info("No URLs found in sitemap")
            return

        logger.info(f"Starting to process {len(urls)} URLs from sitemap...")

        successful_urls = 0
        failed_urls = 0
        total_chunks = 0
        results_data = []
        console = Console()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(complete_style="green", finished_style="bold green"),
            TaskProgressColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
            console=console,
            transient=False
        ) as progress:

            task = progress.add_task(
                "[green]Processing URLs...",
                total=len(urls)
            )

            for url_idx, page_url in enumerate(urls, 1):
                try:
                    # Update progress
                    progress.update(
                        task,
                        description=f"[green]Processing[/green] │ [green]{successful_urls} ✓[/green] [red]{failed_urls} ✗[/red] │ URL {url_idx}/{len(urls)}: {page_url[:40]}..."
                    )

                    # Fetch and chunk URL content
                    chunks = web_processor.process_url(page_url, max_tokens)

                    # Process each chunk
                    for chunk_idx, (chunk_text, token_count) in enumerate(chunks, 1):
                        try:
                            # Invoke FlowHunt flow
                            process_id = flowhunt_client.invoke_flow(
                                flow_id=index_flow_id,
                                human_input=chunk_text,
                                variables={
                                    'url_index': url_idx,
                                    'total_urls': len(urls),
                                    'chunk_index': chunk_idx,
                                    'total_chunks': len(chunks),
                                    'token_count': token_count,
                                    'url': page_url,
                                    'source': 'sitemap'
                                },
                                singleton=False,
                            )

                            # Add to results data
                            results_data.append({
                                'url_index': url_idx,
                                'url': page_url,
                                'chunk_index': chunk_idx,
                                'token_count': token_count,
                                'chunk_preview': chunk_text[:100].replace('\n', ' '),
                                'flow_process_id': process_id,
                                'status': 'success',
                                'indexed_at': datetime.now().isoformat()
                            })

                            total_chunks += 1

                            if verbose:
                                console.print(f"[green]✓[/green] Indexed {page_url} chunk {chunk_idx}/{len(chunks)} ({token_count} tokens)")

                            # Rate limiting
                            time.sleep(0.5)

                        except Exception as e:
                            if verbose:
                                console.print(f"[red]✗[/red] Failed to index chunk {chunk_idx} from {page_url}: {str(e)}")

                            results_data.append({
                                'url_index': url_idx,
                                'url': page_url,
                                'chunk_index': chunk_idx,
                                'token_count': token_count,
                                'chunk_preview': chunk_text[:100].replace('\n', ' '),
                                'flow_process_id': f"ERROR: {str(e)}",
                                'status': 'failed',
                                'indexed_at': datetime.now().isoformat()
                            })

                    successful_urls += 1

                except Exception as e:
                    failed_urls += 1
                    if verbose:
                        console.print(f"[red]✗[/red] Failed to process URL {page_url}: {str(e)}")

                    results_data.append({
                        'url_index': url_idx,
                        'url': page_url,
                        'chunk_index': 0,
                        'token_count': 0,
                        'chunk_preview': '',
                        'flow_process_id': f"ERROR: {str(e)}",
                        'status': 'failed',
                        'indexed_at': datetime.now().isoformat()
                    })

                progress.advance(task, 1)

                # Save checkpoint after each URL
                if results_data:
                    df = pd.DataFrame(results_data)
                    df.to_csv(output_path, index=False)

        duration = time.time() - start_time
        logger.progress_done("Sitemap indexing completed", duration)

        # Display summary
        logger.stats_table("Indexing Results", {
            "Total URLs": len(urls),
            "Successfully processed URLs": successful_urls,
            "Failed URLs": failed_urls,
            "Total chunks indexed": total_chunks,
            "Results saved to": str(output_path)
        })

        logger.info(f"✅ Sitemap indexing complete! Results saved to {output_path}")

    except KeyboardInterrupt:
        logger.error("Indexing interrupted by user")
        if results_data:
            df = pd.DataFrame(results_data)
            df.to_csv(output_path, index=False)
            logger.info(f"Partial results saved to {output_path}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Indexing failed: {str(e)}")
        if 'results_data' in locals() and results_data:
            df = pd.DataFrame(results_data)
            df.to_csv(output_path, index=False)
            logger.info(f"Partial results saved to {output_path}")
        sys.exit(1)


@index.command()
@click.argument('base_url', type=str)
@click.argument('index_flow_id', type=str)
@click.argument('department_id', type=str, required=False)
@click.option('--api-key', required=True, help='LiveAgent API key (read-only for tickets)')
@click.option('--limit', type=int, default=100, help='Maximum number of tickets to index (default: 100)')
@click.option('--output-csv', type=click.Path(), help='Path to save checkpoint CSV (default: liveagent_index_YYYYMMDD.csv)')
@click.option('--resume', is_flag=True, help='Resume from existing checkpoint CSV')
@click.pass_context
def liveagent(ctx, base_url, index_flow_id, department_id, api_key, limit, output_csv, resume):
    """Index closed LiveAgent tickets into FlowHunt.
    
    This command fetches closed/resolved tickets from LiveAgent, formats them as text,
    and processes them through a FlowHunt flow for indexing. Only closed tickets are
    indexed to ensure complete conversation history.
    
    BASE_URL: LiveAgent instance URL (e.g., https://support.qualityunit.com)
    INDEX_FLOW_ID: FlowHunt flow ID to use for indexing
    DEPARTMENT_ID: Optional department ID to filter tickets (e.g., 31ivft8h)
    """
    verbose = ctx.obj.get('verbose', False)
    logger = Logger(verbose=verbose)
    
    # Generate default CSV filename if not provided
    if not output_csv:
        current_date = datetime.now().strftime("%Y%m%d")
        output_csv = f"liveagent_index_{current_date}.csv"
    
    output_path = Path(output_csv)
    
    # Log command start
    config_args = {
        'base_url': base_url,
        'index_flow_id': index_flow_id,
        'department_id': department_id or 'All departments',
        'limit': limit,
        'output_csv': str(output_path),
        'resume': resume
    }
    logger.command_start('index liveagent', config_args)
    
    start_time = time.time()
    
    try:
        # Initialize FlowHunt client
        logger.progress_start("Initializing FlowHunt client...")
        try:
            flowhunt_client = FlowHuntClient.from_config_file()
            logger.progress_done("FlowHunt client initialized")
        except FileNotFoundError:
            logger.error("No FlowHunt configuration found. Please run 'flowhunt auth' first.")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Failed to initialize FlowHunt client: {str(e)}")
            sys.exit(1)
        
        # Initialize LiveAgent client
        logger.progress_start("Initializing LiveAgent client...")
        try:
            liveagent_client = LiveAgentClient(base_url, api_key)
            logger.progress_done("LiveAgent client initialized")
        except Exception as e:
            logger.error(f"Failed to initialize LiveAgent client: {str(e)}")
            sys.exit(1)
        
        # Check for existing checkpoint
        indexed_tickets = set()
        checkpoint_data = []
        
        if resume and output_path.exists():
            logger.progress_start(f"Loading checkpoint from {output_csv}...")
            try:
                existing_df = pd.read_csv(output_path)
                indexed_tickets = set(existing_df['ticket_id'].astype(str))
                checkpoint_data = existing_df.to_dict('records')
                logger.progress_done(f"Loaded checkpoint with {len(indexed_tickets)} already indexed tickets")
            except Exception as e:
                logger.warning(f"Failed to load checkpoint: {str(e)}")
        
        # Fetch tickets from LiveAgent
        if resume and indexed_tickets:
            logger.progress_start(f"Fetching tickets from LiveAgent (limit: {limit}, skipping {len(indexed_tickets)} already indexed)...")
        else:
            logger.progress_start(f"Fetching tickets from LiveAgent (limit: {limit})...")

        try:
            tickets = liveagent_client.paginate_all_tickets(
                department_id=department_id,  # Pass department_id for filtering
                max_tickets=limit,
                skip_ids=indexed_tickets if resume else None  # Pass already indexed IDs to skip
            )
            logger.progress_done(f"Fetched {len(tickets)} new tickets from LiveAgent")
        except Exception as e:
            logger.error(f"Failed to fetch tickets: {str(e)}")
            sys.exit(1)
        
        if not tickets:
            logger.info("No new tickets to index")
            return
        
        # Process tickets
        logger.info(f"Starting to index {len(tickets)} tickets...")
        
        successful = 0
        failed = 0
        console = Console()
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(complete_style="green", finished_style="bold green"),
            TaskProgressColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
            console=console,
            transient=False
        ) as progress:
            
            task = progress.add_task(
                "[green]Indexing tickets...", 
                total=len(tickets)
            )
            
            for ticket in tickets:
                ticket_id = ticket.get('id', 'unknown')
                ticket_code = ticket.get('code', 'N/A')
                ticket_subject = ticket.get('subject', 'No subject')[:50]
                
                try:
                    # Update progress description
                    progress.update(
                        task, 
                        description=f"[green]Indexing[/green] │ [green]{successful} ✓[/green] [red]{failed} ✗[/red] │ #{ticket_code}: {ticket_subject}..."
                    )
                    
                    # Fetch full conversation
                    messages = liveagent_client.get_ticket_messages(ticket_id)
                    
                    # Format ticket as text
                    ticket_text = liveagent_client.format_ticket_as_text(ticket, messages)
                    
                    # Invoke FlowHunt flow
                    process_id = flowhunt_client.invoke_flow(
                        flow_id=index_flow_id,
                        human_input=ticket_text,
                        variables={
                            'ticket_id': ticket_id,
                            'ticket_code': ticket_code,
                            'department_id': ticket.get('department_id', ''),
                            'source': 'liveagent'
                        },
                        singleton=False,
                    )
                    
                    # Add to checkpoint data
                    checkpoint_entry = {
                        'ticket_id': ticket_id,
                        'ticket_code': ticket_code,
                        'ticket_subject': ticket.get('subject', ''),
                        'department_id': ticket.get('department_id', ''),
                        'department_name': ticket.get('department_name', ''),
                        'created_at': ticket.get('date_created', ''),
                        'status': ticket.get('status', ''),
                        'customer_email': ticket.get('customer_email', ''),
                        'flow_input_length': len(ticket_text),
                        'flow_process_id': process_id,
                        'indexed_at': datetime.now().isoformat()
                    }
                    checkpoint_data.append(checkpoint_entry)
                    
                    # Save checkpoint after each successful index
                    df = pd.DataFrame(checkpoint_data)
                    df.to_csv(output_path, index=False)
                    
                    successful += 1
                    
                    if verbose:
                        console.print(f"[green]✓[/green] Indexed ticket #{ticket_code} (process: {process_id})")
                    
                    # Rate limiting to be nice to both APIs
                    time.sleep(1)
                    
                except Exception as e:
                    failed += 1
                    
                    if verbose:
                        console.print(f"[red]✗[/red] Failed to index ticket #{ticket_code}: {str(e)}")
                    
                    # Still add to checkpoint with error status
                    checkpoint_entry = {
                        'ticket_id': ticket_id,
                        'ticket_code': ticket_code,
                        'ticket_subject': ticket.get('subject', ''),
                        'department_id': ticket.get('department_id', ''),
                        'department_name': ticket.get('department_name', ''),
                        'created_at': ticket.get('date_created', ''),
                        'status': ticket.get('status', ''),
                        'customer_email': ticket.get('customer_email', ''),
                        'flow_input_length': 0,
                        'flow_process_id': f"ERROR: {str(e)}",
                        'indexed_at': datetime.now().isoformat()
                    }
                    checkpoint_data.append(checkpoint_entry)
                    
                    # Save checkpoint even on errors
                    df = pd.DataFrame(checkpoint_data)
                    df.to_csv(output_path, index=False)
                
                progress.advance(task, 1)
        
        duration = time.time() - start_time
        logger.progress_done("Indexing completed", duration)
        
        # Display summary
        logger.stats_table("Indexing Results", {
            "Total tickets processed": len(tickets),
            "Successfully indexed": successful,
            "Failed": failed,
            "Checkpoint saved to": str(output_path)
        })
        
        logger.info(f"✅ LiveAgent indexing complete! Checkpoint saved to {output_path}")
        
    except KeyboardInterrupt:
        logger.error("Indexing interrupted by user")
        if checkpoint_data:
            df = pd.DataFrame(checkpoint_data)
            df.to_csv(output_path, index=False)
            logger.info(f"Partial checkpoint saved to {output_path}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Indexing failed: {str(e)}")
        if checkpoint_data:
            df = pd.DataFrame(checkpoint_data)
            df.to_csv(output_path, index=False)
            logger.info(f"Partial checkpoint saved to {output_path}")
        sys.exit(1)


@index.command()
@click.argument('folder_path', type=str)
@click.argument('index_flow_id', type=str)
@click.option('--max-tokens', type=int, default=4000, help='Maximum tokens per chunk (default: 4000)')
@click.option('--output-csv', type=str, help='Path to save processing results CSV')
@click.option('--sequential', is_flag=True, help='Process chunks sequentially, waiting for each to complete before starting the next')
@click.pass_context
def folder(ctx, folder_path, index_flow_id, max_tokens, output_csv, sequential):
    """Index all PDF and DOCX files in a folder through a FlowHunt flow.

    This command scans a folder for PDF and DOCX files, extracts text from each,
    chunks it based on token count, and processes each chunk through a FlowHunt flow.

    FOLDER_PATH: Path to the folder containing PDF and/or DOCX files
    INDEX_FLOW_ID: FlowHunt flow ID to use for processing chunks
    """
    verbose = ctx.obj.get('verbose', False)
    logger = Logger(verbose=verbose)

    folder = Path(folder_path)

    if not folder.exists():
        logger.error(f"Path does not exist: {folder}")
        sys.exit(1)

    if not folder.is_dir():
        logger.error(f"Path is not a directory: {folder}")
        sys.exit(1)

    # Generate default CSV filename if not provided
    if not output_csv:
        current_date = datetime.now().strftime("%Y%m%d")
        output_csv = f"folder_index_{current_date}_{folder.name}.csv"

    output_path = Path(output_csv)

    # Log command start
    config_args = {
        'folder_path': str(folder),
        'index_flow_id': index_flow_id,
        'max_tokens': max_tokens,
        'output_csv': str(output_path),
        'sequential': sequential
    }
    logger.command_start('index folder', config_args)

    start_time = time.time()

    try:
        # Initialize FlowHunt client
        logger.progress_start("Initializing FlowHunt client...")
        try:
            flowhunt_client = FlowHuntClient.from_config_file()
            logger.progress_done("FlowHunt client initialized")
        except FileNotFoundError:
            logger.error("No FlowHunt configuration found. Please run 'flowhunt auth' first.")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Failed to initialize FlowHunt client: {str(e)}")
            sys.exit(1)

        # Initialize processors
        logger.progress_start("Initializing document processors...")
        try:
            pdf_processor = PDFProcessor(max_tokens=max_tokens)
            docx_processor = DOCXProcessor(max_tokens=max_tokens)
            logger.progress_done("Document processors initialized")
        except Exception as e:
            logger.error(f"Failed to initialize processors: {str(e)}")
            sys.exit(1)

        # Find all PDF and DOCX files in the folder
        logger.progress_start(f"Scanning folder: {folder.name}...")

        # Use os.listdir to avoid potential glob issues with Python 3.13+
        import os
        all_entries = os.listdir(folder)
        pdf_files = [folder / f for f in all_entries if f.lower().endswith('.pdf')]
        docx_files = [folder / f for f in all_entries if f.lower().endswith('.docx')]

        # Filter out temporary Word files (starting with ~$)
        docx_files = [f for f in docx_files if not f.name.startswith("~$")]

        all_files = pdf_files + docx_files
        logger.progress_done(f"Found {len(pdf_files)} PDF files and {len(docx_files)} DOCX files")

        if not all_files:
            logger.info("No PDF or DOCX files found in folder")
            return

        # Process all files
        logger.info(f"Starting to process {len(all_files)} files...")

        successful_files = 0
        failed_files = 0
        total_chunks = 0
        results_data = []
        console = Console()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(complete_style="green", finished_style="bold green"),
            TaskProgressColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
            console=console,
            transient=False
        ) as progress:

            task = progress.add_task(
                "[green]Processing files...",
                total=len(all_files)
            )

            for file_idx, file_path in enumerate(all_files, 1):
                file_type = "PDF" if file_path.suffix.lower() == ".pdf" else "DOCX"

                try:
                    # Update progress
                    progress.update(
                        task,
                        description=f"[green]Processing[/green] │ [green]{successful_files} ✓[/green] [red]{failed_files} ✗[/red] │ {file_type} {file_idx}/{len(all_files)}: {file_path.name[:30]}..."
                    )

                    # Process file based on type
                    if file_path.suffix.lower() == ".pdf":
                        chunks = pdf_processor.process_pdf(file_path, max_tokens)
                    else:  # .docx
                        chunks = docx_processor.process_docx(file_path, max_tokens)

                    if verbose:
                        console.print(f"[blue]ℹ[/blue] {file_path.name}: {len(chunks)} chunks extracted")

                    # Process each chunk
                    for chunk_idx, (chunk_text, token_count) in enumerate(chunks, 1):
                        try:
                            # Invoke FlowHunt flow
                            process_id = flowhunt_client.invoke_flow(
                                flow_id=index_flow_id,
                                human_input=chunk_text,
                                variables={
                                    'file_index': file_idx,
                                    'total_files': len(all_files),
                                    'chunk_index': chunk_idx,
                                    'total_chunks': len(chunks),
                                    'token_count': token_count,
                                    'filename': file_path.name,
                                    'file_path': str(file_path),
                                    'file_type': file_type.lower(),
                                    'source': 'folder'
                                },
                                singleton=False,
                            )

                            # If sequential mode, poll until completion
                            if sequential:
                                if verbose:
                                    console.print(f"[blue]⏳[/blue] Waiting for {file_path.name} chunk {chunk_idx}/{len(chunks)} to complete (process: {process_id})...")

                                poll_interval = 2  # seconds
                                max_wait_time = 300  # 5 minutes timeout
                                elapsed_time = 0

                                while elapsed_time < max_wait_time:
                                    is_ready, result = flowhunt_client.get_flow_results(index_flow_id, process_id)

                                    if is_ready:
                                        if result and result != "NOCONTENT":
                                            # Success
                                            results_data.append({
                                                'file_index': file_idx,
                                                'filename': file_path.name,
                                                'file_type': file_type,
                                                'chunk_index': chunk_idx,
                                                'token_count': token_count,
                                                'chunk_preview': chunk_text[:100].replace('\n', ' '),
                                                'flow_process_id': process_id,
                                                'status': 'success',
                                                'result': result[:200] + ('...' if len(result) > 200 else ''),
                                                'indexed_at': datetime.now().isoformat()
                                            })
                                            total_chunks += 1

                                            if verbose:
                                                console.print(f"[green]✓[/green] {file_path.name} chunk {chunk_idx}/{len(chunks)} completed successfully ({token_count} tokens)")
                                        else:
                                            # Completed but no content
                                            results_data.append({
                                                'file_index': file_idx,
                                                'filename': file_path.name,
                                                'file_type': file_type,
                                                'chunk_index': chunk_idx,
                                                'token_count': token_count,
                                                'chunk_preview': chunk_text[:100].replace('\n', ' '),
                                                'flow_process_id': process_id,
                                                'status': 'failed',
                                                'error': 'No content returned',
                                                'indexed_at': datetime.now().isoformat()
                                            })

                                            if verbose:
                                                console.print(f"[red]✗[/red] {file_path.name} chunk {chunk_idx}/{len(chunks)} returned no content")

                                        break

                                    # Wait before next poll
                                    time.sleep(poll_interval)
                                    elapsed_time += poll_interval

                                # Check if timed out
                                if elapsed_time >= max_wait_time:
                                    results_data.append({
                                        'file_index': file_idx,
                                        'filename': file_path.name,
                                        'file_type': file_type,
                                        'chunk_index': chunk_idx,
                                        'token_count': token_count,
                                        'chunk_preview': chunk_text[:100].replace('\n', ' '),
                                        'flow_process_id': process_id,
                                        'status': 'failed',
                                        'error': 'Timeout waiting for completion',
                                        'indexed_at': datetime.now().isoformat()
                                    })

                                    if verbose:
                                        console.print(f"[red]✗[/red] {file_path.name} chunk {chunk_idx}/{len(chunks)} timed out")

                            else:
                                # Non-sequential mode - just add to results data immediately
                                results_data.append({
                                    'file_index': file_idx,
                                    'filename': file_path.name,
                                    'file_type': file_type,
                                    'chunk_index': chunk_idx,
                                    'token_count': token_count,
                                    'chunk_preview': chunk_text[:100].replace('\n', ' '),
                                    'flow_process_id': process_id,
                                    'status': 'success',
                                    'indexed_at': datetime.now().isoformat()
                                })

                                total_chunks += 1

                                if verbose:
                                    console.print(f"[green]✓[/green] Indexed {file_path.name} chunk {chunk_idx}/{len(chunks)} ({token_count} tokens)")

                                # Rate limiting
                                time.sleep(0.5)

                        except Exception as e:
                            if verbose:
                                console.print(f"[red]✗[/red] Failed to index chunk {chunk_idx} from {file_path.name}: {str(e)}")

                            results_data.append({
                                'file_index': file_idx,
                                'filename': file_path.name,
                                'file_type': file_type,
                                'chunk_index': chunk_idx,
                                'token_count': token_count,
                                'chunk_preview': chunk_text[:100].replace('\n', ' '),
                                'flow_process_id': f"ERROR: {str(e)}",
                                'status': 'failed',
                                'indexed_at': datetime.now().isoformat()
                            })

                    successful_files += 1

                except Exception as e:
                    failed_files += 1
                    if verbose:
                        console.print(f"[red]✗[/red] Failed to process file {file_path.name}: {str(e)}")

                    results_data.append({
                        'file_index': file_idx,
                        'filename': file_path.name,
                        'file_type': file_type,
                        'chunk_index': 0,
                        'token_count': 0,
                        'chunk_preview': '',
                        'flow_process_id': f"ERROR: {str(e)}",
                        'status': 'failed',
                        'indexed_at': datetime.now().isoformat()
                    })

                progress.advance(task, 1)

                # Save checkpoint after each file
                if results_data:
                    df = pd.DataFrame(results_data)
                    df.to_csv(output_path, index=False)

        duration = time.time() - start_time
        logger.progress_done("Folder indexing completed", duration)

        # Display summary
        logger.stats_table("Indexing Results", {
            "Total files": len(all_files),
            "PDF files": len(pdf_files),
            "DOCX files": len(docx_files),
            "Successfully processed files": successful_files,
            "Failed files": failed_files,
            "Total chunks indexed": total_chunks,
            "Results saved to": str(output_path)
        })

        logger.info(f"✅ Folder indexing complete! Results saved to {output_path}")

    except KeyboardInterrupt:
        logger.error("Indexing interrupted by user")
        if results_data:
            df = pd.DataFrame(results_data)
            df.to_csv(output_path, index=False)
            logger.info(f"Partial results saved to {output_path}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Indexing failed: {str(e)}")
        if 'results_data' in locals() and results_data:
            df = pd.DataFrame(results_data)
            df.to_csv(output_path, index=False)
            logger.info(f"Partial results saved to {output_path}")
        sys.exit(1)


if __name__ == '__main__':
    main()
