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
@click.pass_context
def evaluate(ctx, csv_file, flow_id, judge_flow_id, output_dir, batch_size):
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
        
        # Run evaluation
        click.echo("Starting evaluation...")
        with click.progressbar(length=len(evaluation_data), label='Evaluating') as bar:
            results = []
            for i in range(0, len(evaluation_data), batch_size):
                batch = evaluation_data.iloc[i:i+batch_size]
                batch_results = evaluator.evaluate_batch(flow_id, batch)
                results.extend(batch_results)
                bar.update(len(batch))
        
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
@click.pass_context
def auth(ctx):
    """Authenticate with FlowHunt API.
    
    This command will prompt for API credentials and store them securely.
    """
    verbose = ctx.obj.get('verbose', False)
    
    if verbose:
        click.echo("Setting up FlowHunt API authentication")
    
    # Prompt for API key
    api_key = click.prompt("Enter your FlowHunt API key", hide_input=True, type=str)

    try:
        # Test the credentials by creating a client and making a simple API call
        client = FlowHuntClient(api_key=api_key, base_url="https://api.flowhunt.io")
        
        click.echo("Testing API connection...")
        # Try to list flows to verify the credentials work
        flows = client.list_flows()
        
        # Save configuration
        client.save_config()
        
        click.echo("✓ Authentication successful!")
        click.echo(f"✓ Configuration saved to {Path.home() / '.flowhunt' / 'config.json'}")
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


if __name__ == '__main__':
    main()
