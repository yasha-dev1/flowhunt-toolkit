#!/usr/bin/env python3
"""Main CLI module for Flowhunt Toolkit."""

import click
import sys
from typing import Optional
from pathlib import Path

from . import __version__
from .core.client import FlowHuntClient
from .core.evaluator import FlowEvaluator


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
@click.option('--output', '-o', type=click.Path(), help='Output file for evaluation results')
@click.option('--batch-size', type=int, default=10, help='Batch size for processing (default: 10)')
@click.pass_context
def evaluate(ctx, csv_file, flow_id, judge_flow_id, output, batch_size):
    """Evaluate a flow using LLM as a judge.
    
    This command takes a CSV file with 'question' and 'expected_answer' columns
    and evaluates the specified flow's performance using an LLM judge.
    
    CSV_FILE: Path to CSV file with question and expected_answer columns
    FLOW_ID: The FlowHunt flow ID to evaluate
    """
    verbose = ctx.obj.get('verbose', False)
    
    if verbose:
        click.echo(f"Evaluating flow {flow_id} with CSV: {csv_file}")
        click.echo(f"Judge flow ID: {judge_flow_id or 'default public flow'}")
        click.echo(f"Batch size: {batch_size}")
        if output:
            click.echo(f"Output file: {output}")
    
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
                batch_results = evaluator._evaluate_batch(flow_id, batch)
                results.extend(batch_results)
                bar.update(len(batch))
        
        # Calculate summary statistics
        summary = evaluator.calculate_summary_stats(results)
        
        # Display results
        click.echo("\nEvaluation completed!")
        click.echo(f"Total questions: {summary.get('total_questions', len(results))}")
        
        # Calculate actual summary stats
        scores = [r.get('judge_score', 0) for r in results if isinstance(r.get('judge_score'), (int, float))]
        if scores:
            avg_score = sum(scores) / len(scores)
            click.echo(f"Average score: {avg_score:.2f}/10")
            pass_rate = len([s for s in scores if s >= 7]) / len(scores) * 100
            click.echo(f"Pass rate (≥7): {pass_rate:.1f}%")
        
        # Save results if output file specified
        if output:
            try:
                evaluator.save_results(results, Path(output))
                click.echo(f"Results saved to {output}")
            except Exception as e:
                click.echo(f"Warning: Failed to save results: {str(e)}", err=True)
        
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
@click.option('--check-interval', type=int, default=10, help='Seconds between result checks (default: 10)')
@click.option('--max-parallel', type=int, default=50, help='Maximum number of tasks to schedule in parallel (default: 50)')
@click.pass_context
def batch_run(ctx, input_file, flow_id, output_dir, output_file, format, overwrite, check_interval, max_parallel):
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
    
    # Validate that output_dir and output_file are not both specified
    if output_dir and output_file:
        click.echo("Error: --output-dir and --output-file cannot be used together. Choose one.", err=True)
        sys.exit(1)
    
    if verbose:
        click.echo(f"Running batch execution for flow {flow_id}")
        click.echo(f"Input file: {input_file}")
        if output_dir:
            click.echo(f"Output directory: {output_dir}")
        if output_file:
            click.echo(f"Output file: {output_file}")

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
        
        input_path = Path(input_file)
        
        # Check if this is a CSV file with the expected format for batch processing
        if input_path.suffix.lower() == '.csv':
            try:
                # Validate CSV format and columns
                import csv
                with open(input_path, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    fieldnames = reader.fieldnames
                    
                    # Check if CSV has headers
                    if not fieldnames:
                        click.echo("Error: CSV file must have headers.", err=True)
                        sys.exit(1)
                    
                    # Define allowed columns
                    mandatory_columns = {'flow_input'}
                    optional_columns = {'filename', 'flow_variable'}
                    allowed_columns = mandatory_columns | optional_columns
                    
                    # Check for mandatory columns
                    missing_mandatory = mandatory_columns - set(fieldnames)
                    if missing_mandatory:
                        click.echo(f"Error: CSV file must contain the following mandatory column(s): {', '.join(missing_mandatory)}", err=True)
                        sys.exit(1)
                    
                    # Check for invalid columns
                    invalid_columns = set(fieldnames) - allowed_columns
                    if invalid_columns:
                        click.echo(f"Error: CSV file contains invalid column(s): {', '.join(invalid_columns)}", err=True)
                        click.echo(f"Allowed columns are: {', '.join(sorted(allowed_columns))}", err=True)
                        sys.exit(1)
                    
                    # Check if filename column exists and output_dir is required
                    has_filename = 'filename' in fieldnames
                    has_flow_variable = 'flow_variable' in fieldnames
                    
                    if has_filename and not output_dir:
                        click.echo("Error: --output-dir is required when CSV file contains 'filename' column.", err=True)
                        sys.exit(1)
                    
                    # Check if output_dir is specified without filename column
                    if not has_filename and output_dir:
                        click.echo("Error: --output-dir only makes sense when CSV file contains 'filename' column.", err=True)
                        sys.exit(1)
                
                # If we reach here, CSV validation passed
                # Use optimized batch processing if we have filename or flow_variable columns
                if has_filename or has_flow_variable:
                    if has_filename:
                        click.echo("Using optimized batch processing for CSV with flow_input/filename columns...")
                    else:
                        click.echo("Using optimized batch processing for CSV with flow_input/flow_variable columns...")
                    
                    stats = client.batch_execute_from_csv(
                        csv_file=str(input_path),
                        flow_id=flow_id,
                        output_dir=output_dir,
                        overwrite=overwrite,
                        check_interval=check_interval,
                        max_parallel=max_parallel
                    )
                    
                    click.echo(f"\nBatch execution completed!")
                    click.echo(f"Total inputs: {stats['total']}")
                    click.echo(f"Completed successfully: {stats['completed']}")
                    click.echo(f"Failed: {stats['failed']}")
                    click.echo(f"Skipped (files already exist): {stats['skipped']}")
                    
                    return
                else:
                    # CSV has only flow_input - fall through to standard processing
                    click.echo("Processing CSV with flow_input column using standard method...")
                    
            except Exception as e:
                if "CSV file must have headers" in str(e) or "CSV file must contain" in str(e) or "CSV file contains invalid" in str(e) or "--output-dir is required" in str(e):
                    # Re-raise validation errors
                    raise
                if verbose:
                    click.echo(f"Could not use optimized batch processing: {e}")
                click.echo("Falling back to standard processing...")
        
        # Standard processing for other formats or CSV without expected columns
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
                click.echo(f"Error: Unsupported file format. Use .csv, .json, or .txt files.", err=True)
                sys.exit(1)
                
            click.echo(f"Loaded {len(inputs_list)} inputs from {input_file}")
            
        except Exception as e:
            click.echo(f"Error: Failed to load input file: {str(e)}", err=True)
            sys.exit(1)
        
        # Generate automatic output file if not specified
        if not output_file:
            from datetime import datetime
            current_date = datetime.now().strftime("%Y-%m-%d")
            output_file = f"{current_date}-{flow_id}-output.{format}"
            click.echo(f"No output file specified. Results will be saved to: {output_file}")
        
        # Execute flows sequentially
        results = []
        click.echo("Starting batch execution...")
        
        with click.progressbar(inputs_list, label='Processing') as bar:
            for i, inputs in enumerate(bar):
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
                    
                    result = client.execute_flow(flow_id, variables=variables, human_input=human_input)
                    results.append({
                        'input_index': i,
                        'flow_input': human_input,
                        'variables': variables,
                        'result': result,
                        'status': 'success'
                    })
                except Exception as e:
                    results.append({
                        'input_index': i,
                        'flow_input': human_input if 'human_input' in locals() else str(inputs),
                        'variables': variables if 'variables' in locals() else {},
                        'result': None,
                        'status': 'error',
                        'error': str(e)
                    })
        
        # Display summary
        successful = len([r for r in results if r['status'] == 'success'])
        failed = len(results) - successful
        
        click.echo(f"\nBatch execution completed!")
        click.echo(f"Total inputs: {len(results)}")
        click.echo(f"Successful: {successful}")
        click.echo(f"Failed: {failed}")
        
        # Save results if output file specified or auto-generated
        if output_file:
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
                
                click.echo(f"Results saved to {output_file}")
            except Exception as e:
                click.echo(f"Warning: Failed to save results: {str(e)}", err=True)

    except KeyboardInterrupt:
        click.echo("\nBatch execution interrupted by user.", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
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


if __name__ == '__main__':
    main()
