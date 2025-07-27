"""FlowHunt API client wrapper."""

from typing import Optional, Dict, Any, List, Tuple
import os
import json
import time
import csv
from pathlib import Path
from tqdm import tqdm


class FlowHuntClient:
    """Wrapper for FlowHunt API client using official FlowHunt SDK."""
    
    def __init__(self, api_key: Optional[str] = None, base_url: str = "https://api.flowhunt.io"):
        """Initialize FlowHunt client.
        
        Args:
            api_key: FlowHunt API key. If not provided, will look for FLOWHUNT_API_KEY env var.
            base_url: FlowHunt API base URL.
        """
        self.api_key = api_key or os.getenv('FLOWHUNT_API_KEY')
        self.base_url = base_url
        self._workspace_id = None
        
        if not self.api_key:
            raise ValueError("FlowHunt API key is required. Set FLOWHUNT_API_KEY environment variable or pass api_key parameter.")
        
        # Initialize FlowHunt SDK client
        try:
            import flowhunt
            self.flowhunt = flowhunt
            
            self.configuration = flowhunt.Configuration(
                host=self.base_url
            )
            self.configuration.api_key['APIKeyHeader'] = self.api_key
            
        except ImportError:
            raise ImportError("FlowHunt SDK is required. Install with: pip install flowhunt")
        except Exception as e:
            raise Exception(f"Failed to initialize FlowHunt SDK: {e}")
    
    def get_workspace_id(self) -> str:
        """Get the workspace ID from FlowHunt API.
        
        Returns:
            Workspace ID
        """
        if self._workspace_id:
            return self._workspace_id
            
        with self.flowhunt.ApiClient(self.configuration) as api_client:
            api_instance = self.flowhunt.WebAuthApi(api_client)
            try:
                api_response = api_instance.get_user()
                self._workspace_id = api_response.api_key_workspace_id
                return self._workspace_id
            except self.flowhunt.ApiException as e:
                raise Exception(f"Failed to get workspace ID: {e}")
    
    def invoke_flow(self, flow_id: str, variables: Dict[str, Any] = None, human_input: str = "") -> str:
        """Invoke a FlowHunt flow and return the process ID.
        
        Args:
            flow_id: The FlowHunt flow ID to invoke
            variables: Variables to pass to the flow
            human_input: Human input text for the flow
            
        Returns:
            Process ID for tracking the flow execution
        """
        workspace_id = self.get_workspace_id()
        
        with self.flowhunt.ApiClient(self.configuration) as api_client:
            api_instance = self.flowhunt.FlowsApi(api_client)
            
            try:
                flow_invoke_request = self.flowhunt.FlowInvokeRequest(
                    variables=variables or {},
                    human_input=human_input
                )
                
                response = api_instance.invoke_flow_singleton(
                    flow_id=flow_id,
                    workspace_id=workspace_id,
                    flow_invoke_request=flow_invoke_request
                )
                
                return response.id
                
            except self.flowhunt.ApiException as e:
                raise Exception(f"Failed to invoke flow {flow_id}: {e}")
    
    def get_flow_results(self, flow_id: str, process_id: str) -> Tuple[bool, Optional[str]]:
        """Check if a flow has completed and get the results.
        
        Args:
            flow_id: The FlowHunt flow ID
            process_id: Process ID to check
            
        Returns:
            Tuple of (is_ready, result_text)
        """
        workspace_id = self.get_workspace_id()
        
        with self.flowhunt.ApiClient(self.configuration) as api_client:
            api_instance = self.flowhunt.FlowsApi(api_client)
            
            try:
                response = api_instance.get_invoked_flow_results(
                    flow_id=flow_id,
                    task_id=process_id,
                    workspace_id=workspace_id
                )
                
                if response.status == "SUCCESS":
                    generated_content = json.loads(response.result)
                    content = ""
                    for output in generated_content['outputs'][0]['outputs']:
                        part = output['results']['message']['result'].strip()
                        if part.startswith("```"):
                            part = "\n".join(part.splitlines()[1:]).strip()
                        if part.endswith("```"):
                            part = part[:-3].strip()
                        content += part + "\n"
                    content = content.strip()
                    
                    if not content:
                        content = "NOCONTENT"
                    
                    return True, content.strip()
                else:
                    return False, None
                    
            except self.flowhunt.ApiException as e:
                raise Exception(f"Failed to get flow results for process {process_id}: {e}")
    
    def execute_flow(self, flow_id: str, variables: Dict[str, Any] = None, human_input: str = "", timeout: int = 300) -> str:
        """Execute a FlowHunt flow synchronously and return the result.
        
        Args:
            flow_id: The FlowHunt flow ID to execute
            variables: Variables to pass to the flow
            human_input: Human input text for the flow
            timeout: Maximum time to wait for completion in seconds
            
        Returns:
            Flow execution result
        """
        process_id = self.invoke_flow(flow_id, variables, human_input)
        
        start_time = time.time()
        check_interval = 5
        
        while time.time() - start_time < timeout:
            is_ready, result = self.get_flow_results(flow_id, process_id)
            if is_ready:
                return result
            time.sleep(check_interval)
        
        raise TimeoutError(f"Flow execution timed out after {timeout} seconds")
    
    def get_flow_info(self, flow_id: str) -> Dict[str, Any]:
        """Get information about a FlowHunt flow.
        
        Args:
            flow_id: The FlowHunt flow ID
            
        Returns:
            Flow metadata and configuration
        """
        workspace_id = self.get_workspace_id()
        
        with self.flowhunt.ApiClient(self.configuration) as api_client:
            api_instance = self.flowhunt.FlowsApi(api_client)
            
            try:
                response = api_instance.get(
                    flow_id=flow_id,
                    workspace_id=workspace_id
                )
                return response.to_dict()
                
            except self.flowhunt.ApiException as e:
                raise Exception(f"Failed to get flow info for {flow_id}: {e}")
    
    def list_flows(self) -> List[Dict[str, Any]]:
        """List available FlowHunt flows.
        
        Returns:
            List of flow metadata
        """
        workspace_id = self.get_workspace_id()
        
        with self.flowhunt.ApiClient(self.configuration) as api_client:
            api_instance = self.flowhunt.FlowsApi(api_client)
            
            try:
                response = api_instance.search(
                    workspace_id=workspace_id,
                    flow_search_request=self.flowhunt.FlowSearchRequest()
                )
                return [flow.to_dict() for flow in response]
                
            except self.flowhunt.ApiException as e:
                raise Exception(f"Failed to list flows: {e}")
    
    def batch_execute_from_csv(self, csv_file: str, flow_id: str, output_dir: str, 
                              overwrite: bool = False, check_interval: int = 10) -> Dict[str, int]:
        """Execute a flow for each row in a CSV file.
        
        Args:
            csv_file: Path to CSV file with 'flow_input' and 'filename' columns
            flow_id: FlowHunt flow ID to execute
            output_dir: Directory to save output files
            overwrite: Whether to overwrite existing files
            check_interval: Seconds between result checks
            
        Returns:
            Dictionary with execution statistics
        """
        # Read CSV file
        topics = self._read_csv_topics(csv_file)
        
        if not topics:
            return {"completed": 0, "failed": 0, "skipped": 0, "total": 0}
        
        # Filter existing files if not overwriting
        skipped_count = 0
        if not overwrite:
            topics, skipped_count = self._filter_existing_files(topics, output_dir)
        
        if not topics:
            return {"completed": 0, "failed": 0, "skipped": skipped_count, "total": skipped_count}
        
        # Execute batch processing
        return self._process_batch_topics(topics, flow_id, output_dir, check_interval, skipped_count)
    
    def _read_csv_topics(self, csv_file: str) -> List[Dict[str, str]]:
        """Read topics from CSV file."""
        topics = []
        
        # Detect delimiter
        delimiter = self._detect_csv_delimiter(csv_file)
        
        try:
            csv.field_size_limit(1000000)  # 1MB limit
            
            with open(csv_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f, delimiter=delimiter)
                
                if 'flow_input' not in reader.fieldnames or 'filename' not in reader.fieldnames:
                    raise ValueError("CSV file must contain 'flow_input' and 'filename' columns")
                
                for row in reader:
                    flow_input = row['flow_input'].strip()
                    filename = row['filename'].strip()
                    
                    if flow_input and filename:
                        topics.append({
                            'flow_input': flow_input,
                            'filename': filename
                        })
        
        except Exception as e:
            raise Exception(f"Error reading CSV file: {e}")
        
        return topics
    
    def _detect_csv_delimiter(self, file_path: str) -> str:
        """Detect CSV delimiter."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                first_line = f.readline().strip()
                
                delimiter_counts = {
                    ',': first_line.count(','),
                    ';': first_line.count(';'),
                    '\t': first_line.count('\t')
                }
                
                max_delimiter = max(delimiter_counts, key=delimiter_counts.get)
                return max_delimiter if delimiter_counts[max_delimiter] > 0 else ';'
                
        except Exception:
            return ';'
    
    def _filter_existing_files(self, topics: List[Dict[str, str]], output_dir: str) -> Tuple[List[Dict[str, str]], int]:
        """Filter out topics with existing output files."""
        filtered_topics = []
        skipped_count = 0
        
        for topic in topics:
            output_path = Path(output_dir) / topic['filename']
            if output_path.exists():
                skipped_count += 1
            else:
                filtered_topics.append(topic)
        
        return filtered_topics, skipped_count
    
    def _process_batch_topics(self, topics: List[Dict[str, str]], flow_id: str, 
                             output_dir: str, check_interval: int, skipped_count: int) -> Dict[str, int]:
        """Process topics in batch."""
        workspace_id = self.get_workspace_id()
        
        with self.flowhunt.ApiClient(self.configuration) as api_client:
            api_instance = self.flowhunt.FlowsApi(api_client)
            
            # Track tasks
            pending_tasks = {}
            completed_tasks = []
            failed_tasks = []
            
            # Schedule all tasks
            print(f"Scheduling {len(topics)} tasks...")
            progress_bar = tqdm(total=len(topics), desc="Scheduling")
            
            for topic in topics:
                try:
                    flow_invoke_request = self.flowhunt.FlowInvokeRequest(
                        variables={
                            "today": time.strftime("%Y-%m-%d"),
                            "v": "1",
                            "filename": topic['filename'],
                        },
                        human_input=topic['flow_input']
                    )
                    
                    response = api_instance.invoke_flow_singleton(
                        flow_id=flow_id,
                        workspace_id=workspace_id,
                        flow_invoke_request=flow_invoke_request
                    )
                    
                    pending_tasks[response.id] = topic
                    
                except Exception as e:
                    print(f"Failed to schedule task for '{topic['flow_input']}': {e}")
                    failed_tasks.append(topic)
                
                progress_bar.update(1)
            
            progress_bar.close()
            
            # Process results
            print(f"Processing {len(pending_tasks)} tasks...")
            progress_bar = tqdm(total=len(pending_tasks), desc="Processing")
            
            while pending_tasks:
                time.sleep(check_interval)
                process_ids = list(pending_tasks.keys())
                completed_in_batch = 0
                
                for process_id in process_ids:
                    topic_data = pending_tasks[process_id]
                    
                    try:
                        is_ready, content = self.get_flow_results(flow_id, process_id)
                        
                        if is_ready:
                            del pending_tasks[process_id]
                            completed_in_batch += 1
                            
                            if content and content != "NOCONTENT":
                                self._save_content(content, topic_data['filename'], output_dir)
                                completed_tasks.append(topic_data)
                            else:
                                failed_tasks.append(topic_data)
                                
                    except Exception as e:
                        print(f"Error processing task {process_id}: {e}")
                        del pending_tasks[process_id]
                        failed_tasks.append(topic_data)
                        completed_in_batch += 1
                
                progress_bar.update(completed_in_batch)
            
            progress_bar.close()
        
        return {
            "completed": len(completed_tasks),
            "failed": len(failed_tasks),
            "skipped": skipped_count,
            "total": len(topics) + skipped_count
        }
    
    def _save_content(self, content: str, filename: str, output_dir: str) -> None:
        """Save content to file."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        file_path = output_path / filename
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
    
    @classmethod
    def from_config_file(cls, config_path: Optional[Path] = None) -> 'FlowHuntClient':
        """Create client from configuration file.
        
        Args:
            config_path: Path to configuration file. If not provided, will look in standard locations.
            
        Returns:
            Configured FlowHunt client
        """
        if config_path is None:
            # Look in standard locations
            possible_paths = [
                Path.home() / '.flowhunt' / 'config.json',
                Path.cwd() / '.flowhunt.json',
                Path.home() / '.config' / 'flowhunt' / 'config.json'
            ]
            
            for path in possible_paths:
                if path.exists():
                    config_path = path
                    break
            else:
                raise FileNotFoundError("No FlowHunt configuration file found. Please run 'flowhunt auth' first.")
        
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            api_key = config.get('api_key')
            base_url = config.get('base_url', 'https://api.flowhunt.io')
            
            if not api_key:
                raise ValueError("No API key found in configuration file")
            
            return cls(api_key=api_key, base_url=base_url)
            
        except (json.JSONDecodeError, KeyError) as e:
            raise ValueError(f"Invalid configuration file format: {str(e)}")
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    def save_config(self, config_path: Optional[Path] = None) -> None:
        """Save current configuration to file.
        
        Args:
            config_path: Path to save configuration. If not provided, uses default location.
        """
        if config_path is None:
            config_path = Path.home() / '.flowhunt' / 'config.json'
        
        # Create directory if it doesn't exist
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        config = {
            'api_key': self.api_key,
            'base_url': self.base_url
        }
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        # Set secure permissions (readable only by owner)
        config_path.chmod(0o600)
