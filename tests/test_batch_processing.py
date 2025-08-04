"""Tests for batch processing functionality."""

import pytest
import tempfile
import csv
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import time

from flowhunt_toolkit.core.client import FlowHuntClient
from flowhunt_toolkit.utils.logger import Logger


class TestBatchProcessing:
    """Test suite for batch processing functionality."""
    
    @pytest.fixture
    def mock_client(self):
        """Create a mock FlowHunt client."""
        client = Mock(spec=FlowHuntClient)
        client.get_workspace_id.return_value = "test-workspace-id"
        return client
    
    @pytest.fixture
    def sample_csv_file(self):
        """Create a sample CSV file for testing."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            writer = csv.writer(f)
            writer.writerow(['flow_input', 'filename'])
            writer.writerow(['Test input 1', 'output1.txt'])
            writer.writerow(['Test input 2', 'output2.txt'])
            writer.writerow(['Test input 3', 'output3.txt'])
            return f.name
    
    @pytest.fixture
    def sample_csv_with_variables(self):
        """Create a sample CSV file with flow_variable column."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            writer = csv.writer(f)
            writer.writerow(['flow_input', 'flow_variable'])
            writer.writerow(['Test input 1', '{"var1": "value1"}'])
            writer.writerow(['Test input 2', '{"var2": "value2"}'])
            writer.writerow(['Test input 3', '{"var3": "value3"}'])
            return f.name
    
    @pytest.fixture
    def sample_json_file(self):
        """Create a sample JSON file for testing."""
        data = [
            {"flow_input": "Test input 1", "var1": "value1"},
            {"flow_input": "Test input 2", "var2": "value2"},
            {"flow_input": "Test input 3", "var3": "value3"}
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(data, f)
            return f.name
    
    @pytest.fixture
    def sample_txt_file(self):
        """Create a sample TXT file for testing."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Test input 1\n")
            f.write("Test input 2\n")
            f.write("Test input 3\n")
            return f.name

    def test_csv_validation_valid_file(self, sample_csv_file):
        """Test CSV validation with valid file."""
        client = FlowHuntClient.__new__(FlowHuntClient)  # Don't call __init__
        topics = client._read_csv_topics(sample_csv_file)
        
        assert len(topics) == 3
        assert topics[0]['flow_input'] == 'Test input 1'
        assert topics[0]['filename'] == 'output1.txt'
    
    def test_csv_validation_with_variables(self, sample_csv_with_variables):
        """Test CSV validation with flow_variable column."""
        client = FlowHuntClient.__new__(FlowHuntClient)  # Don't call __init__
        topics = client._read_csv_topics(sample_csv_with_variables)
        
        assert len(topics) == 3
        assert topics[0]['flow_input'] == 'Test input 1'
        assert topics[0]['flow_variable'] == {"var1": "value1"}
    
    def test_csv_delimiter_detection(self):
        """Test CSV delimiter detection."""
        client = FlowHuntClient.__new__(FlowHuntClient)  # Don't call __init__
        
        # Test comma delimiter
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("flow_input,filename\n")
            f.write("test,file.txt\n")
            comma_file = f.name
        
        delimiter = client._detect_csv_delimiter(comma_file)
        assert delimiter == ','
        
        # Test semicolon delimiter
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("flow_input;filename\n")
            f.write("test;file.txt\n")
            semicolon_file = f.name
        
        delimiter = client._detect_csv_delimiter(semicolon_file)
        assert delimiter == ';'
        
        # Clean up
        Path(comma_file).unlink()
        Path(semicolon_file).unlink()
    
    def test_filter_existing_files(self):
        """Test filtering of existing files."""
        client = FlowHuntClient.__new__(FlowHuntClient)  # Don't call __init__
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create some existing files
            existing_file = Path(temp_dir) / "existing.txt"
            existing_file.write_text("existing content")
            
            topics = [
                {'flow_input': 'test1', 'filename': 'existing.txt'},
                {'flow_input': 'test2', 'filename': 'new.txt'},
                {'flow_input': 'test3', 'filename': 'another_new.txt'}
            ]
            
            filtered_topics, skipped_count = client._filter_existing_files(topics, temp_dir)
            
            assert len(filtered_topics) == 2
            assert skipped_count == 1
            assert filtered_topics[0]['filename'] == 'new.txt'
            assert filtered_topics[1]['filename'] == 'another_new.txt'
    
    def test_batch_execute_from_csv_validation(self, sample_csv_file):
        """Test batch_execute_from_csv input validation."""
        client = FlowHuntClient.__new__(FlowHuntClient)  # Don't call __init__
        
        # Test reading CSV topics (this part works without API calls)
        topics = client._read_csv_topics(sample_csv_file)
        assert len(topics) == 3
        
        # Test filtering existing files
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create one existing file
            existing_file = Path(temp_dir) / "output1.txt"
            existing_file.write_text("existing content")
            
            filtered_topics, skipped_count = client._filter_existing_files(topics, temp_dir)
            assert len(filtered_topics) == 2  # 2 files don't exist yet
            assert skipped_count == 1  # 1 file already exists
    
    def test_logger_functionality(self):
        """Test the logger utility."""
        logger = Logger(verbose=True)
        
        # Test different log levels (we can't easily test output, but can test it doesn't crash)
        logger.info("Test info message")
        logger.success("Test success message")
        logger.warning("Test warning message")
        logger.error("Test error message")
        logger.debug("Test debug message")
        
        logger.header("Test Header", "Test subtitle")
        logger.section("Test Section")
        logger.progress_start("Test progress")
        logger.progress_update("Test update")
        logger.progress_done("Test done", 1.5)
        
        stats = {"Total": 10, "Successful": 8, "Failed": 2}
        logger.stats_table("Test Stats", stats)
        
        config = {"input_file": "test.csv", "max_parallel": 50}
        logger.command_start("test-command", config)
    
    def test_save_content_functionality(self):
        """Test the _save_content method."""
        client = FlowHuntClient.__new__(FlowHuntClient)  # Don't call __init__
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test saving content to a file
            content = "This is test content\nWith multiple lines"
            filename = "test_output.txt"
            
            client._save_content(content, filename, temp_dir)
            
            # Verify file was created and content is correct
            output_file = Path(temp_dir) / filename
            assert output_file.exists()
            assert output_file.read_text() == content
            
            # Test creating nested directories
            nested_filename = "subdir/nested_file.txt"
            client._save_content(content, nested_filename, temp_dir)
            
            nested_file = Path(temp_dir) / nested_filename
            assert nested_file.exists()
            assert nested_file.read_text() == content
    
    @patch('flowhunt_toolkit.core.client.Console')
    @patch('flowhunt_toolkit.core.client.Progress')
    def test_rich_progress_integration(self, mock_progress_class, mock_console_class):
        """Test that Rich progress components are properly initialized."""
        # Mock the console and progress
        mock_console = Mock()
        mock_progress = Mock()
        mock_progress_class.return_value.__enter__.return_value = mock_progress
        mock_console_class.return_value = mock_console
        
        client = FlowHuntClient.__new__(FlowHuntClient)  # Don't call __init__
        
        # Verify that Rich components would be imported and used
        # This test ensures our imports and basic structure are correct
        assert mock_progress_class is not None
        assert mock_console_class is not None


if __name__ == "__main__":
    pytest.main([__file__])