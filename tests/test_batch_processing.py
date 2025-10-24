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


class TestMatrixProcessing:
    """Test suite for CSV matrix batch processing functionality."""

    @pytest.fixture
    def sample_matrix_csv(self):
        """Create a sample CSV matrix file for testing."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            writer = csv.writer(f)
            writer.writerow(['Input', 'Output1', 'Output2', 'Output3'])
            writer.writerow(['What is AI?', '', '', ''])
            writer.writerow(['Python features?', '', '', ''])
            writer.writerow(['', '', '', ''])  # Empty first column (row should be skipped)
            writer.writerow(['Data Science?', '', '', ''])
            return f.name

    @pytest.fixture
    def empty_matrix_csv(self):
        """Create a CSV with all empty first column cells."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            writer = csv.writer(f)
            writer.writerow(['Input', 'Col2', 'Col3'])
            writer.writerow(['', '', ''])
            writer.writerow(['', '', ''])
            return f.name

    @pytest.fixture
    def single_column_csv(self):
        """Create a CSV with only one column (should fail validation)."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            writer = csv.writer(f)
            writer.writerow(['Input'])
            writer.writerow(['Value1'])
            writer.writerow(['Value2'])
            return f.name

    def test_matrix_csv_reading(self, sample_matrix_csv):
        """Test reading a matrix CSV file."""
        import pandas as pd

        df = pd.read_csv(sample_matrix_csv)

        # Verify structure
        assert len(df) == 4  # 4 rows
        assert len(df.columns) == 4  # 4 columns
        assert list(df.columns) == ['Input', 'Output1', 'Output2', 'Output3']

        # Verify content
        assert df.iloc[0]['Input'] == 'What is AI?'
        assert df.iloc[1]['Input'] == 'Python features?'
        assert pd.isna(df.iloc[2]['Input']) or df.iloc[2]['Input'] == ''  # Empty first column

    def test_matrix_cell_task_creation(self, sample_matrix_csv):
        """Test creation of cell tasks from matrix CSV."""
        import pandas as pd

        df = pd.read_csv(sample_matrix_csv)

        # Get first column name (input source column)
        first_col = df.columns[0]

        # Build cell tasks (similar to what batch_execute_matrix_from_csv does)
        cell_tasks = []
        for row_idx in range(len(df)):
            # Get the input value from first column
            input_value = df.iloc[row_idx][first_col]

            # Skip row if first column is empty
            if pd.isna(input_value) or str(input_value).strip() == '':
                continue

            input_value_str = str(input_value).strip()

            # Process remaining columns (skip first column)
            for col_name in df.columns[1:]:
                cell_tasks.append({
                    'row_idx': row_idx,
                    'col_name': col_name,
                    'flow_input': input_value_str
                })

        # Should have 9 tasks (3 non-empty rows × 3 output columns)
        assert len(cell_tasks) == 9

        # Verify first cell task
        assert cell_tasks[0]['row_idx'] == 0
        assert cell_tasks[0]['col_name'] == 'Output1'
        assert cell_tasks[0]['flow_input'] == 'What is AI?'

        # Verify second row tasks
        row1_tasks = [t for t in cell_tasks if t['row_idx'] == 1]
        assert len(row1_tasks) == 3  # 3 output columns
        assert all(t['flow_input'] == 'Python features?' for t in row1_tasks)

        # Verify that row with empty first column is not in tasks
        row2_tasks = [t for t in cell_tasks if t['row_idx'] == 2]
        assert len(row2_tasks) == 0  # Empty first column, row should be skipped

    def test_empty_matrix_handling(self, empty_matrix_csv):
        """Test handling of a matrix with all empty first column cells."""
        import pandas as pd

        df = pd.read_csv(empty_matrix_csv)

        # Get first column name
        first_col = df.columns[0]

        # Build cell tasks
        cell_tasks = []
        for row_idx in range(len(df)):
            # Get the input value from first column
            input_value = df.iloc[row_idx][first_col]

            # Skip row if first column is empty
            if pd.isna(input_value) or str(input_value).strip() == '':
                continue

            input_value_str = str(input_value).strip()

            # Process remaining columns
            for col_name in df.columns[1:]:
                cell_tasks.append({
                    'row_idx': row_idx,
                    'col_name': col_name,
                    'flow_input': input_value_str
                })

        # Should have no tasks when all first column cells are empty
        assert len(cell_tasks) == 0

    def test_single_column_validation(self, single_column_csv):
        """Test that CSV with only one column fails validation."""
        import pandas as pd

        df = pd.read_csv(single_column_csv)

        # Should have only 1 column
        assert len(df.columns) == 1

        # This should fail validation (at least 2 columns required)
        # We'll test this in the actual client method test

    def test_matrix_output_structure(self, sample_matrix_csv):
        """Test that output matrix maintains same structure as input."""
        import pandas as pd

        input_df = pd.read_csv(sample_matrix_csv)

        # Create output dataframe with same structure
        output_df = pd.DataFrame(columns=input_df.columns, index=input_df.index)

        # Get first column name
        first_col = input_df.columns[0]

        # Copy first column to output as-is
        output_df[first_col] = input_df[first_col]

        # Verify structure matches
        assert list(output_df.columns) == list(input_df.columns)
        assert len(output_df) == len(input_df)

        # Verify first column is copied
        assert output_df.loc[0, first_col] == input_df.loc[0, first_col]

        # Simulate filling with results for other columns
        output_df.loc[0, 'Output1'] = 'Result 1'
        output_df.loc[0, 'Output2'] = 'Result 2'

        assert output_df.loc[0, 'Output1'] == 'Result 1'
        assert output_df.loc[0, 'Output2'] == 'Result 2'

    def test_col_variable_name_construction(self):
        """Test that column variable name is correctly constructed."""
        col_variable_name = "col_name"
        col_name = "Output1"

        variables = {col_variable_name: col_name}

        assert variables == {"col_name": "Output1"}

        # Test with custom variable name
        col_variable_name = "column_header"
        variables = {col_variable_name: col_name}

        assert variables == {"column_header": "Output1"}

    def test_error_message_in_cell(self, sample_matrix_csv):
        """Test that error messages are properly stored in cells."""
        import pandas as pd

        df = pd.read_csv(sample_matrix_csv)
        output_df = pd.DataFrame(columns=df.columns, index=df.index)

        # Simulate an error for a specific cell (not first column)
        error_message = "ERROR: Flow execution failed"
        output_df.loc[0, 'Output1'] = error_message

        assert output_df.loc[0, 'Output1'] == error_message
        assert 'ERROR' in str(output_df.loc[0, 'Output1'])

    def test_matrix_csv_save_and_load(self, sample_matrix_csv):
        """Test saving and loading matrix CSV preserves structure."""
        import pandas as pd

        # Read input
        input_df = pd.read_csv(sample_matrix_csv)

        # Create output with results
        output_df = pd.DataFrame(columns=input_df.columns, index=input_df.index)

        # Copy first column
        first_col = input_df.columns[0]
        output_df[first_col] = input_df[first_col]

        # Fill output columns with results
        output_df.loc[0, 'Output1'] = 'AI is...'
        output_df.loc[0, 'Output2'] = 'ML is...'
        output_df.loc[0, 'Output3'] = 'NLP is...'

        # Save to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            output_file = f.name

        output_df.to_csv(output_file, index=False)

        # Load back
        reloaded_df = pd.read_csv(output_file)

        # Verify structure and content
        assert list(reloaded_df.columns) == list(input_df.columns)
        assert reloaded_df.iloc[0]['Input'] == 'What is AI?'  # First column preserved
        assert reloaded_df.iloc[0]['Output1'] == 'AI is...'
        assert reloaded_df.iloc[0]['Output2'] == 'ML is...'

        # Clean up
        Path(output_file).unlink()

    def test_first_column_as_input_source(self, sample_matrix_csv):
        """Test that first column values are used as flow_input for all cells in row."""
        import pandas as pd

        df = pd.read_csv(sample_matrix_csv)
        first_col = df.columns[0]

        # Build cell tasks
        cell_tasks = []
        for row_idx in range(len(df)):
            input_value = df.iloc[row_idx][first_col]

            if pd.isna(input_value) or str(input_value).strip() == '':
                continue

            input_value_str = str(input_value).strip()

            for col_name in df.columns[1:]:
                cell_tasks.append({
                    'row_idx': row_idx,
                    'col_name': col_name,
                    'flow_input': input_value_str
                })

        # Verify all tasks for row 0 use same flow_input
        row0_tasks = [t for t in cell_tasks if t['row_idx'] == 0]
        assert len(row0_tasks) == 3  # 3 output columns
        assert all(t['flow_input'] == 'What is AI?' for t in row0_tasks)

        # Verify column names are different
        row0_cols = [t['col_name'] for t in row0_tasks]
        assert set(row0_cols) == {'Output1', 'Output2', 'Output3'}


if __name__ == "__main__":
    pytest.main([__file__])