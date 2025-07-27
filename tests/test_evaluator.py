"""Tests for flow evaluator functionality."""

import pytest
import pandas as pd
from unittest.mock import Mock, patch
from pathlib import Path

from flowhunt_toolkit.core.evaluator import FlowEvaluator
from flowhunt_toolkit.core.client import FlowHuntClient


@pytest.fixture
def mock_client():
    """Create a mock FlowHunt client."""
    return Mock(spec=FlowHuntClient)


@pytest.fixture
def evaluator(mock_client):
    """Create a FlowEvaluator instance with mock client."""
    return FlowEvaluator(mock_client)


@pytest.fixture
def sample_csv_data():
    """Create sample CSV data for testing."""
    return pd.DataFrame({
        'flow_input': ['What is 2+2?', 'What is the capital of France?'],
        'expected_output': ['4', 'Paris']
    })


def test_evaluator_initialization(mock_client):
    """Test FlowEvaluator initialization."""
    evaluator = FlowEvaluator(mock_client)
    assert evaluator.client == mock_client
    assert evaluator.judge_flow_id == FlowEvaluator.DEFAULT_JUDGE_FLOW_ID
    
    custom_judge_id = "custom-judge-flow"
    evaluator_custom = FlowEvaluator(mock_client, custom_judge_id)
    assert evaluator_custom.judge_flow_id == custom_judge_id


def test_load_evaluation_data_valid_csv(evaluator, tmp_path):
    """Test loading valid CSV data."""
    csv_file = tmp_path / "test.csv"
    csv_file.write_text("flow_input,expected_output\nWhat is 2+2?,4\nWhat is 3+3?,6\n")
    
    df = evaluator.load_evaluation_data(csv_file)
    assert len(df) == 2
    assert list(df.columns) == ['flow_input', 'expected_output']
    assert df.iloc[0]['flow_input'] == 'What is 2+2?'
    assert str(df.iloc[0]['expected_output']) == '4'


def test_load_evaluation_data_missing_columns(evaluator, tmp_path):
    """Test loading CSV with missing required columns."""
    csv_file = tmp_path / "test.csv"
    csv_file.write_text("query,answer\nWhat is 2+2?,4\n")
    
    with pytest.raises(ValueError, match="missing required columns"):
        evaluator.load_evaluation_data(csv_file)


def test_judge_answer(evaluator):
    """Test judge answer functionality."""
    expected = "4"
    actual = "The answer is 4"
    
    # Mock the client's execute_flow method
    evaluator.client.execute_flow.return_value = '{"total_rating": 8, "reasoning": "Good answer", "correctness": "Correct"}'
    
    result = evaluator._judge_answer(expected, actual)
    
    assert result["total_rating"] == 8
    assert result["reasoning"] == "Good answer"
    assert result["correctness"] == "Correct"


def test_evaluate_batch_structure(evaluator, sample_csv_data):
    """Test that evaluate_batch returns correct structure."""
    flow_id = "test-flow-id"
    
    # Mock the client methods
    evaluator.client.execute_flow.side_effect = [
        "The answer is 4",  # First flow execution
        '{"total_rating": 8, "reasoning": "Good answer", "correctness": "Correct"}',  # First judge
        "Paris is the capital",  # Second flow execution
        '{"total_rating": 9, "reasoning": "Perfect answer", "correctness": "Correct"}'  # Second judge
    ]
    
    results = evaluator.evaluate_batch(flow_id, sample_csv_data)
    
    assert len(results) == 2
    for result in results:
        assert 'question' in result
        assert 'expected_answer' in result
        assert 'actual_answer' in result
        assert 'judge_score' in result
        assert 'flow_id' in result
        assert 'judge_score' in result
        assert 'judge_reasoning' in result
        assert 'flow_id' in result
        assert 'judge_flow_id' in result
        assert result['flow_id'] == flow_id


def test_save_results_json(evaluator, tmp_path):
    """Test saving results to JSON file."""
    results = [
        {"question": "test", "score": 8},
        {"question": "test2", "score": 9}
    ]
    output_file = tmp_path / "results.json"
    
    evaluator.save_results(results, output_file)
    
    assert output_file.exists()
    import json
    with open(output_file) as f:
        saved_data = json.load(f)
    assert saved_data == results


def test_save_results_csv(evaluator, tmp_path):
    """Test saving results to CSV file."""
    results = [
        {"question": "test", "score": 8},
        {"question": "test2", "score": 9}
    ]
    output_file = tmp_path / "results.csv"
    
    evaluator.save_results(results, output_file)
    
    assert output_file.exists()
    df = pd.read_csv(output_file)
    assert len(df) == 2
    assert list(df.columns) == ['question', 'score']


def test_save_results_unsupported_format(evaluator, tmp_path):
    """Test saving results with unsupported format."""
    results = [{"question": "test", "score": 8}]
    output_file = tmp_path / "results.txt"
    
    with pytest.raises(ValueError, match="Unsupported output format"):
        evaluator.save_results(results, output_file)
