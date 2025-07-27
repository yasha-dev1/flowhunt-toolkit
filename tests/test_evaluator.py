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
        'question': ['What is 2+2?', 'What is the capital of France?'],
        'expected_answer': ['4', 'Paris']
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
    csv_file.write_text("question,expected_answer\nWhat is 2+2?,4\nWhat is 3+3?,6\n")
    
    df = evaluator.load_evaluation_data(csv_file)
    assert len(df) == 2
    assert list(df.columns) == ['question', 'expected_answer']
    assert df.iloc[0]['question'] == 'What is 2+2?'
    assert str(df.iloc[0]['expected_answer']) == '4'


def test_load_evaluation_data_missing_columns(evaluator, tmp_path):
    """Test loading CSV with missing required columns."""
    csv_file = tmp_path / "test.csv"
    csv_file.write_text("query,answer\nWhat is 2+2?,4\n")
    
    with pytest.raises(ValueError, match="missing required columns"):
        evaluator.load_evaluation_data(csv_file)


def test_create_judge_prompt(evaluator):
    """Test judge prompt creation."""
    question = "What is 2+2?"
    expected = "4"
    actual = "The answer is 4"
    
    prompt = evaluator._create_judge_prompt(question, expected, actual)
    
    assert question in prompt
    assert expected in prompt
    assert actual in prompt
    assert "score" in prompt.lower()
    assert "reasoning" in prompt.lower()


def test_evaluate_batch_structure(evaluator, sample_csv_data):
    """Test that evaluate_batch returns correct structure."""
    flow_id = "test-flow-id"
    
    results = evaluator._evaluate_batch(flow_id, sample_csv_data)
    
    assert len(results) == 2
    for result in results:
        assert 'question' in result
        assert 'expected_answer' in result
        assert 'actual_answer' in result
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
