"""Tests for CLI functionality."""

import pytest
from click.testing import CliRunner
from flowhunt_toolkit.cli import main


def test_cli_help():
    """Test that CLI help command works."""
    runner = CliRunner()
    result = runner.invoke(main, ['--help'])
    assert result.exit_code == 0
    assert 'Flowhunt Toolkit' in result.output


def test_cli_version():
    """Test that CLI version command works."""
    runner = CliRunner()
    result = runner.invoke(main, ['--version'])
    assert result.exit_code == 0


def test_evaluate_command_help():
    """Test that evaluate command help works."""
    runner = CliRunner()
    result = runner.invoke(main, ['evaluate', '--help'])
    assert result.exit_code == 0
    assert 'LLM as a judge' in result.output


def test_evaluate_command_missing_config():
    """Test that evaluate command fails with missing configuration."""
    runner = CliRunner()
    # Create a dummy CSV file for testing
    with runner.isolated_filesystem():
        with open('test.csv', 'w') as f:
            f.write('question,expected_answer\n')
            f.write('What is 2+2?,4\n')
        
        result = runner.invoke(main, ['evaluate', 'test.csv', 'test-flow-id'])
        assert result.exit_code == 1
        assert 'No FlowHunt configuration found' in result.output


def test_flows_inspect_command_missing_config():
    """Test that flows inspect command fails with missing configuration."""
    runner = CliRunner()
    result = runner.invoke(main, ['flows', 'inspect', 'test-flow-id'])
    assert result.exit_code == 1
    assert 'No FlowHunt configuration found' in result.output


def test_batch_run_command_help():
    """Test that batch-run command help works."""
    runner = CliRunner()
    result = runner.invoke(main, ['batch-run', '--help'])
    assert result.exit_code == 0
    assert 'batch mode' in result.output


def test_auth_command_prompts_for_key():
    """Test that auth command prompts for API key."""
    runner = CliRunner()
    # Test with empty input (simulates user pressing Ctrl+C or just Enter)
    result = runner.invoke(main, ['auth'], input='\n')
    assert result.exit_code == 1
    assert 'Enter your FlowHunt API key' in result.output


def test_flows_list_command_missing_config():
    """Test that flows list command fails with missing configuration."""
    runner = CliRunner()
    result = runner.invoke(main, ['flows', 'list'])
    assert result.exit_code == 1
    assert 'No FlowHunt configuration found' in result.output
