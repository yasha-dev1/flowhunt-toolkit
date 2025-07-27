# FlowHunt Dev Toolkit

🚀 **A comprehensive CLI toolkit for FlowHunt Flow Engineers** to streamline flow development, evaluation, and management.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![GitHub release](https://img.shields.io/github/release/yasha-dev1/flowhunt-toolkit.svg)](https://github.com/yasha-dev1/flowhunt-toolkit/releases)

## ✨ Features

- **🔍 Flow Evaluation**: LLM-as-a-Judge evaluation system with comprehensive statistics
- **📊 Batch Processing**: Execute flows in batches with CSV input/output
- **🔧 Flow Management**: List, inspect, and manage your FlowHunt flows
- **📈 Rich Reports**: Generate detailed evaluation reports with statistics and visualizations
- **🔐 Authentication**: Secure API authentication management
- **⚡ Performance**: Optimized for large-scale flow operations

## 🚀 Quick Install

### One-Line Installation (Recommended)

For **macOS** and **Linux** systems:

```bash
curl -sSL https://raw.githubusercontent.com/yasha-dev1/flowhunt-toolkit/main/install.sh | bash
```

This will:
- ✅ Install all dependencies
- ✅ Download and install FlowHunt Toolkit
- ✅ Add `flowhunt` command to your PATH
- ✅ Set up everything automatically

### Manual Installation

If you prefer manual installation:

```bash
# Clone the repository
git clone https://github.com/yasha-dev1/flowhunt-toolkit.git
cd flowhunt-toolkit

# Install with pip
pip install -e .
```

### Verify Installation

```bash
flowhunt --help
flowhunt --version
```

## 🔧 Quick Start

### 1. Authentication

First, authenticate with your FlowHunt API:

```bash
flowhunt auth
```

### 2. List Your Flows

```bash
flowhunt flows list
```

### 3. Evaluate a Flow

Create a CSV file with your test data:

```csv
flow_input,expected_output
"What is 2+2?","4"
"What is the capital of France?","Paris"
```

Run evaluation:

```bash
flowhunt evaluate your-flow-id path/to/test-data.csv --judge-flow-id your-judge-flow-id
```

### 4. Batch Execute Flows

```bash
flowhunt batch-run your-flow-id input.csv --output-dir results/
```

## 📚 Commands Overview

| Command | Description |
|---------|-------------|
| `flowhunt auth` | Authenticate with FlowHunt API |
| `flowhunt evaluate` | Evaluate flows using LLM-as-a-Judge |
| `flowhunt flows list` | List all available flows |
| `flowhunt flows inspect` | Inspect flow configuration |
| `flowhunt batch-run` | Execute flows in batch mode |

## 📖 Detailed Usage

### Flow Evaluation

The evaluation system uses LLM-as-a-Judge methodology:

```bash
flowhunt evaluate FLOW_ID TEST_DATA.csv \
  --judge-flow-id JUDGE_FLOW_ID \
  --output-dir eval_results/ \
  --batch-size 10 \
  --verbose
```

**Features:**
- 📊 Comprehensive statistics (mean, median, std, quartiles)
- 📈 Score distribution analysis
- 📋 Automated CSV result export
- 🎯 Pass/fail rate calculation
- 🔍 Error tracking and reporting

### CSV Input Format

For evaluation:
```csv
flow_input,expected_output
"Your question here","Expected answer"
```

For batch execution:
```csv
flow_input,filename,flow_variable
"Input data","output1.json","additional_var"
```

### Batch Processing

```bash
flowhunt batch-run FLOW_ID input.csv \
  --output-dir results/ \
  --batch-size 5 \
  --max-workers 3
```

## 🛠️ Development

### Prerequisites

- Python 3.8+
- pip
- git

### Setup Development Environment

```bash
git clone https://github.com/yasha-dev1/flowhunt-toolkit.git
cd flowhunt-toolkit

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=flowhunt_toolkit

# Run specific test file
pytest tests/test_evaluator.py
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📞 Support

If you encounter any issues or have questions:

- 🐛 [Report bugs](https://github.com/yasha-dev1/flowhunt-toolkit/issues)
- 💡 [Request features](https://github.com/yasha-dev1/flowhunt-toolkit/issues)
- 📖 [View documentation](https://github.com/yasha-dev1/flowhunt-toolkit)

## 🏆 Acknowledgments

- FlowHunt team for providing the excellent platform
- Contributors and beta testers
- Open source community

---

**Made with ❤️ for FlowHunt Flow Engineers** 
