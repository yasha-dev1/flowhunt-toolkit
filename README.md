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
- **🔄 Data Indexing**: Index external data sources (LiveAgent, PDFs, etc.) into FlowHunt

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
| `flowhunt index` | Index external data sources into FlowHunt |

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

## 🔄 Indexing

The indexing feature allows you to import data from external sources into FlowHunt by processing them through flows.

### PDF Document Indexing

Index PDF documents by extracting text, chunking it based on token count, and processing each chunk through a FlowHunt flow. Uses tiktoken for accurate token counting.

#### Basic Usage

```bash
flowhunt index pdf <PDF_PATH> <INDEX_FLOW_ID> \
    --max-tokens 8000 \
    --output-csv results.csv
```

#### Parameters

- **PDF_PATH**: Path to the PDF file to process
- **INDEX_FLOW_ID**: The FlowHunt flow ID to process text chunks through
- **--max-tokens**: Maximum tokens per chunk (default: 8000)
- **--output-csv**: Path to save processing results CSV (auto-generated if not specified)

#### Examples

**Index a PDF with default 8k token chunks:**
```bash
flowhunt index pdf document.pdf flow-id-123
```

**Index with custom token limit:**
```bash
flowhunt index pdf document.pdf flow-id-123 --max-tokens 4000
```

**Specify output location:**
```bash
flowhunt index pdf document.pdf flow-id-123 --output-csv my_results.csv
```

#### Features

- **📄 PDF Text Extraction**: Extracts text from all pages using pypdf
- **🔢 Token Counting**: Uses tiktoken (cl100k_base encoding) for accurate GPT-4/ChatGPT token counting
- **✂️ Smart Chunking**: Intelligently splits text by paragraphs and sentences to respect token limits
- **📊 Progress Tracking**: Real-time progress bar with success/failure counts
- **📝 CSV Export**: Tracks all processed chunks with metadata:
  - Chunk index and token count
  - Text preview
  - Flow process ID for tracking in FlowHunt
  - Processing timestamp and status
- **⚡ Rate Limiting**: Built-in delays to respect API limits
- **🔄 Error Handling**: Continues processing on errors, saves partial results

#### Flow Variables

Each chunk is processed with the following variables available in your flow:
- `chunk_index`: Current chunk number (1-based)
- `total_chunks`: Total number of chunks
- `token_count`: Number of tokens in this chunk
- `pdf_filename`: Original PDF filename
- `source`: Always set to "pdf"

#### CSV Output Format

The results CSV contains:
```csv
chunk_index,token_count,chunk_preview,flow_process_id,status,indexed_at
```

This allows you to:
- Track which chunks have been processed
- Monitor processing success/failure
- Verify token counts per chunk
- Link back to FlowHunt processes

### LiveAgent Ticket Indexing

Index closed/resolved support tickets from LiveAgent into FlowHunt for knowledge base creation, analysis, or training data. Only closed tickets are indexed to ensure complete conversation history.

#### Basic Usage

```bash
flowhunt index liveagent <BASE_URL> <INDEX_FLOW_ID> [DEPARTMENT_ID] \
    --api-key <LIVEAGENT_API_KEY> \
    --limit 100 \
    --output-csv checkpoint.csv
```

#### Parameters

- **BASE_URL**: Your LiveAgent instance URL (e.g., `https://support.qualityunit.com`)
- **INDEX_FLOW_ID**: The FlowHunt flow ID to process tickets through
- **DEPARTMENT_ID** (optional): Filter tickets by department ID (e.g., `31ivft8h`)
- **--api-key**: LiveAgent API key (requires read-only access to tickets)
- **--limit**: Maximum number of tickets to index (default: 100)
- **--output-csv**: Path to save checkpoint CSV for tracking progress
- **--resume**: Resume indexing from a previous checkpoint file

#### Examples

**Index all tickets (up to 100):**
```bash
flowhunt index liveagent https://support.example.com flow-id-123 \
    --api-key your-api-key \
    --limit 100 \
    --output-csv tickets_index.csv
```

**Index tickets from specific department:**
```bash
flowhunt index liveagent https://support.example.com flow-id-123 dept-456 \
    --api-key your-api-key \
    --limit 50 \
    --output-csv dept_tickets.csv
```

**Resume interrupted indexing:**
```bash
flowhunt index liveagent https://support.example.com flow-id-123 \
    --api-key your-api-key \
    --resume \
    --output-csv tickets_index.csv
```

#### Features

- **📊 Progress Tracking**: Real-time progress bar with success/failure counts
- **💾 Checkpoint System**: Automatically saves progress after each ticket
- **🔄 Resume Capability**: Continue from where you left off if interrupted
- **📝 CSV Export**: Tracks all indexed tickets with metadata:
  - Ticket ID, code, and subject
  - Department information
  - Customer email and status
  - Flow process ID for tracking in FlowHunt
  - Indexing timestamp
- **⚡ Rate Limiting**: Built-in delays to respect API limits
- **🎯 Filtering**: Filter by department, indexes only closed tickets
- **📧 Email Focus**: Indexes only email channel tickets

#### Ticket Format

Each ticket is formatted as structured text containing:
- Ticket metadata (ID, subject, status, department, customer)
- Full conversation history with timestamps
- Messages clearly labeled by sender (Agent/Customer)

#### CSV Checkpoint Format

The checkpoint CSV contains:
```csv
ticket_id,ticket_code,ticket_subject,department_id,department_name,created_at,status,customer_email,flow_input_length,flow_process_id,indexed_at
```

This allows you to:
- Track which tickets have been indexed
- Monitor indexing success/failure
- Resume indexing without duplicates
- Analyze indexing metrics

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
