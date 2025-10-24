# FlowHunt Dev Toolkit

🚀 **A comprehensive CLI toolkit for FlowHunt Flow Engineers** to streamline flow development, evaluation, and management.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![GitHub release](https://img.shields.io/github/release/yasha-dev1/flowhunt-toolkit.svg)](https://github.com/yasha-dev1/flowhunt-toolkit/releases)

## ✨ Features

- **🔍 Flow Evaluation**: LLM-as-a-Judge evaluation system with comprehensive statistics
- **📊 Batch Processing**: Execute flows in batches with CSV input/output
- **🔢 CSV Matrix Processing**: Process each cell in a CSV as a separate flow execution
- **🔧 Flow Management**: List, inspect, and manage your FlowHunt flows
- **📈 Rich Reports**: Generate detailed evaluation reports with statistics and visualizations
- **🔐 Authentication**: Secure API authentication management
- **⚡ Performance**: Optimized for large-scale flow operations with parallel execution
- **🔄 Data Indexing**: Index external data sources (LiveAgent, PDFs, DOCX, URLs, sitemaps, folders) into FlowHunt

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
| `flowhunt batch-run-matrix` | Process CSV matrix cells as individual flow executions |
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

FlowHunt Toolkit provides two powerful batch processing modes for different use cases.

#### Standard Batch Processing (`batch-run`)

Process multiple inputs from a CSV file where each row is a separate flow execution:

```bash
flowhunt batch-run FLOW_ID input.csv \
  --output-dir results/ \
  --max-parallel 50 \
  --check-interval 2
```

**CSV Format for batch-run:**
```csv
flow_input,filename,flow_variable
"Process this text","output1.txt","{\"key\": \"value\"}"
"Another input","output2.txt","{\"key\": \"value2\"}"
```

**Features:**
- 🚀 **Parallel Execution**: Control parallelism with `--max-parallel` (default: 50)
- 📁 **File Output**: Save results to individual files using `filename` column
- 🔄 **Variables Support**: Pass JSON variables via `flow_variable` column
- ⚡ **Fast Polling**: Configurable `--check-interval` (default: 2s)
- 📊 **Progress Tracking**: Real-time progress bars with success/failure counts

**Options:**
- `--output-file`: Single output file for aggregated results
- `--output-dir`: Directory for individual file outputs (requires `filename` column)
- `--format`: Output format (csv or json)
- `--max-parallel`: Maximum parallel executions (default: 50)
- `--check-interval`: Seconds between result checks (default: 2)
- `--sequential`: Force sequential execution instead of parallel
- `--overwrite`: Overwrite existing output files

#### CSV Matrix Processing (`batch-run-matrix`)

Process each row where the **first column contains the input** and subsequent columns represent different processing variants - perfect for A/B testing, multi-variant evaluation, or systematic comparisons:

```bash
flowhunt batch-run-matrix input.csv FLOW_ID \
  --output-file results.csv \
  --col-variable-name "variant" \
  --max-parallel 100
```

**CSV Input Example:**
```csv
Input,Approach1,Approach2,Approach3
What is AI?,,,
Python features?,,,
Data Science?,,,
```

**How it works:**
- **First column** = input source for that row
- **Each subsequent column** = a processing variant that receives the first column's value
- For row 1: "What is AI?" is sent to Approach1, Approach2, and Approach3 (3 executions)
- For row 2: "Python features?" is sent to Approach1, Approach2, and Approach3 (3 executions)
- Total: 6 executions (2 rows × 3 processing columns)
- First column → preserved as-is in output
- Column name → flow variable (e.g., `{"variant": "Approach1"}`)
- Rows with empty first column are skipped entirely
- Output CSV maintains the same structure as input

**Output Example:**
```csv
Input,Approach1,Approach2,Approach3
What is AI?,Artificial Intelligence refers to...,AI is the simulation of...,Artificial Intelligence means...
Python features?,Python offers dynamic typing...,Python provides simplicity...,Python includes extensive libraries...
Data Science?,Data Science is an interdisciplinary...,Data Science combines statistics...,Data Science involves extracting insights...
```

**Features:**
- 🔢 **Row-Based Processing**: First column value sent to all other columns in that row
- 🏷️ **Variant Context**: Column name passed as flow variable to distinguish variants
- ⚡ **High Parallelism**: Process hundreds of cells simultaneously
- 📊 **Structure Preservation**: First column preserved, other cells filled with results
- 🎯 **Smart Handling**: Empty first column rows skipped, errors shown in cells
- 💪 **Singleton Execution**: Uses `invoke_flow_singleton` for reliability

**Requirements:**
- CSV must have at least 2 columns (first = input, rest = processing columns)
- Total executions = (non-empty rows) × (number of columns - 1)

**Options:**
- `--output-file`: Output CSV path (auto-generated if not specified)
- `--col-variable-name`: Variable name for column headers (default: `col_name`)
- `--max-parallel`: Maximum parallel executions (default: 50)
- `--check-interval`: Seconds between result checks (default: 2)

**Flow Variables Available:**
For each execution, your flow receives:
```python
{
  "col_name": "Approach1"  # or your custom variable name
}
# flow_input = value from first column of that row
```

**Use Cases:**
- 🎯 **A/B Testing**: Test different prompts/approaches on same inputs (first column = input, other columns = different prompt variants)
- 🌐 **Multi-Language Translation**: Translate inputs to multiple languages (first column = text, other columns = target languages)
- 🔄 **Prompt Comparison**: Compare different prompt strategies (first column = task, other columns = different prompt templates)
- 📊 **Model Comparison**: Test same input across different models (first column = query, other columns = different model configurations)
- 🧪 **Systematic Evaluation**: Evaluate multiple approaches against same test cases

**Error Handling:**
Failed cells will contain error messages like:
```
ERROR: Flow execution failed - timeout occurred
```

This makes it easy to identify and retry failed cells.

**Complete Example:**

Input CSV (`questions.csv`):
```csv
Question,GPT4_Approach,Claude_Approach,Gemini_Approach
Explain quantum computing,,,
What is machine learning?,,,
Define blockchain,,,
```

Command:
```bash
flowhunt batch-run-matrix questions.csv my-flow-id \
  --col-variable-name "model_variant" \
  --max-parallel 100
```

In your flow, you can use the `model_variant` variable to customize behavior:
- When `model_variant = "GPT4_Approach"` → use GPT-4 specific prompt
- When `model_variant = "Claude_Approach"` → use Claude specific prompt
- When `model_variant = "Gemini_Approach"` → use Gemini specific prompt

Output CSV:
```csv
Question,GPT4_Approach,Claude_Approach,Gemini_Approach
Explain quantum computing,Quantum computing uses quantum bits...,Quantum computing leverages quantum mechanics...,Quantum computing exploits quantum phenomena...
What is machine learning?,Machine learning is a subset of AI...,Machine learning enables computers to learn...,Machine learning allows systems to improve...
Define blockchain,Blockchain is a distributed ledger...,Blockchain is a decentralized database...,Blockchain is an immutable record system...
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

### URL Indexing

Index individual web pages by extracting content, chunking it based on token count, and processing each chunk through a FlowHunt flow. HTML is converted to markdown for better formatting.

#### Basic Usage

```bash
flowhunt index url <URL> <INDEX_FLOW_ID> \
    --max-tokens 8000 \
    --output-csv results.csv
```

#### Parameters

- **URL**: The web page URL to process
- **INDEX_FLOW_ID**: The FlowHunt flow ID to process text chunks through
- **--max-tokens**: Maximum tokens per chunk (default: 8000)
- **--output-csv**: Path to save processing results CSV (auto-generated if not specified)

#### Examples

**Index a single URL with default settings:**
```bash
flowhunt index url https://example.com/article flow-id-123
```

**Index with custom token limit:**
```bash
flowhunt index url https://example.com/docs flow-id-123 --max-tokens 4000
```

**Specify output location:**
```bash
flowhunt index url https://example.com/page flow-id-123 --output-csv my_results.csv
```

#### Features

- **🌐 Web Content Extraction**: Fetches and extracts content from any URL
- **📝 HTML to Markdown**: Converts HTML to clean markdown format
- **🔢 Token Counting**: Uses tiktoken (cl100k_base encoding) for accurate token counting
- **✂️ Smart Chunking**: Intelligently splits text by paragraphs and sentences
- **📊 Progress Tracking**: Real-time progress bar with success/failure counts
- **📝 CSV Export**: Tracks all processed chunks with metadata
- **⚡ Rate Limiting**: Built-in delays to respect API limits

#### Flow Variables

Each chunk is processed with the following variables:
- `chunk_index`: Current chunk number (1-based)
- `total_chunks`: Total number of chunks
- `token_count`: Number of tokens in this chunk
- `url`: The source URL
- `source`: Always set to "url"

### Sitemap Indexing

Index all URLs from a sitemap.xml by processing each page's content through a FlowHunt flow. Supports both regular sitemaps and sitemap index files.

#### Basic Usage

```bash
flowhunt index sitemap <SITEMAP_URL> <INDEX_FLOW_ID> \
    --max-tokens 8000 \
    --limit 100 \
    --output-csv results.csv
```

#### Parameters

- **SITEMAP_URL**: URL of the sitemap.xml file
- **INDEX_FLOW_ID**: The FlowHunt flow ID to process text chunks through
- **--max-tokens**: Maximum tokens per chunk (default: 8000)
- **--limit**: Maximum number of URLs to process from sitemap (optional)
- **--output-csv**: Path to save processing results CSV (auto-generated if not specified)

#### Examples

**Index entire sitemap:**
```bash
flowhunt index sitemap https://example.com/sitemap.xml flow-id-123
```

**Index with URL limit:**
```bash
flowhunt index sitemap https://example.com/sitemap.xml flow-id-123 --limit 50
```

**Custom token limit and output:**
```bash
flowhunt index sitemap https://example.com/sitemap.xml flow-id-123 \
    --max-tokens 4000 \
    --output-csv sitemap_results.csv
```

#### Features

- **🗺️ Sitemap Parsing**: Automatically parses sitemap.xml files
- **📑 Sitemap Index Support**: Recursively processes sitemap index files
- **🌐 Multi-URL Processing**: Processes all URLs found in the sitemap
- **📝 HTML to Markdown**: Converts each page's HTML to markdown
- **🔢 Token Counting**: Uses tiktoken for accurate token counting
- **✂️ Smart Chunking**: Intelligently splits text by paragraphs and sentences
- **📊 Progress Tracking**: Real-time progress bar showing URL and chunk progress
- **💾 Checkpoint System**: Saves progress after each URL
- **📝 CSV Export**: Tracks all processed chunks with URL and chunk metadata
- **⚡ Rate Limiting**: Built-in delays to respect API limits

#### Flow Variables

Each chunk is processed with the following variables:
- `url_index`: Current URL number (1-based)
- `total_urls`: Total number of URLs in sitemap
- `chunk_index`: Current chunk number within the URL (1-based)
- `total_chunks`: Total chunks for this URL
- `token_count`: Number of tokens in this chunk
- `url`: The source URL
- `source`: Always set to "sitemap"

#### CSV Output Format

The results CSV contains:
```csv
url_index,url,chunk_index,token_count,chunk_preview,flow_process_id,status,indexed_at
```

This allows you to:
- Track which URLs and chunks have been processed
- Monitor processing success/failure per URL and chunk
- Resume processing if interrupted
- Link back to FlowHunt processes

### DOCX Document Indexing

Index DOCX documents by extracting text, chunking it based on token count, and processing each chunk through a FlowHunt flow. Uses tiktoken for accurate token counting.

#### Basic Usage

```bash
flowhunt index docx <DOCX_PATH> <INDEX_FLOW_ID> \
    --max-tokens 8000 \
    --output-csv results.csv
```

#### Parameters

- **DOCX_PATH**: Path to the DOCX file to process
- **INDEX_FLOW_ID**: The FlowHunt flow ID to process text chunks through
- **--max-tokens**: Maximum tokens per chunk (default: 8000)
- **--output-csv**: Path to save processing results CSV (auto-generated if not specified)

#### Examples

**Index a DOCX with default 8k token chunks:**
```bash
flowhunt index docx document.docx flow-id-123
```

**Index with custom token limit:**
```bash
flowhunt index docx document.docx flow-id-123 --max-tokens 4000
```

**Specify output location:**
```bash
flowhunt index docx document.docx flow-id-123 --output-csv my_results.csv
```

#### Features

- **📄 DOCX Text Extraction**: Extracts text from all paragraphs using python-docx
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
- `docx_filename`: Original DOCX filename
- `source`: Always set to "docx"

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

### Folder Indexing

Index all PDF and DOCX files in a folder by automatically detecting file types, extracting text, chunking it, and processing each chunk through a FlowHunt flow. Perfect for batch processing document collections.

#### Basic Usage

```bash
flowhunt index folder <FOLDER_PATH> <INDEX_FLOW_ID> \
    --max-tokens 8000 \
    --output-csv results.csv
```

#### Parameters

- **FOLDER_PATH**: Path to the folder containing PDF and/or DOCX files
- **INDEX_FLOW_ID**: The FlowHunt flow ID to process text chunks through
- **--max-tokens**: Maximum tokens per chunk (default: 8000)
- **--output-csv**: Path to save processing results CSV (auto-generated if not specified)

#### Examples

**Index all documents in a folder:**
```bash
flowhunt index folder ./documents flow-id-123
```

**Index with custom token limit:**
```bash
flowhunt index folder ./documents flow-id-123 --max-tokens 4000
```

**Specify output location:**
```bash
flowhunt index folder ./documents flow-id-123 --output-csv folder_results.csv
```

#### Features

- **🔍 Auto-Detection**: Automatically finds and processes PDF and DOCX files
- **📂 Multi-Format Support**: Handles both PDF and DOCX files in the same folder
- **🗑️ Smart Filtering**: Filters out temporary Word files (starting with ~$)
- **📄 Text Extraction**: Uses specialized processors for each file type
- **🔢 Token Counting**: Uses tiktoken for accurate token counting
- **✂️ Smart Chunking**: Intelligently splits text by paragraphs and sentences
- **📊 Progress Tracking**: Real-time progress bar showing file and chunk progress
- **💾 Checkpoint System**: Saves progress after each file
- **📝 CSV Export**: Tracks all processed chunks with file and chunk metadata
- **⚡ Rate Limiting**: Built-in delays to respect API limits
- **🔄 Error Handling**: Continues processing on errors, saves partial results

#### Flow Variables

Each chunk is processed with the following variables:
- `file_index`: Current file number (1-based)
- `total_files`: Total number of files in folder
- `chunk_index`: Current chunk number within the file (1-based)
- `total_chunks`: Total chunks for this file
- `token_count`: Number of tokens in this chunk
- `filename`: Original filename
- `file_type`: File type ("pdf" or "docx")
- `source`: Always set to "folder"

#### CSV Output Format

The results CSV contains:
```csv
file_index,filename,file_type,chunk_index,token_count,chunk_preview,flow_process_id,status,indexed_at
```

This allows you to:
- Track which files and chunks have been processed
- Monitor processing success/failure per file and chunk
- See file type breakdown (PDF vs DOCX)
- Resume processing if interrupted
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
