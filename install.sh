#!/bin/bash

# FlowHunt Toolkit Installation Script
# Supports macOS and Linux systems
# Usage: curl -sSL https://raw.githubusercontent.com/yasha-dev1/flowhunt-toolkit/main/install.sh | bash

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
REPO_URL="https://github.com/yasha-dev1/flowhunt-toolkit"
REPO_NAME="flowhunt-toolkit"
INSTALL_DIR="$HOME/.flowhunt-toolkit"
BIN_DIR="$HOME/.local/bin"

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to detect OS
detect_os() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        echo "linux"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        echo "macos"
    else
        echo "unknown"
    fi
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check Python version
check_python() {
    if command_exists python3; then
        PYTHON_CMD="python3"
    elif command_exists python; then
        PYTHON_CMD="python"
    else
        print_error "Python is not installed. Please install Python 3.8 or higher."
        exit 1
    fi

    # Check Python version
    PYTHON_VERSION=$($PYTHON_CMD -c "import sys; print('.'.join(map(str, sys.version_info[:2])))")
    PYTHON_MAJOR=$($PYTHON_CMD -c "import sys; print(sys.version_info.major)")
    PYTHON_MINOR=$($PYTHON_CMD -c "import sys; print(sys.version_info.minor)")

    if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 8 ]); then
        print_error "Python 3.8 or higher is required. Found Python $PYTHON_VERSION"
        exit 1
    fi

    print_success "Python $PYTHON_VERSION found"
}

# Function to check pip
check_pip() {
    if command_exists pip3; then
        PIP_CMD="pip3"
    elif command_exists pip; then
        PIP_CMD="pip"
    else
        print_error "pip is not installed. Please install pip."
        exit 1
    fi
    print_success "pip found"
}

# Function to install system dependencies
install_system_deps() {
    OS=$(detect_os)
    print_status "Detected OS: $OS"

    case $OS in
        "linux")
            if command_exists apt-get; then
                print_status "Installing system dependencies with apt..."
                sudo apt-get update -qq
                sudo apt-get install -y git curl python3-pip python3-venv
            elif command_exists yum; then
                print_status "Installing system dependencies with yum..."
                sudo yum install -y git curl python3-pip python3-venv
            elif command_exists dnf; then
                print_status "Installing system dependencies with dnf..."
                sudo dnf install -y git curl python3-pip python3-venv
            elif command_exists pacman; then
                print_status "Installing system dependencies with pacman..."
                sudo pacman -S --noconfirm git curl python-pip python-virtualenv
            else
                print_warning "Could not detect package manager. Please ensure git, curl, and python3-pip are installed."
            fi
            ;;
        "macos")
            if command_exists brew; then
                print_status "Installing system dependencies with Homebrew..."
                brew install git curl python3
            else
                print_warning "Homebrew not found. Please ensure git, curl, and Python 3 are installed."
                print_status "You can install Homebrew from: https://brew.sh"
            fi
            ;;
        *)
            print_warning "Unsupported OS. Please ensure git, curl, and Python 3.8+ are installed."
            ;;
    esac
}

# Function to create directories
create_directories() {
    print_status "Creating installation directories..."
    mkdir -p "$INSTALL_DIR"
    mkdir -p "$BIN_DIR"
    print_success "Directories created"
}

# Function to download and install
install_flowhunt() {
    print_status "Downloading FlowHunt Toolkit..."
    
    # Remove existing installation if it exists
    if [ -d "$INSTALL_DIR" ]; then
        print_status "Removing existing installation..."
        rm -rf "$INSTALL_DIR"
    fi

    # Clone the repository
    git clone "$REPO_URL.git" "$INSTALL_DIR"
    cd "$INSTALL_DIR"

    print_status "Installing FlowHunt Toolkit..."
    
    # Install using pip in user mode
    $PIP_CMD install --user -e .

    print_success "FlowHunt Toolkit installed successfully"
}

# Function to create wrapper script
create_wrapper() {
    print_status "Creating wrapper script..."
    
    cat > "$BIN_DIR/flowhunt" << 'EOF'
#!/bin/bash
# FlowHunt Toolkit Wrapper Script

# Find the Python executable
if command -v python3 >/dev/null 2>&1; then
    PYTHON_CMD="python3"
elif command -v python >/dev/null 2>&1; then
    PYTHON_CMD="python"
else
    echo "Error: Python not found" >&2
    exit 1
fi

# Execute the flowhunt CLI
exec $PYTHON_CMD -m flowhunt_toolkit.cli "$@"
EOF

    chmod +x "$BIN_DIR/flowhunt"
    print_success "Wrapper script created at $BIN_DIR/flowhunt"
}

# Function to update PATH
update_path() {
    print_status "Updating PATH..."
    
    # Detect shell
    SHELL_NAME=$(basename "$SHELL")
    
    case $SHELL_NAME in
        "bash")
            SHELL_RC="$HOME/.bashrc"
            ;;
        "zsh")
            SHELL_RC="$HOME/.zshrc"
            ;;
        "fish")
            SHELL_RC="$HOME/.config/fish/config.fish"
            ;;
        *)
            SHELL_RC="$HOME/.profile"
            ;;
    esac

    # Add to PATH if not already present
    if ! echo "$PATH" | grep -q "$BIN_DIR"; then
        if [ "$SHELL_NAME" = "fish" ]; then
            echo "set -gx PATH $BIN_DIR \$PATH" >> "$SHELL_RC"
        else
            echo "export PATH=\"$BIN_DIR:\$PATH\"" >> "$SHELL_RC"
        fi
        print_success "Added $BIN_DIR to PATH in $SHELL_RC"
    else
        print_success "$BIN_DIR already in PATH"
    fi

    # Also add to current session
    export PATH="$BIN_DIR:$PATH"
}

# Function to verify installation
verify_installation() {
    print_status "Verifying installation..."
    
    if [ -x "$BIN_DIR/flowhunt" ]; then
        print_success "FlowHunt Toolkit installed successfully!"
        echo
        echo -e "${GREEN}Installation Complete!${NC}"
        echo
        echo "To get started:"
        echo "  1. Restart your terminal or run: source ~/.bashrc (or ~/.zshrc)"
        echo "  2. Run: flowhunt --help"
        echo "  3. Authenticate: flowhunt auth"
        echo
        echo "For more information, visit: $REPO_URL"
        echo
        
        # Try to run the command
        if "$BIN_DIR/flowhunt" --version >/dev/null 2>&1; then
            print_success "✓ flowhunt command is working"
        else
            print_warning "Installation completed but command test failed. You may need to restart your terminal."
        fi
    else
        print_error "Installation failed. Please check the logs above."
        exit 1
    fi
}

# Function to cleanup on error
cleanup() {
    if [ $? -ne 0 ]; then
        print_error "Installation failed. Cleaning up..."
        rm -rf "$INSTALL_DIR" 2>/dev/null || true
        rm -f "$BIN_DIR/flowhunt" 2>/dev/null || true
    fi
}

# Main installation function
main() {
    echo -e "${BLUE}"
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║                    FlowHunt Toolkit Installer                ║"
    echo "║                                                              ║"
    echo "║  This script will install FlowHunt Toolkit on your system   ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
    echo

    # Set up error handling
    trap cleanup EXIT

    # Check prerequisites
    print_status "Checking system requirements..."
    
    # Check if running as root (not recommended)
    if [ "$EUID" -eq 0 ]; then
        print_warning "Running as root is not recommended. Installing for root user..."
        BIN_DIR="/usr/local/bin"
        INSTALL_DIR="/opt/flowhunt-toolkit"
    fi

    # Install system dependencies if needed
    if ! command_exists git || ! command_exists curl; then
        print_status "Installing system dependencies..."
        install_system_deps
    fi

    # Check Python and pip
    check_python
    check_pip

    # Create directories
    create_directories

    # Install FlowHunt Toolkit
    install_flowhunt

    # Create wrapper script
    create_wrapper

    # Update PATH
    update_path

    # Verify installation
    verify_installation

    # Clear error trap on success
    trap - EXIT
}

# Run main function
main "$@"
