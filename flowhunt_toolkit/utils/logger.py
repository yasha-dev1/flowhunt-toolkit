"""Colorful logger utility for CLI commands."""

import click
from typing import Optional
from datetime import datetime


class Logger:
    """Logger with colorful output and borders like Claude CLI."""
    
    # Color definitions
    COLORS = {
        'blue': '\033[34m',
        'green': '\033[32m',
        'yellow': '\033[33m',
        'red': '\033[31m',
        'cyan': '\033[36m',
        'magenta': '\033[35m',
        'gray': '\033[90m',
        'bold': '\033[1m',
        'reset': '\033[0m'
    }
    
    def __init__(self, verbose: bool = False):
        """Initialize logger.
        
        Args:
            verbose: Enable verbose output
        """
        self.verbose = verbose
    
    def _format_message(self, message: str, color: str = '', prefix: str = '') -> str:
        """Format message with color and prefix."""
        color_code = self.COLORS.get(color, '')
        reset_code = self.COLORS['reset']
        
        if prefix:
            prefix = f"{self.COLORS['bold']}{prefix}{reset_code} "
        
        return f"{color_code}{prefix}{message}{reset_code}"
    
    def _print_border(self, char: str = '─', length: int = 60, color: str = 'gray'):
        """Print a colored border."""
        border = char * length
        click.echo(self._format_message(border, color))
    
    def info(self, message: str, prefix: str = ''):
        """Print info message."""
        click.echo(self._format_message(message, 'cyan', prefix))
    
    def success(self, message: str, prefix: str = '✓'):
        """Print success message."""
        click.echo(self._format_message(message, 'green', prefix))
    
    def warning(self, message: str, prefix: str = '⚠'):
        """Print warning message."""
        click.echo(self._format_message(message, 'yellow', prefix))
    
    def error(self, message: str, prefix: str = '✗'):
        """Print error message."""
        click.echo(self._format_message(message, 'red', prefix), err=True)
    
    def debug(self, message: str, prefix: str = '→'):
        """Print debug message (only if verbose)."""
        if self.verbose:
            click.echo(self._format_message(message, 'gray', prefix))
    
    def header(self, title: str, subtitle: str = ''):
        """Print a section header with border."""
        self._print_border('━', 60, 'blue')
        title_msg = self._format_message(title, 'blue', '🚀')
        click.echo(title_msg)
        if subtitle:
            click.echo(self._format_message(subtitle, 'gray'))
        self._print_border('─', 60, 'gray')
    
    def section(self, title: str):
        """Print a section divider."""
        click.echo()
        click.echo(self._format_message(title, 'bold'))
        self._print_border('─', len(title), 'gray')
    
    def progress_start(self, message: str):
        """Start a progress indicator."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        msg = f"[{timestamp}] {message}"
        click.echo(self._format_message(msg, 'cyan', '⏳'))
    
    def progress_update(self, message: str):
        """Show progress update."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        msg = f"[{timestamp}] {message}"
        click.echo(self._format_message(msg, 'blue', '•'))
    
    def progress_done(self, message: str, duration: Optional[float] = None):
        """Mark progress as done."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        if duration:
            msg = f"[{timestamp}] {message} (took {duration:.1f}s)"
        else:
            msg = f"[{timestamp}] {message}"
        click.echo(self._format_message(msg, 'green', '✓'))
    
    def stats_table(self, title: str, stats: dict):
        """Display statistics in a formatted table."""
        self.section(title)
        
        max_key_len = max(len(str(k)) for k in stats.keys()) if stats else 0
        
        for key, value in stats.items():
            key_formatted = f"{key:<{max_key_len}}"
            
            # Color-code values based on key
            if 'error' in key.lower() or 'failed' in key.lower():
                value_color = 'red' if value > 0 else 'green'
            elif 'success' in key.lower() or 'completed' in key.lower():
                value_color = 'green'
            elif 'skip' in key.lower():
                value_color = 'yellow'
            else:
                value_color = 'cyan'
            
            key_msg = self._format_message(key_formatted, 'gray')
            value_msg = self._format_message(str(value), value_color, '→')
            click.echo(f"{key_msg} {value_msg}")
    
    def command_start(self, command: str, args: dict):
        """Log command start with arguments."""
        self.header(f"Flowhunt {command.upper()}", "Starting batch execution...")
        
        if args:
            self.section("Configuration")
            for key, value in args.items():
                if value is not None:
                    key_formatted = f"{key.replace('_', ' ').title():<20}"
                    key_msg = self._format_message(key_formatted, 'gray')
                    value_msg = self._format_message(str(value), 'cyan')
                    click.echo(f"{key_msg} {value_msg}")
        click.echo()