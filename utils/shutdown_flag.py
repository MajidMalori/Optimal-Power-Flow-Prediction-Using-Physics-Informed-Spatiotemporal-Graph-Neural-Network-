"""
Global shutdown flag for graceful signal handling.
Avoids reentrant call issues by using a flag instead of direct printing.
"""

_shutdown_flag = False

def set_shutdown():
    """Set the shutdown flag."""
    global _shutdown_flag
    _shutdown_flag = True

def get_shutdown():
    """Get the shutdown flag."""
    return _shutdown_flag

def reset_shutdown():
    """Reset the shutdown flag (for testing)."""
    global _shutdown_flag
    _shutdown_flag = False

