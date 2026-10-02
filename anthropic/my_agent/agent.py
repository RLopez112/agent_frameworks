import sys
from pathlib import Path

# Ensure parent directory is in sys.path so agent.py is accessible
parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from agent import (
    Agent,
    AnthropicModel,
    api_base_url,
    model_name_at_endpoint,
    model,
    root_agent,
    main,
)

if __name__ == "__main__":
    main()
