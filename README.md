# Sahha MCP Server

## Installation
```
# Create a new directory for our project
uv init sahha-mcp
cd sahha-mcp

# Create virtual environment and activate it
uv venv
.venv\Scripts\activate

# Install dependencies
uv add mcp[cli] httpx 
```


## Coverage
Taken swagger documentation from [here](https://api.sahha.ai/api-docs/index.html)

## Usage
Example with claude desktop config 

```
 "sahha_mcp": {
            "command": "uv",
            "args": [
                "--directory",
                "\path\to\mcp\folder",
                "run",
                "sahha_mcp.py"
            ]
        }
```
