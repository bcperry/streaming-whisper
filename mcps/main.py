import logging
import random
import time
from fastmcp import FastMCP
from typing import Dict, List, Any, Optional, Union, Literal
from pydantic import BaseModel, Field
from faker import Faker


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create MCP server
mcp = FastMCP("demo_server")

@mcp.tool()
async def get_name() -> str:
    '''This tool returns a name.'''
    return Faker().name()

if __name__ == "__main__":

    mcp.run(transport='streamable-http', host='0.0.0.0', port=8001)
