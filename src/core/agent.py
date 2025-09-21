import logging
from semantic_kernel.connectors.mcp import MCPStreamableHttpPlugin
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.ollama import OllamaChatCompletion
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
from semantic_kernel.contents.chat_history import ChatHistory

from semantic_kernel.connectors.ai.ollama.ollama_prompt_execution_settings import OllamaChatPromptExecutionSettings

import json
import httpx
from functools import wraps

# Store the original request method BEFORE we replace it
_original_request = httpx.AsyncClient.request

async def logged_request(self, method, url, **kwargs):
    logging.info(f"\n=== HTTP REQUEST ===")
    logging.info(f"Method: {method}")
    logging.info(f"URL: {url}")
    
    # Log headers
    headers = kwargs.get('headers', {})
    logging.info(f"Headers: {dict(headers) if headers else 'None'}")
    
    # Log request body
    content = kwargs.get('content')
    json_data = kwargs.get('json')
    data = kwargs.get('data')
    
    if content:
        try:
            # Try to parse as JSON for pretty printing
            if isinstance(content, (str, bytes)):
                content_str = content.decode() if isinstance(content, bytes) else content
                parsed = json.loads(content_str)
                logging.info(f"Request Body (JSON):\n{json.dumps(parsed, indent=2)}")
            else:
                logging.info(f"Request Body (Raw): {content}")
        except:
            logging.info(f"Request Body (Raw): {content}")
    elif json_data:
        logging.info(f"Request Body (JSON):\n{json.dumps(json_data, indent=2)}")
    elif data:
        logging.info(f"Request Body (Data): {data}")
    else:
        logging.info("Request Body: None")
    
    logging.info(f"=== END REQUEST ===\n")
    
    # Call the ACTUAL original method, not the monkey-patched one
    return await _original_request(self, method, url, **kwargs)

# Apply the monkey patch
httpx.AsyncClient.request = logged_request
logging.info("HTTP request logging enabled!")


class Action_Agent:
    def __init__(self):
        # Initialize basic components
        self.kernel = None
        self.mcp_server = None
        self.chat_completion = None
        self.execution_settings = None
        self.history = ChatHistory(system_message="you review text and decide if action is needed. " \
        "you should always use your tools to get a name from that tool, " \
        "then respond with the name you retrieved with  and '<somename>: action required' or '<somename>: no action required'")

    async def connect_mcp(self):
        """Connect to the MCP server"""
        self.mcp_server = MCPStreamableHttpPlugin(
            name="test",
            url="http://localhost:8001/mcp",
        )
        
        try:
            await self.mcp_server.connect()
            available_tools = await self.mcp_server.session.list_tools()
            logging.info(f"Successfully connected to MCP server: <{self.mcp_server.name}> with tools {available_tools}")
            # await self.mcp_server.close()
        except Exception as e:
            logging.error(f"Error connecting to {self.mcp_server.name} MCP server: {e}")

    async def initialize_kernel(self):
        """Initialize the kernel and all its components - must be called after __init__"""
        # Initialize the kernel
        self.kernel = Kernel()
        
        # Connect to MCP server first
        await self.connect_mcp()

        # Add Ollama chat completion
        self.chat_completion = OllamaChatCompletion(
            # ai_model_id="qwen3:0.6b",
            ai_model_id="gpt-oss:20b",
            host="http://ollama.home",
        )

        self.kernel.add_service(self.chat_completion)

        # Add MCP plugin if connection was successful
        if self.mcp_server:
            self.kernel.add_plugin(self.mcp_server)

        # Enable planning
        self.execution_settings = OllamaChatPromptExecutionSettings()
        self.execution_settings.function_choice_behavior = FunctionChoiceBehavior.Auto()

    async def review_text(self, text):
        """Process the input text and return the agent's response.
        Args:
            text (str): The input text to be processed.
        Returns:
            str: The response from the agent.
        """
        
        # Ensure kernel is initialized
        if self.kernel is None:
            await self.initialize_kernel()

        # Add user input to the history
        self.history.add_user_message(text)

        # Get the response from the AI
        result = await self.chat_completion.get_chat_message_content(
            chat_history=self.history,
            settings=self.execution_settings,
            kernel=self.kernel,
        )

        # Add the message from the agent to the chat history
        self.history.add_message(result)

        return result

