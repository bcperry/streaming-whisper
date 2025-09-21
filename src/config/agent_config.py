"""
Agent configuration management for streaming-whisper application.

This module provides configuration classes for managing agent settings
including AI models, MCP servers, and system prompts.
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class MCPServerConfig(BaseModel):
    """Configuration for an MCP server connection"""
    name: str = Field(description="Unique name for the MCP server")
    url: str = Field(description="URL endpoint for the MCP server")
    enabled: bool = Field(default=True, description="Whether this MCP server is enabled")


class AgentConfig(BaseModel):
    """Configuration for a single agent"""
    name: str = Field(description="Unique name for the agent")
    model_id: str = Field(description="AI model identifier (e.g., 'gpt-oss:20b')")
    host: str = Field(description="Host URL for the AI service (e.g., 'http://ollama.home')")
    system_prompt: str = Field(default="You are a helpful assistant.", description="System prompt for the agent")
    mcp_servers: List[str] = Field(default_factory=list, description="List of MCP server names to use")
    enabled: bool = Field(default=True, description="Whether this agent is enabled")


class AgentSettings(BaseSettings):
    """Agent service configuration"""
    default_function_choice_behavior: str = Field(default="Auto", description="Default function choice behavior")
    max_chat_history: int = Field(default=100, description="Maximum chat history length to maintain")
    
    # Default MCP servers
    mcp_servers: Dict[str, MCPServerConfig] = Field(
        default_factory=dict,
        description="Available MCP servers"
    )
    
    # Default agents
    agents: Dict[str, AgentConfig] = Field(
        default_factory=lambda: {
            "default": AgentConfig(
                name="default",
                model_id="gpt-oss:20b",
                host="http://ollama.home",
                system_prompt="You are a helpful assistant.",
                mcp_servers=[]
            )
        },
        description="Available agent configurations"
    )
    
    class Config:
        env_prefix = "AGENT_"


# Global instance
agent_settings = AgentSettings()