"""
Agent service for managing AI agents with MCP server connections.

This service manages multiple AI agents, each with their own models,
system prompts, and MCP server connections.
"""

import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.ollama import OllamaChatCompletion
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
from semantic_kernel.connectors.ai.ollama.ollama_prompt_execution_settings import OllamaChatPromptExecutionSettings
from semantic_kernel.contents.chat_history import ChatHistory
from semantic_kernel.connectors.mcp import MCPStreamableHttpPlugin

from src.config.agent_config import agent_settings, AgentConfig, MCPServerConfig
from src.utils.logging import get_application_logger

logger = get_application_logger('agent_service')


@dataclass
class AgentInstance:
    """Container for an agent instance with its kernel and components"""
    name: str
    kernel: Kernel
    chat_completion: OllamaChatCompletion
    execution_settings: OllamaChatPromptExecutionSettings
    chat_history: ChatHistory
    mcp_plugins: Dict[str, MCPStreamableHttpPlugin]


class AgentService:
    """Service for managing multiple AI agents with MCP server connections"""
    
    def __init__(self):
        self.agents: Dict[str, AgentInstance] = {}
        self.mcp_servers: Dict[str, MCPStreamableHttpPlugin] = {}
        logger.info("Agent service initialized")
    
    async def initialize(self):
        """Initialize MCP servers and create default agents"""
        try:
            # Initialize MCP servers
            await self._initialize_mcp_servers()
            
            # Create default agents
            for agent_name, agent_config in agent_settings.agents.items():
                if agent_config.enabled:
                    await self.create_agent(agent_name, agent_config)
            
            logger.info(f"Agent service initialized with {len(self.agents)} agents")
            
        except Exception as e:
            logger.error(f"Failed to initialize agent service: {e}")
            raise
    
    async def _initialize_mcp_servers(self):
        """Initialize all configured MCP servers"""
        for server_name, server_config in agent_settings.mcp_servers.items():
            if server_config.enabled:
                try:
                    mcp_plugin = MCPStreamableHttpPlugin(
                        name=server_config.name,
                        url=server_config.url
                    )
                    
                    await mcp_plugin.connect()
                    available_tools = await mcp_plugin.session.list_tools()
                    
                    self.mcp_servers[server_name] = mcp_plugin
                    logger.info(f"Connected to MCP server '{server_name}' with {len(available_tools)} tools")
                    
                except Exception as e:
                    logger.error(f"Failed to connect to MCP server '{server_name}': {e}")
    
    async def create_agent(self, agent_name: str, config: AgentConfig) -> AgentInstance:
        """Create a new agent instance with the given configuration"""
        try:
            # Initialize kernel
            kernel = Kernel()
            
            # Create chat completion service
            chat_completion = OllamaChatCompletion(
                ai_model_id=config.model_id,
                host=config.host,
            )
            kernel.add_service(chat_completion)
            
            # Add MCP plugins to kernel
            agent_mcp_plugins = {}
            for mcp_server_name in config.mcp_servers:
                if mcp_server_name in self.mcp_servers:
                    mcp_plugin = self.mcp_servers[mcp_server_name]
                    kernel.add_plugin(mcp_plugin)
                    agent_mcp_plugins[mcp_server_name] = mcp_plugin
                    logger.debug(f"Added MCP server '{mcp_server_name}' to agent '{agent_name}'")
            
            # Configure execution settings
            execution_settings = OllamaChatPromptExecutionSettings()
            execution_settings.function_choice_behavior = FunctionChoiceBehavior.Auto()
            
            # Initialize chat history with system prompt if provided
            chat_history = ChatHistory()
            if config.system_prompt:
                chat_history.add_system_message(config.system_prompt)
            
            # Create agent instance
            agent_instance = AgentInstance(
                name=agent_name,
                kernel=kernel,
                chat_completion=chat_completion,
                execution_settings=execution_settings,
                chat_history=chat_history,
                mcp_plugins=agent_mcp_plugins
            )
            
            self.agents[agent_name] = agent_instance
            logger.info(f"Created agent '{agent_name}' with model '{config.model_id}'")
            
            return agent_instance
            
        except Exception as e:
            logger.error(f"Failed to create agent '{agent_name}': {e}")
            raise
    
    async def process_transcription(self, agent_name: str, transcription_text: str) -> Optional[str]:
        """
        Process a transcription with the specified agent
        
        Args:
            agent_name: Name of the agent to use
            transcription_text: The transcription text to process
            
        Returns:
            Agent's response or None if failed
        """
        if agent_name not in self.agents:
            logger.error(f"Agent '{agent_name}' not found")
            return None
        
        agent = self.agents[agent_name]
        
        try:
            # Add transcription to chat history
            message = f"New transcription received: {transcription_text}"
            agent.chat_history.add_user_message(message)
            
            # Get response from agent
            result = await agent.chat_completion.get_chat_message_content(
                chat_history=agent.chat_history,
                settings=agent.execution_settings,
                kernel=agent.kernel,
            )
            
            # Add response to chat history
            agent.chat_history.add_message(result)
            
            # Trim chat history if too long
            await self._trim_chat_history(agent)
            
            response = str(result)
            logger.info(f"Agent '{agent_name}' processed transcription: {len(transcription_text)} chars -> {len(response)} chars")
            
            return response
            
        except Exception as e:
            logger.error(f"Failed to process transcription with agent '{agent_name}': {e}")
            return None
    
    async def send_message(self, agent_name: str, message: str) -> Optional[str]:
        """
        Send a direct message to an agent
        
        Args:
            agent_name: Name of the agent to use
            message: Message to send
            
        Returns:
            Agent's response or None if failed
        """
        if agent_name not in self.agents:
            logger.error(f"Agent '{agent_name}' not found")
            return None
        
        agent = self.agents[agent_name]
        
        try:
            # Add message to chat history
            agent.chat_history.add_user_message(message)
            
            # Get response from agent
            result = await agent.chat_completion.get_chat_message_content(
                chat_history=agent.chat_history,
                settings=agent.execution_settings,
                kernel=agent.kernel,
            )
            
            # Add response to chat history
            agent.chat_history.add_message(result)
            
            # Trim chat history if too long
            await self._trim_chat_history(agent)
            
            response = str(result)
            logger.info(f"Agent '{agent_name}' responded to message: {len(message)} chars -> {len(response)} chars")
            
            return response
            
        except Exception as e:
            logger.error(f"Failed to send message to agent '{agent_name}': {e}")
            return None
    
    async def _trim_chat_history(self, agent: AgentInstance):
        """Trim chat history if it exceeds the maximum length"""
        max_length = agent_settings.max_chat_history
        
        if len(agent.chat_history.messages) > max_length:
            # Keep system message (if any) and recent messages
            messages = agent.chat_history.messages
            system_messages = [msg for msg in messages if hasattr(msg, 'role') and msg.role == 'system']
            recent_messages = messages[-(max_length - len(system_messages)):]
            
            # Create new chat history
            new_history = ChatHistory()
            for msg in system_messages + recent_messages:
                new_history.add_message(msg)
            
            agent.chat_history = new_history
            logger.debug(f"Trimmed chat history for agent '{agent.name}' to {len(new_history.messages)} messages")
    
    def get_agent_names(self) -> List[str]:
        """Get list of available agent names"""
        return list(self.agents.keys())
    
    def get_agent_info(self, agent_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific agent"""
        if agent_name not in self.agents:
            return None
        
        agent = self.agents[agent_name]
        config = agent_settings.agents.get(agent_name)
        
        return {
            "name": agent.name,
            "model_id": config.model_id if config else "unknown",
            "host": config.host if config else "unknown",
            "mcp_servers": list(agent.mcp_plugins.keys()),
            "chat_history_length": len(agent.chat_history.messages)
        }
    
    async def shutdown(self):
        """Shutdown the agent service and close all connections"""
        try:
            # Close MCP server connections
            for server_name, mcp_plugin in self.mcp_servers.items():
                try:
                    await mcp_plugin.close()
                    logger.info(f"Closed MCP server connection: {server_name}")
                except Exception as e:
                    logger.error(f"Error closing MCP server '{server_name}': {e}")
            
            # Clear agents
            self.agents.clear()
            self.mcp_servers.clear()
            
            logger.info("Agent service shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during agent service shutdown: {e}")


# Global instance
agent_service = AgentService()