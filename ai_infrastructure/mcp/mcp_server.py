"""
Enhanced MCP Server implementation using FastAPI
Provides a complete Model Context Protocol server with advanced features
"""

import asyncio
import inspect
import json
import uuid
from typing import Any, Dict, List, Optional, Callable, Union, Awaitable
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator

from .tool_registry import ToolRegistry
from .utils import (
    validate_tool_definition,
    safe_execute_tool,
    format_tool_result,
    create_error_response,
    logger
)


# Pydantic Models for MCP Protocol
class ToolDefinition(BaseModel):
    """Tool definition schema"""
    name: str = Field(..., description="Unique name of the tool")
    description: str = Field(..., description="Detailed description of the tool")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Tool parameters schema")
    returns: Dict[str, Any] = Field(default_factory=dict, description="Return value schema")
    category: str = Field("general", description="Tool category for organization")
    version: str = Field("1.0.0", description="Tool version")
    deprecated: bool = Field(False, description="Whether the tool is deprecated")


class ToolCall(BaseModel):
    """Tool call request schema"""
    tool_name: str = Field(..., description="Name of the tool to call")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Tool parameters")
    call_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique call identifier")
    timeout: Optional[int] = Field(None, description="Execution timeout in seconds")


class ToolResult(BaseModel):
    """Tool execution result schema"""
    call_id: str = Field(..., description="Unique call identifier")
    tool_name: str = Field(..., description="Name of the executed tool")
    result: Any = Field(..., description="Tool execution result")
    success: bool = Field(..., description="Whether execution was successful")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    execution_time: float = Field(..., description="Execution time in seconds")
    timestamp: str = Field(..., description="Execution timestamp")


class HealthResponse(BaseModel):
    """Health check response schema"""
    status: str = Field(..., description="Server status")
    version: str = Field(..., description="Server version")
    uptime: float = Field(..., description="Server uptime in seconds")
    tools_registered: int = Field(..., description="Number of registered tools")
    active_connections: int = Field(..., description="Number of active connections")


class MCPServer:
    """
    Enhanced MCP Server with advanced features
    """
    
    def __init__(
        self,
        name: str = "MCP Server",
        version: str = "1.0.0",
        description: str = "Enhanced MCP Server",
        host: str = "0.0.0.0",
        port: int = 8000,
        enable_cors: bool = True,
        max_workers: int = 10
    ):
        self.name = name
        self.version = version
        self.description = description
        self.host = host
        self.port = port
        self.start_time = datetime.now()
        self.active_connections = 0
        self.max_workers = max_workers
        
        # Initialize tool registry
        self.tool_registry = ToolRegistry()
        
        # Create FastAPI application
        self.app = self._create_fastapi_app(enable_cors)
        
        # Setup routes
        self._setup_routes()
        
        # Background tasks
        self.background_tasks = set()
        
        logger.info(f"Initialized MCP Server: {name} v{version}")

    def _create_fastapi_app(self, enable_cors: bool) -> FastAPI:
        """Create and configure FastAPI application"""
        
        @asynccontextmanager
        async def lifespan(app: FastAPI):
            # Startup
            logger.info(f"Starting {self.name} on {self.host}:{self.port}")
            yield
            # Shutdown
            logger.info(f"Shutting down {self.name}")
            # Cancel all background tasks
            for task in self.background_tasks:
                task.cancel()
        
        app = FastAPI(
            title=self.name,
            description=self.description,
            version=self.version,
            lifespan=lifespan
        )
        
        if enable_cors:
            app.add_middleware(
                CORSMiddleware,
                allow_origins=["*"],
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            )
        
        return app

    def _setup_routes(self):
        """Setup all API routes"""
        
        @self.app.get("/", response_model=Dict[str, Any])
        async def root():
            """Root endpoint with server info"""
            return {
                "name": self.name,
                "version": self.version,
                "description": self.description,
                "status": "running"
            }

        @self.app.get("/health", response_model=HealthResponse)
        async def health_check():
            """Health check endpoint"""
            uptime = (datetime.now() - self.start_time).total_seconds()
            return HealthResponse(
                status="healthy",
                version=self.version,
                uptime=uptime,
                tools_registered=len(self.tool_registry.list_tools()),
                active_connections=self.active_connections
            )

        @self.app.get("/tools", response_model=List[ToolDefinition])
        async def list_tools(category: Optional[str] = None):
            """List all available tools, optionally filtered by category"""
            tools = self.tool_registry.list_tools()
            if category:
                tools = [tool for tool in tools if tool.get("category") == category]
            return tools

        @self.app.get("/tools/{tool_name}", response_model=ToolDefinition)
        async def get_tool(tool_name: str):
            """Get specific tool definition"""
            tool = self.tool_registry.get_tool(tool_name)
            if not tool:
                raise HTTPException(status_code=404, detail=f"Tool '{tool_name}' not found")
            return tool

        @self.app.post("/tools/call", response_model=ToolResult)
        async def call_tool(tool_call: ToolCall, background_tasks: BackgroundTasks):
            """Execute a tool with given parameters"""
            self.active_connections += 1
            try:
                start_time = datetime.now()
                
                # Validate tool exists
                if not self.tool_registry.tool_exists(tool_call.tool_name):
                    raise HTTPException(
                        status_code=404, 
                        detail=f"Tool '{tool_call.tool_name}' not found"
                    )
                
                # Execute tool
                result = await self.tool_registry.execute_tool(
                    tool_call.tool_name,
                    tool_call.parameters,
                    timeout=tool_call.timeout
                )
                
                execution_time = (datetime.now() - start_time).total_seconds()
                
                return ToolResult(
                    call_id=tool_call.call_id,
                    tool_name=tool_call.tool_name,
                    result=result,
                    success=True,
                    error_message=None,
                    execution_time=execution_time,
                    timestamp=datetime.now().isoformat()
                )
                
            except HTTPException:
                raise
            except Exception as e:
                execution_time = (datetime.now() - start_time).total_seconds()
                error_msg = f"Tool execution failed: {str(e)}"
                logger.error(error_msg)
                
                return ToolResult(
                    call_id=tool_call.call_id,
                    tool_name=tool_call.tool_name,
                    result=None,
                    success=False,
                    error_message=error_msg,
                    execution_time=execution_time,
                    timestamp=datetime.now().isoformat()
                )
            finally:
                self.active_connections -= 1

        @self.app.post("/tools/batch", response_model=List[ToolResult])
        async def batch_call_tools(tool_calls: List[ToolCall]):
            """Execute multiple tools in batch"""
            self.active_connections += 1
            try:
                results = []
                for tool_call in tool_calls:
                    try:
                        # Execute each tool sequentially (can be enhanced for parallel execution)
                        result = await self.tool_registry.execute_tool(
                            tool_call.tool_name,
                            tool_call.parameters,
                            timeout=tool_call.timeout
                        )
                        
                        results.append(ToolResult(
                            call_id=tool_call.call_id,
                            tool_name=tool_call.tool_name,
                            result=result,
                            success=True,
                            error_message=None,
                            execution_time=0.0,  # Would need individual timing
                            timestamp=datetime.now().isoformat()
                        ))
                    except Exception as e:
                        results.append(ToolResult(
                            call_id=tool_call.call_id,
                            tool_name=tool_call.tool_name,
                            result=None,
                            success=False,
                            error_message=str(e),
                            execution_time=0.0,
                            timestamp=datetime.now().isoformat()
                        ))
                
                return results
            finally:
                self.active_connections -= 1

        @self.app.get("/metrics")
        async def get_metrics():
            """Get server metrics"""
            uptime = (datetime.now() - self.start_time).total_seconds()
            tools = self.tool_registry.list_tools()
            
            return {
                "server": {
                    "name": self.name,
                    "version": self.version,
                    "uptime_seconds": uptime,
                    "active_connections": self.active_connections
                },
                "tools": {
                    "total": len(tools),
                    "by_category": self._get_tools_by_category(tools)
                }
            }

    def _get_tools_by_category(self, tools: List[Dict]) -> Dict[str, int]:
        """Count tools by category"""
        categories = {}
        for tool in tools:
            category = tool.get("category", "uncategorized")
            categories[category] = categories.get(category, 0) + 1
        return categories

    def register_tool(
        self,
        name: str,
        function: Callable,
        description: str = "",
        parameters: Optional[Dict[str, Any]] = None,
        returns: Optional[Dict[str, Any]] = None,
        category: str = "general",
        version: str = "1.0.0",
        deprecated: bool = False
    ) -> bool:
        """Register a tool with the server"""
        return self.tool_registry.register_tool(
            name=name,
            function=function,
            description=description,
            parameters=parameters,
            returns=returns,
            category=category,
            version=version,
            deprecated=deprecated
        )

    def register_tools_from_module(self, module) -> int:
        """Register all tools from a module that have @tool decorator"""
        return self.tool_registry.register_tools_from_module(module)

    async def start(self):
        """Start the MCP server"""
        import uvicorn
        
        config = uvicorn.Config(
            self.app,
            host=self.host,
            port=self.port,
            log_level="info"
        )
        server = uvicorn.Server(config)
        
        logger.info(f"Starting {self.name} on {self.host}:{self.port}")
        await server.serve()

    def get_app(self) -> FastAPI:
        """Get the FastAPI application instance"""
        return self.app


# Global MCP Server instance
_default_server: Optional[MCPServer] = None

def create_mcp_server(
    name: str = "MCP Server",
    version: str = "1.0.0",
    description: str = "Enhanced MCP Server",
    host: str = "0.0.0.0",
    port: int = 8000,
    enable_cors: bool = True
) -> MCPServer:
    """Create a global MCP server instance"""
    global _default_server
    _default_server = MCPServer(
        name=name,
        version=version,
        description=description,
        host=host,
        port=port,
        enable_cors=enable_cors
    )
    return _default_server

def get_mcp_server() -> MCPServer:
    """Get the global MCP server instance"""
    global _default_server
    if _default_server is None:
        _default_server = create_mcp_server()
    return _default_server

# FastAPI app instance for direct use
mcp_app = get_mcp_server().get_app()