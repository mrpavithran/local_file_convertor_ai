"""
Enhanced MCP Server implementation using FastAPI
FIXED VERSION - Fully functional with proper imports and error handling
"""

import os
import sys
import asyncio
import inspect
import json
import uuid
from typing import Any, Dict, List, Optional, Callable, Union, Awaitable
from datetime import datetime
from contextlib import asynccontextmanager
import logging

# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

logger = logging.getLogger(__name__)

# Import with fallbacks
try:
    from fastapi import FastAPI, HTTPException, BackgroundTasks
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    logger.warning("FastAPI not available. MCP server will run in limited mode.")
    
    # Create minimal replacements
    class BaseModel:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
    
    def Field(default=..., description=""):
        return default

    class HTTPException(Exception):
        def __init__(self, status_code, detail):
            self.status_code = status_code
            self.detail = detail

    class BackgroundTasks:
        def add_task(self, func, *args, **kwargs):
            pass

# Import tool registry
try:
    from .tool_registry import ToolRegistry, get_tool_registry
    TOOL_REGISTRY_AVAILABLE = True
except ImportError as e:
    logger.warning(f"ToolRegistry not available: {e}")
    TOOL_REGISTRY_AVAILABLE = False


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
        host: str = "localhost",
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
        if TOOL_REGISTRY_AVAILABLE:
            self.tool_registry = get_tool_registry()
        else:
            self.tool_registry = None
            logger.warning("Tool registry not available - running in limited mode")
        
        # Create FastAPI application if available
        if FASTAPI_AVAILABLE:
            self.app = self._create_fastapi_app(enable_cors)
            self._setup_routes()
        else:
            self.app = None
            logger.warning("FastAPI not available - API endpoints disabled")
        
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
        if not FASTAPI_AVAILABLE or self.app is None:
            return
        
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
            tools_count = len(self.tool_registry.list_tools()) if self.tool_registry else 0
            
            return HealthResponse(
                status="healthy",
                version=self.version,
                uptime=uptime,
                tools_registered=tools_count,
                active_connections=self.active_connections
            )

        @self.app.get("/tools", response_model=List[ToolDefinition])
        async def list_tools(category: Optional[str] = None):
            """List all available tools, optionally filtered by category"""
            if not self.tool_registry:
                raise HTTPException(status_code=503, detail="Tool registry not available")
            
            tools = self.tool_registry.list_tools()
            if category:
                tools = [tool for tool in tools if tool.get("category") == category]
            return tools

        @self.app.get("/tools/{tool_name}", response_model=ToolDefinition)
        async def get_tool(tool_name: str):
            """Get specific tool definition"""
            if not self.tool_registry:
                raise HTTPException(status_code=503, detail="Tool registry not available")
            
            tool = self.tool_registry.get_tool(tool_name)
            if not tool:
                raise HTTPException(status_code=404, detail=f"Tool '{tool_name}' not found")
            return tool

        @self.app.post("/tools/call", response_model=ToolResult)
        async def call_tool(tool_call: ToolCall, background_tasks: BackgroundTasks):
            """Execute a tool with given parameters"""
            if not self.tool_registry:
                raise HTTPException(status_code=503, detail="Tool registry not available")
            
            self.active_connections += 1
            start_time = datetime.now()
            
            try:
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

        @self.app.get("/metrics")
        async def get_metrics():
            """Get server metrics"""
            uptime = (datetime.now() - self.start_time).total_seconds()
            tools = self.tool_registry.list_tools() if self.tool_registry else []
            
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
        if not self.tool_registry:
            logger.error("Cannot register tool - tool registry not available")
            return False
            
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

    def register_simple_tool(self, name: str, func: Callable, description: str = "") -> bool:
        """Register a simple tool without complex configuration"""
        if not self.tool_registry:
            logger.error("Cannot register tool - tool registry not available")
            return False
            
        return self.tool_registry.register_simple_tool(name, func, description)

    def register_tools_from_module(self, module) -> int:
        """Register all tools from a module that have @tool decorator"""
        if not self.tool_registry:
            logger.error("Cannot register tools - tool registry not available")
            return 0
            
        return self.tool_registry.register_tools_from_module(module)

    async def start(self):
        """Start the MCP server"""
        if not FASTAPI_AVAILABLE:
            logger.error("Cannot start server - FastAPI not available")
            return
            
        try:
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
        except ImportError:
            logger.error("Cannot start server - uvicorn not available")
        except Exception as e:
            logger.error(f"Failed to start server: {e}")

    def start_in_thread(self):
        """Start the server in a background thread"""
        if not FASTAPI_AVAILABLE:
            logger.error("Cannot start server - FastAPI not available")
            return
            
        def run_server():
            try:
                import uvicorn
                uvicorn.run(self.app, host=self.host, port=self.port, log_level="info")
            except Exception as e:
                logger.error(f"Server thread failed: {e}")
        
        import threading
        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()
        logger.info(f"Started {self.name} in background thread on {self.host}:{self.port}")
        return server_thread

    def get_app(self) -> Optional[FastAPI]:
        """Get the FastAPI application instance"""
        return self.app

    def get_server_info(self) -> Dict[str, Any]:
        """Get server information"""
        uptime = (datetime.now() - self.start_time).total_seconds()
        tools_count = len(self.tool_registry.list_tools()) if self.tool_registry else 0
        
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "status": "running",
            "uptime_seconds": uptime,
            "tools_registered": tools_count,
            "active_connections": self.active_connections,
            "host": self.host,
            "port": self.port
        }


# Global MCP Server instance
_default_server: Optional[MCPServer] = None

def create_mcp_server(
    name: str = "AI File Converter MCP Server",
    version: str = "1.0.0",
    description: str = "MCP Server for AI File Conversion Tools",
    host: str = "localhost",
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

def initialize_mcp_with_tools() -> bool:
    """Initialize MCP server with basic file conversion tools"""
    try:
        server = get_mcp_server()
        
        # Register basic system tools
        def system_info_tool() -> Dict[str, Any]:
            """Get system information"""
            import platform
            return {
                "platform": platform.platform(),
                "python_version": platform.python_version(),
                "system": platform.system(),
                "processor": platform.processor()
            }
        
        server.register_simple_tool("system_info", system_info_tool, "Get system information")
        
        # Register file tools if available
        try:
            from file_operations.file_manager import FileManager
            file_manager = FileManager()
            
            def list_files_tool(directory: str = ".") -> Dict[str, Any]:
                """List files in a directory"""
                return file_manager.list_files(directory)
            
            server.register_simple_tool("list_files", list_files_tool, "List files in directory")
            
        except ImportError:
            logger.warning("FileManager not available - file tools disabled")
        
        logger.info("MCP server initialized with basic tools")
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize MCP server with tools: {e}")
        return False

# FastAPI app instance for direct use (if available)
if FASTAPI_AVAILABLE:
    mcp_app = get_mcp_server().get_app()
else:
    mcp_app = None