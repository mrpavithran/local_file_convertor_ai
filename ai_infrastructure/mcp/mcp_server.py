"""
Enhanced Model Context Protocol (MCP) server using FastAPI.
Exposes endpoints to list tools and run tools via HTTP POST with advanced features.
"""
import asyncio
import importlib
import traceback
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, validator
import uvicorn

from .tool_registry import ToolRegistry

# Security scheme
security = HTTPBearer(auto_error=False)

class RunRequest(BaseModel):
    tool: str = Field(..., description="Name of the tool to execute")
    args: Dict[str, Any] = Field(default={}, description="Tool arguments")
    timeout_seconds: int = Field(default=30, ge=1, le=300, description="Execution timeout in seconds")
    request_id: Optional[str] = Field(default=None, description="Optional request ID for tracking")

    @validator('args')
    def validate_args(cls, v):
        """Ensure args is a dictionary."""
        if not isinstance(v, dict):
            raise ValueError('args must be a dictionary')
        return v

class RunResponse(BaseModel):
    ok: bool = Field(..., description="Whether the execution was successful")
    result: Optional[Any] = Field(default=None, description="Tool execution result")
    error: Optional[str] = Field(default=None, description="Error message if execution failed")
    trace: Optional[str] = Field(default=None, description="Stack trace if execution failed")
    request_id: Optional[str] = Field(default=None, description="Request ID for tracking")
    execution_time: Optional[float] = Field(default=None, description="Execution time in seconds")

class ToolInfo(BaseModel):
    name: str = Field(..., description="Tool name")
    description: str = Field(..., description="Tool description")
    parameters: Dict[str, Any] = Field(..., description="Tool parameter schema")
    is_async: bool = Field(..., description="Whether the tool supports async execution")
    timeout: int = Field(..., description="Default timeout in seconds")

class HealthResponse(BaseModel):
    ok: bool = Field(..., description="Server health status")
    tools_registered: List[str] = Field(..., description="List of registered tool names")
    server_version: str = Field(..., description="Server version")
    uptime: float = Field(..., description="Server uptime in seconds")

# Global variables
registry: Optional[ToolRegistry] = None
startup_time: float = 0

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global registry, startup_time
    import time
    startup_time = time.time()
    
    # Initialize tool registry
    registry = ToolRegistry()
    
    # Auto-discover and register tools
    await registry.auto_discover_tools()
    
    yield  # Server runs here
    
    # Shutdown
    if registry:
        await registry.cleanup()

app = FastAPI(
    title="MCP Server",
    description="Model Context Protocol Server for tool execution",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dependency for authentication
async def verify_auth(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)):
    """Verify authentication token."""
    # Implement your authentication logic here
    # For now, this is a placeholder that allows all requests
    if credentials:
        # Validate token here
        pass
    return True

# Dependency for tool registry
async def get_registry():
    if registry is None:
        raise HTTPException(status_code=503, detail="Tool registry not initialized")
    return registry

@app.get("/healthz", response_model=HealthResponse)
async def health():
    """Health check endpoint with detailed server information."""
    import time
    tools = list(registry.list_tools().keys()) if registry else []
    return {
        "ok": True,
        "tools_registered": tools,
        "server_version": "1.0.0",
        "uptime": time.time() - startup_time
    }

@app.get("/tools", response_model=Dict[str, ToolInfo])
async def list_tools(
    registry: ToolRegistry = Depends(get_registry),
    auth: bool = Depends(verify_auth)
):
    """List all available tools with detailed information."""
    tools = registry.list_tools()
    tool_info = {}
    
    for name, tool in tools.items():
        tool_info[name] = ToolInfo(
            name=name,
            description=tool.description,
            parameters=tool.parameters_schema,
            is_async=tool.is_async,
            timeout=tool.timeout
        )
    
    return tool_info

@app.post("/run", response_model=RunResponse)
async def run_tool(
    req: RunRequest,
    background_tasks: BackgroundTasks,
    registry: ToolRegistry = Depends(get_registry),
    auth: bool = Depends(verify_auth)
):
    """Execute a tool with the provided arguments."""
    import time
    
    # Generate request ID if not provided
    request_id = req.request_id or str(uuid4())
    
    # Validate tool exists
    if req.tool not in registry.list_tools():
        raise HTTPException(
            status_code=404, 
            detail=f"Tool '{req.tool}' not found. Available tools: {list(registry.list_tools().keys())}"
        )
    
    start_time = time.time()
    
    try:
        tool = registry.get_tool(req.tool)
        
        # Validate arguments against tool schema
        validation_error = tool.validate_arguments(req.args)
        if validation_error:
            raise HTTPException(status_code=400, detail=f"Invalid arguments: {validation_error}")
        
        # Execute tool with timeout
        if tool.is_async:
            result = await asyncio.wait_for(
                tool.run(**req.args),
                timeout=req.timeout_seconds
            )
        else:
            # Run synchronous tools in thread pool
            result = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    None, tool.run, **req.args
                ),
                timeout=req.timeout_seconds
            )
        
        execution_time = time.time() - start_time
        
        return RunResponse(
            ok=True,
            result=result,
            request_id=request_id,
            execution_time=execution_time
        )
        
    except asyncio.TimeoutError:
        execution_time = time.time() - start_time
        raise HTTPException(
            status_code=408,
            detail=f"Tool execution timed out after {req.timeout_seconds} seconds"
        )
    except HTTPException:
        raise
    except Exception as e:
        execution_time = time.time() - start_time
        tb = traceback.format_exc()
        
        # Log the error (in production, use proper logging)
        print(f"Error executing tool {req.tool}: {str(e)}")
        print(f"Traceback: {tb}")
        
        return RunResponse(
            ok=False,
            error=str(e),
            trace=tb,
            request_id=request_id,
            execution_time=execution_time
        )

@app.get("/tools/{tool_name}/schema")
async def get_tool_schema(
    tool_name: str,
    registry: ToolRegistry = Depends(get_registry),
    auth: bool = Depends(verify_auth)
):
    """Get the JSON schema for a specific tool's parameters."""
    if tool_name not in registry.list_tools():
        raise HTTPException(status_code=404, detail=f"Tool '{tool_name}' not found")
    
    tool = registry.get_tool(tool_name)
    return {
        "name": tool_name,
        "description": tool.description,
        "parameters": tool.parameters_schema,
        "required": tool.required_parameters,
        "is_async": tool.is_async,
        "timeout": tool.timeout
    }

@app.post("/batch-run")
async def batch_run_tools(
    requests: List[RunRequest],
    registry: ToolRegistry = Depends(get_registry),
    auth: bool = Depends(verify_auth)
):
    """Execute multiple tools in batch."""
    results = []
    
    for req in requests:
        try:
            # Use the existing run_tool logic for each request
            response = await run_tool(req, BackgroundTasks(), registry, auth)
            results.append(response.dict())
        except HTTPException as e:
            results.append(RunResponse(
                ok=False,
                error=e.detail,
                request_id=req.request_id
            ).dict())
    
    return {"results": results}

# Error handlers
@app.exception_handler(500)
async def internal_server_error_handler(request, exc):
    """Handle internal server errors."""
    return JSONResponse(
        status_code=500,
        content={
            "ok": False,
            "error": "Internal server error",
            "request_id": getattr(request.state, 'request_id', 'unknown')
        }
    )

# Background task for cleanup
async def cleanup_old_requests():
    """Background task to clean up old request data."""
    # Implement cleanup logic here
    pass

if __name__ == "__main__":
    # Configuration from environment variables
    import os
    
    host = os.getenv("MCP_HOST", "0.0.0.0")
    port = int(os.getenv("MCP_PORT", "8765"))
    reload = os.getenv("MCP_RELOAD", "false").lower() == "true"
    
    uvicorn.run(
        "mcp_server:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )