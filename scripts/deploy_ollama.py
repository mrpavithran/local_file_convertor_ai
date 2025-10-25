#!/usr/bin/env python3
"""
Ollama deployment and setup script for AI File System.
"""

import os
import sys
import subprocess
import requests
import time
from pathlib import Path
from typing import Dict, List, Optional
import yaml

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from rich.table import Table


def deploy_ollama(install_locally: bool = True, model: str = "mistral") -> bool:
    """
    Deploy and configure Ollama for AI File System.
    
    Args:
        install_locally: Whether to install Ollama locally
        model: Default model to download
        
    Returns:
        True if deployment successful
    """
    console = Console()
    
    console.print(Panel.fit(
        "🚀 Ollama Deployment for AI File System",
        style="bold blue"
    ))
    
    try:
        # Check if Ollama is already running
        if check_ollama_running():
            console.print("✅ Ollama is already running")
            return setup_ollama_model(model)
        
        # Install Ollama if requested
        if install_locally and not is_ollama_installed():
            console.print("📥 Installing Ollama...")
            if not install_ollama():
                console.print("❌ Failed to install Ollama")
                return False
        
        # Start Ollama service
        console.print("🔧 Starting Ollama service...")
        if not start_ollama_service():
            console.print("❌ Failed to start Ollama service")
            return False
        
        # Wait for service to be ready
        console.print("⏳ Waiting for Ollama to be ready...")
        if not wait_for_ollama_ready():
            console.print("❌ Ollama service didn't start properly")
            return False
        
        # Setup default model
        return setup_ollama_model(model)
        
    except Exception as e:
        console.print(f"❌ Deployment failed: {e}")
        return False


def check_ollama_running() -> bool:
    """Check if Ollama service is running."""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        return response.status_code == 200
    except:
        return False


def is_ollama_installed() -> bool:
    """Check if Ollama is installed."""
    try:
        # Check if ollama command exists
        result = subprocess.run(["ollama", "--version"], 
                              capture_output=True, text=True)
        return result.returncode == 0
    except:
        return False


def install_ollama() -> bool:
    """Install Ollama based on the operating system."""
    import platform
    
    system = platform.system().lower()
    console = Console()
    
    try:
        if system == "linux":
            console.print("🐧 Installing Ollama on Linux...")
            # Use the official install script
            install_cmd = "curl -fsSL https://ollama.ai/install.sh | sh"
            result = subprocess.run(install_cmd, shell=True, capture_output=True, text=True)
            
        elif system == "darwin":  # macOS
            console.print("🍎 Installing Ollama on macOS...")
            # Use Homebrew or direct download
            try:
                result = subprocess.run(["brew", "install", "ollama"], 
                                      capture_output=True, text=True)
            except:
                # Fallback to curl installation
                install_cmd = "curl -fsSL https://ollama.ai/install.sh | sh"
                result = subprocess.run(install_cmd, shell=True, capture_output=True, text=True)
                
        elif system == "windows":
            console.print("🪟 Installing Ollama on Windows...")
            # Download and run installer
            download_url = "https://ollama.ai/download/OllamaSetup.exe"
            download_path = Path.home() / "Downloads" / "OllamaSetup.exe"
            
            # Download installer
            response = requests.get(download_url)
            with open(download_path, 'wb') as f:
                f.write(response.content)
            
            # Run installer
            result = subprocess.run([str(download_path), "/S"], capture_output=True, text=True)
            
        else:
            console.print(f"❌ Unsupported operating system: {system}")
            return False
        
        return result.returncode == 0
        
    except Exception as e:
        console.print(f"❌ Installation failed: {e}")
        return False


def start_ollama_service() -> bool:
    """Start Ollama service."""
    import platform
    
    system = platform.system().lower()
    
    try:
        if system == "linux" or system == "darwin":
            # Start ollama service in background
            subprocess.Popen(["ollama", "serve"], 
                           stdout=subprocess.DEVNULL, 
                           stderr=subprocess.DEVNULL)
        elif system == "windows":
            # On Windows, the installer should have set up the service
            subprocess.Popen(["ollama", "serve"],
                           stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL)
        
        return True
    except Exception as e:
        console = Console()
        console.print(f"❌ Failed to start Ollama: {e}")
        return False


def wait_for_ollama_ready(timeout: int = 60) -> bool:
    """Wait for Ollama service to be ready."""
    console = Console()
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        
        task = progress.add_task("Waiting for Ollama...", total=timeout)
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            if check_ollama_running():
                progress.update(task, completed=timeout)
                console.print("✅ Ollama is ready!")
                return True
            
            time.sleep(2)
            progress.update(task, advance=2)
    
    console.print("❌ Ollama didn't start within timeout period")
    return False


def setup_ollama_model(model: str = "mistral") -> bool:
    """Download and setup the specified Ollama model."""
    console = Console()
    
    console.print(f"📥 Setting up model: {model}")
    
    try:
        # Check if model already exists
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code == 200:
            models = response.json().get('models', [])
            existing_models = [m['name'] for m in models]
            
            if any(model in name for name in existing_models):
                console.print(f"✅ Model {model} already exists")
                return True
        
        # Download model
        console.print(f"📥 Downloading {model} model (this may take a while)...")
        
        with Progress() as progress:
            task = progress.add_task(f"Downloading {model}...", total=100)
            
            # Use subprocess to get real-time progress
            process = subprocess.Popen(
                ["ollama", "pull", model],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True
            )
            
            # Monitor progress
            for line in process.stdout:
                if "downloading" in line.lower() or "pulling" in line.lower():
                    progress.update(task, advance=1)
                console.print(f"    {line.strip()}")
            
            process.wait()
        
        if process.returncode == 0:
            console.print(f"✅ Model {model} downloaded successfully")
            
            # Update configuration
            update_model_config(model)
            return True
        else:
            console.print(f"❌ Failed to download model {model}")
            return False
            
    except Exception as e:
        console.print(f"❌ Model setup failed: {e}")
        return False


def update_model_config(model: str):
    """Update system configuration with the chosen model."""
    config_path = Path(__file__).parent.parent / "config" / "system_config.yaml"
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Update AI configuration
        if 'system' not in config:
            config['system'] = {}
        
        config['system']['default_ai_model'] = model
        config['system']['ai_provider'] = 'ollama'
        
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        console = Console()
        console.print(f"✅ Configuration updated with model: {model}")
        
    except Exception as e:
        console = Console()
        console.print(f"⚠️  Could not update configuration: {e}")


def list_available_models() -> List[str]:
    """List available Ollama models."""
    try:
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code == 200:
            models = response.json().get('models', [])
            return [model['name'] for model in models]
        return []
    except:
        return []


def main():
    """Main function for standalone execution."""
    console = Console()
    
    parser = argparse.ArgumentParser(description="Deploy Ollama for AI File System")
    parser.add_argument("--model", default="mistral", help="Model to download")
    parser.add_argument("--no-install", action="store_true", help="Skip installation")
    parser.add_argument("--list-models", action="store_true", help="List available models")
    
    args = parser.parse_args()
    
    if args.list_models:
        models = list_available_models()
        if models:
            table = Table(title="Available Ollama Models")
            table.add_column("Model Name", style="bold")
            
            for model in models:
                table.add_row(model)
            
            console.print(table)
        else:
            console.print("No models found or Ollama not running")
        return
    
    success = deploy_ollama(
        install_locally=not args.no_install,
        model=args.model
    )
    
    if success:
        console.print("🎉 Ollama deployment completed successfully!")
        sys.exit(0)
    else:
        console.print("💥 Ollama deployment failed!")
        sys.exit(1)


if __name__ == "__main__":
    import argparse
    main()