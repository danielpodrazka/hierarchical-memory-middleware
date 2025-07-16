#!/usr/bin/env python3
"""Example usage of the hierarchical memory middleware with different models."""

import asyncio
import os
from hierarchical_memory_middleware.model_manager import ModelManager
from hierarchical_memory_middleware.config import Config
from hierarchical_memory_middleware.middleware.conversation_manager import HierarchicalConversationManager
from hierarchical_memory_middleware.storage import DuckDBStorage


async def demo_model_manager():
    """Demonstrate the model manager capabilities."""
    print("\n=== Model Manager Demo ===")
    
    # List all available models
    print("\nAvailable models:")
    models = ModelManager.list_available_models()
    for model_name, provider in models.items():
        print(f"  {model_name} ({provider})")
    
    # Check which models have API keys configured
    print("\nModel access status:")
    missing_keys = ModelManager.get_missing_api_keys()
    for model_name in models.keys():
        has_key = ModelManager.validate_model_access(model_name)
        status = "✅" if has_key else "❌"
        key_name = missing_keys.get(model_name, "")
        print(f"  {status} {model_name} {f'(missing {key_name})' if not has_key else ''}")


async def demo_conversation_with_moonshot():
    """Demonstrate conversation with Moonshot model."""
    print("\n=== Moonshot Conversation Demo ===")
    
    # Check if Moonshot API key is available
    if not ModelManager.validate_model_access("moonshot-v1-128k"):
        print("❌ MOONSHOT_API_KEY not found in environment")
        print("   Please set MOONSHOT_API_KEY in your .env file to run this demo")
        return
    
    try:
        # Create config with Moonshot model
        config = Config(
            work_model="moonshot-v1-128k",
            summary_model="moonshot-v1-128k",
            db_path="./demo_conversations.db"
        )
        
        # Initialize conversation manager
        storage = DuckDBStorage(config.db_path)
        manager = HierarchicalConversationManager(config, storage)
        
        # Start a conversation
        conversation_id = await manager.start_conversation()
        print(f"Started conversation: {conversation_id}")
        
        # Have a short conversation
        messages = [
            "Hello! Can you tell me about the Moon?",
            "What's the most interesting fact about lunar exploration?",
        ]
        
        for message in messages:
            print(f"\nUser: {message}")
            response = await manager.chat(message)
            print(f"Assistant: {response[:200]}{'...' if len(response) > 200 else ''}")
        
        # Get conversation summary
        summary = await manager.get_conversation_summary()
        print(f"\nConversation summary: {summary}")
        
    except Exception as e:
        print(f"❌ Error: {e}")


async def demo_multiple_models():
    """Demonstrate using different models."""
    print("\n=== Multiple Models Demo ===")
    
    # Test models that have API keys
    test_models = []
    for model_name in ["claude-4-sonnet", "gpt-4o", "moonshot-v1-128k", "deepseek-chat"]:
        if ModelManager.validate_model_access(model_name):
            test_models.append(model_name)
    
    if not test_models:
        print("❌ No models have API keys configured")
        return
    
    print(f"Testing with models: {test_models}")
    
    for model_name in test_models[:2]:  # Test first 2 available models
        print(f"\n--- Testing {model_name} ---")
        try:
            config = Config(work_model=model_name, db_path=f"./demo_{model_name.replace('-', '_')}.db")
            storage = DuckDBStorage(config.db_path)
            manager = HierarchicalConversationManager(config, storage)
            
            conversation_id = await manager.start_conversation()
            response = await manager.chat("Hello! What model are you?")
            print(f"Response: {response[:150]}{'...' if len(response) > 150 else ''}")
            
        except Exception as e:
            print(f"❌ Error with {model_name}: {e}")


async def main():
    """Main demo function."""
    print("🚀 Hierarchical Memory Middleware - Model Manager Demo")
    print("======================================================")
    
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    await demo_model_manager()
    await demo_conversation_with_moonshot()
    await demo_multiple_models()
    
    print("\n✅ Demo completed!")


if __name__ == "__main__":
    asyncio.run(main())
