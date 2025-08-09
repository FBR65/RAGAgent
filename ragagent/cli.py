import click
import json
import logging
from pathlib import Path
from typing import Optional

from .pipeline import AgenticRAGPipeline, create_pipeline
from .unified_client import UnifiedClient, ClientFactory, ProviderType
from .enhanced_timeout_retry import ResilienceManager, get_resilience_manager
from .config_versioning import ConfigVersionManager, get_version_manager
from .models import (
    PipelineConfig,
    AgentConfig,
    DocumentProcessingConfig,
    ProcessingRequest,
)

logger = logging.getLogger(__name__)


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
@click.option(
    "--config", "-c", type=click.Path(exists=True), help="Path to configuration file"
)
@click.pass_context
def cli(ctx, verbose, config):
    """Agentic RAG CLI - Process documents with AI agents"""

    # Setup logging
    if verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    # Load configuration
    pipeline_config = None
    if config:
        try:
            with open(config, "r") as f:
                config_data = json.load(f)
                pipeline_config = PipelineConfig(**config_data)
        except Exception as e:
            click.echo(f"Error loading configuration: {e}")
            ctx.exit(1)

    # Create pipeline and other components
    pipeline = create_pipeline(pipeline_config)
    unified_client = ClientFactory.create_client()
    resilience_manager = get_resilience_manager()
    version_manager = get_version_manager()

    ctx.ensure_object(dict)
    ctx.obj["pipeline"] = pipeline
    ctx.obj["unified_client"] = unified_client
    ctx.obj["resilience_manager"] = resilience_manager
    ctx.obj["version_manager"] = version_manager


@cli.command()
@click.argument("document_path", type=click.Path(exists=True))
@click.argument("question")
@click.option("--output", "-o", type=click.Path(), help="Output file for results")
@click.option("--model", default="qwen3:latest", help="Model to use")
@click.option(
    "--base-url",
    default="http://localhost:11434",
    help="Base URL for the model service",
)
@click.option("--max-tokens", default=4000, help="Maximum tokens for responses")
@click.option("--temperature", default=0.1, help="Temperature for generation")
@click.option("--no-verification", is_flag=True, help="Disable answer verification")
@click.pass_context
def process(
    ctx,
    document_path,
    question,
    output,
    model,
    base_url,
    max_tokens,
    temperature,
    no_verification,
):
    """Process a document with a question"""

    pipeline = ctx.obj["pipeline"]

    # Create request
    request = ProcessingRequest(
        question=question,
        document_path=document_path,
        config=PipelineConfig(
            agent=AgentConfig(
                model_name=model,
                base_url=base_url,
                max_tokens=max_tokens,
                temperature=temperature,
            ),
            enable_verification=not no_verification,
        ),
    )

    try:
        # Process request
        result = pipeline.process_request(request)

        if result.success:
            response = result.data

            # Display results
            click.echo(f"\n📋 Question: {response.question}")
            click.echo(f"📝 Answer: {response.answer}")
            click.echo(f"📚 Citations: {', '.join(response.citations)}")
            click.echo(f"🎯 Confidence: {response.confidence_score:.2f}")
            click.echo(f"⏱️ Processing time: {response.processing_time:.2f}s")
            click.echo(f"📄 Source paragraphs: {response.source_paragraphs}")

            if response.verification:
                click.echo(f"✅ Verification: {response.verification.is_accurate}")
                click.echo(
                    f"🔍 Verification confidence: {response.verification.confidence}"
                )
                if response.verification.issues_found:
                    click.echo(
                        f"⚠️ Issues: {', '.join(response.verification.issues_found)}"
                    )

            # Save to file if requested
            if output:
                with open(output, "w") as f:
                    json.dump(response.dict(), f, indent=2, ensure_ascii=False)
                click.echo(f"\n💾 Results saved to: {output}")
        else:
            click.echo(f"❌ Error: {result.error.message}")
            if result.error.details:
                click.echo(f"Details: {result.error.details}")

    except Exception as e:
        click.echo(f"❌ Processing failed: {e}")
        ctx.exit(1)


@cli.command()
@click.argument("document_path", type=click.Path(exists=True))
@click.option("--output", "-o", type=click.Path(), help="Output file for chunks")
@click.pass_context
def chunks(ctx, document_path, output):
    """Extract and display chunks from a document"""

    pipeline = ctx.obj["pipeline"]

    try:
        # Process document to get chunks
        chunks = pipeline._process_document(document_path)

        click.echo(f"\n📊 Found {len(chunks)} chunks:")
        click.echo("-" * 50)

        for i, chunk in enumerate(chunks[:10]):  # Show first 10 chunks
            click.echo(f"\nChunk {chunk.id} ({chunk.token_count} tokens):")
            click.echo(f"  Preview: {chunk.text[:200]}...")

        if len(chunks) > 10:
            click.echo(f"\n... and {len(chunks) - 10} more chunks")

        # Save to file if requested
        if output:
            chunks_data = [
                {
                    "id": chunk.id,
                    "text": chunk.text,
                    "token_count": chunk.token_count,
                    "document_type": chunk.document_type.value,
                }
                for chunk in chunks
            ]
            with open(output, "w") as f:
                json.dump(chunks_data, f, indent=2, ensure_ascii=False)
            click.echo(f"\n💾 Chunks saved to: {output}")

    except Exception as e:
        click.echo(f"❌ Chunk extraction failed: {e}")
        ctx.exit(1)


@cli.command()
@click.option("--model", default="qwen3:latest", help="Model to test")
@click.option(
    "--base-url",
    default="http://localhost:11434",
    help="Base URL for the model service",
)
@click.option("--provider", type=click.Choice(list(ProviderType)), help="Provider type")
@click.pass_context
def test(ctx, model, base_url, provider):
    """Test connection to the model service"""

    pipeline = ctx.obj["pipeline"]
    unified_client = ctx.obj["unified_client"]
    resilience_manager = ctx.obj["resilience_manager"]
    version_manager = ctx.obj["version_manager"]

    # Update config for testing
    pipeline.config.agent.model_name = model
    pipeline.config.agent.base_url = base_url

    # Update unified client
    if provider:
        unified_client = ClientFactory.create_client(provider)
        unified_client.update_config(
            {
                "model_name": model,
                "base_url": base_url,
            }
        )

    try:
        # Test pipeline connection
        pipeline_info = pipeline.test_connection()

        # Test unified client connection
        client_info = unified_client.test_connection()

        # Get resilience stats
        resilience_stats = resilience_manager.get_stats()

        # Get version history
        version_history = version_manager.get_version_history()

        click.echo(f"\n🔌 Connection Test Results:")
        click.echo("=" * 50)

        # Pipeline info
        click.echo(f"📋 Pipeline:")
        click.echo(
            f"  Service: {'ollama' if 'ollama' in base_url.lower() else 'openai'}"
        )
        click.echo(f"  Model: {pipeline_info['model_name']}")
        click.echo(f"  Base URL: {pipeline_info['base_url']}")
        click.echo(
            f"  Connection: {'✅ Success' if pipeline_info['ollama_connection'] == 'success' else '❌ Failed'}"
        )
        click.echo(
            f"  Model available: {'✅ Yes' if pipeline_info['model_available'] else '❌ No'}"
        )

        # Unified client info
        click.echo(f"\n🤖 Unified Client:")
        click.echo(f"  Provider: {client_info.get('provider', 'unknown')}")
        click.echo(f"  Model: {client_info.get('model_name', 'unknown')}")
        click.echo(f"  Base URL: {client_info.get('base_url', 'unknown')}")
        click.echo(
            f"  Connection: {'✅ Success' if client_info.get('connection') else '❌ Failed'}"
        )

        # Resilience stats
        click.echo(f"\n🛡️ Resilience Manager:")
        click.echo(f"  Timeout config: {resilience_stats['timeout']}")
        click.echo(f"  Retry strategy: {resilience_stats['retry']['strategy']}")
        click.echo(f"  Circuit breaker: {resilience_stats['circuit_breaker']['state']}")
        click.echo(
            f"  Rate limiter: {resilience_stats['rate_limiter']['dry_run'] and 'DRY RUN' or 'ACTIVE'}"
        )

        # Version manager
        click.echo(f"\n📚 Version Manager:")
        click.echo(f"  Total versions: {len(version_history)}")
        click.echo(
            f"  Current version: {next((v['version'] for v in version_history if v['is_current']), 'None')}"
        )
        click.echo(
            f"  Active versions: {len([v for v in version_history if v['is_active']])}"
        )

        # Errors
        if pipeline_info["error"]:
            click.echo(f"\n❌ Pipeline Error: {pipeline_info['error']}")

        if client_info.get("error"):
            click.echo(f"❌ Client Error: {client_info['error']}")

    except Exception as e:
        click.echo(f"❌ Test failed: {e}")
        ctx.exit(1)


@cli.command()
@click.pass_context
def info(ctx):
    """Display system information"""

    pipeline = ctx.obj["pipeline"]
    unified_client = ctx.obj["unified_client"]
    resilience_manager = ctx.obj["resilience_manager"]
    version_manager = ctx.obj["version_manager"]

    try:
        system_info = pipeline.get_system_info()
        client_info = unified_client.get_info()
        resilience_stats = resilience_manager.get_stats()
        version_history = version_manager.get_version_history()

        click.echo(f"\n📋 System Information:")
        click.echo("=" * 50)

        # Basic system info
        click.echo(f"🎯 Pipeline:")
        click.echo(f"  Version: {system_info['version']}")
        click.echo(f"  Parsers: {system_info['parsers']}")
        click.echo(
            f"  Supported formats: {', '.join(system_info['supported_formats'])}"
        )

        # Unified client info
        click.echo(f"\n🤖 Unified Client:")
        click.echo(f"  Provider: {client_info.get('provider', 'unknown')}")
        click.echo(f"  Model: {client_info.get('model_name', 'unknown')}")
        click.echo(f"  Base URL: {client_info.get('base_url', 'unknown')}")
        click.echo(f"  Config: {json.dumps(client_info.get('config', {}), indent=2)}")

        # Resilience stats
        click.echo(f"\n🛡️ Resilience Manager:")
        click.echo(f"  Timeout config: {resilience_stats['timeout']}")
        click.echo(f"  Retry strategy: {resilience_stats['retry']['strategy']}")
        click.echo(f"  Circuit breaker: {resilience_stats['circuit_breaker']['state']}")
        click.echo(
            f"  Rate limiter: {resilience_stats['rate_limiter']['dry_run'] and 'DRY RUN' or 'ACTIVE'}"
        )

        # Version manager
        click.echo(f"\n📚 Version Manager:")
        click.echo(f"  Total versions: {len(version_history)}")
        click.echo(
            f"  Current version: {next((v['version'] for v in version_history if v['is_current']), 'None')}"
        )
        click.echo(
            f"  Active versions: {len([v for v in version_history if v['is_active']])}"
        )

        # Pipeline config
        click.echo(f"\n⚙️ Pipeline Config:")
        click.echo(f"  {json.dumps(system_info['config'], indent=2)}")

    except Exception as e:
        click.echo(f"❌ Failed to get system info: {e}")
        ctx.exit(1)


@cli.command()
@click.argument("config_file", type=click.Path())
def init_config(config_file):
    """Initialize a configuration file"""

    default_config = {
        "agent": {
            "model_name": "qwen3:latest",
            "base_url": "http://localhost:11434",
            "max_tokens": 4000,
            "temperature": 0.1,
            "timeout": 30,
            "retry_count": 3,
            "retry_delay": 1.0,
        },
        "document_processing": {
            "max_chunk_size": 2000,
            "min_chunk_size": 100,
            "overlap_tokens": 100,
            "max_initial_chunks": 20,
            "max_navigation_depth": 3,
        },
        "enable_verification": True,
        "max_parallel_requests": 3,
        "ocr": {
            "enabled": True,
            "mode": "paragraph_based",
            "quality": "high",
            "language": "eng",
            "dpi": 300,
            "enable_preprocessing": True,
            "enable_postprocessing": True,
            "enable_layout_analysis": True,
            "max_pages": 100,
            "timeout": 300,
            "retry_count": 3,
            "cache_enabled": True,
        },
        "resilience": {
            "timeout": {
                "connect_timeout": 10.0,
                "read_timeout": 30.0,
                "write_timeout": 30.0,
                "total_timeout": 60.0,
            },
            "retry": {
                "max_retries": 3,
                "base_delay": 1.0,
                "max_delay": 60.0,
                "strategy": "exponential_jitter",
            },
            "circuit_breaker": {
                "failure_threshold": 5,
                "recovery_timeout": 60,
            },
            "rate_limiting": {
                "requests_per_second": 10,
                "requests_per_minute": 600,
                "requests_per_hour": 36000,
                "burst_size": 5,
            },
        },
        "hybrid_rag": {
            "enabled": False,
            "search_strategy": "hybrid",
            "vector_db": {
                "provider": "qdrant",
                "host": "localhost",
                "port": 6333,
                "collection_name": "rag_embeddings",
            },
            "embedder": {
                "model": "BGE-large-en-v1.5",
                "batch_size": 32,
                "device": "cpu",
            },
            "fusion_weights": {
                "vector": 0.5,
                "keyword": 0.3,
                "semantic": 0.2,
            },
        },
        "specialized_agents": {
            "enabled": False,
            "default_domain": "general",
            "agents": {
                "legal": {
                    "enabled": True,
                    "model": "gpt-4",
                    "temperature": 0.1,
                    "max_tokens": 4000,
                },
                "medical": {
                    "enabled": True,
                    "model": "gpt-4",
                    "temperature": 0.1,
                    "max_tokens": 4000,
                },
                "technical": {
                    "enabled": True,
                    "model": "gpt-4",
                    "temperature": 0.1,
                    "max_tokens": 4000,
                },
            },
        },
        "learning_system": {
            "enabled": False,
            "adaptation_threshold": 0.8,
            "learning_rate": 0.01,
            "max_history_size": 1000,
            "preference_learning": {
                "enabled": True,
                "update_frequency": "daily",
            },
        },
    }

    try:
        with open(config_file, "w") as f:
            json.dump(default_config, f, indent=2, ensure_ascii=False)
        click.echo(f"✅ Configuration file created: {config_file}")
        click.echo("You can now edit this file and use it with the --config option")
        click.echo("The configuration includes all new features:")
        click.echo("  - OCR support for scanned documents")
        click.echo("  - Enhanced resilience and timeout handling")
        click.echo("  - Hybrid RAG capabilities")
        click.echo("  - Specialized agents for different domains")
        click.echo("  - Learning system for adaptation")
        click.echo("  - Configuration versioning support")

    except Exception as e:
        click.echo(f"❌ Failed to create config file: {e}")
        exit(1)


@cli.command()
@click.pass_context
def versions(ctx):
    """Show configuration version history"""

    version_manager = ctx.obj["version_manager"]

    try:
        version_history = version_manager.get_version_history()

        if not version_history:
            click.echo("No configuration versions found")
            return

        click.echo(f"\n📚 Configuration Version History:")
        click.echo("=" * 70)

        for version_info in version_history:
            status = []
            if version_info["is_current"]:
                status.append("CURRENT")
            if version_info["is_active"]:
                status.append("ACTIVE")

            status_str = " | ".join(status) if status else "INACTIVE"

            click.echo(f"\n📋 Version: {version_info['version']}")
            click.echo(f"   Status: {status_str}")
            click.echo(f"   Created: {version_info['created_at']}")
            click.echo(f"   By: {version_info['created_by']}")
            click.echo(f"   Description: {version_info['description']}")
            click.echo(f"   Hash: {version_info['hash']}")

            if version_info.get("parent_version"):
                click.echo(f"   Parent: {version_info['parent_version']}")

    except Exception as e:
        click.echo(f"❌ Failed to get version history: {e}")
        ctx.exit(1)


@cli.command()
@click.argument("version")
@click.pass_context
def activate_version(ctx, version):
    """Activate specific configuration version"""

    version_manager = ctx.obj["version_manager"]

    try:
        success = version_manager.activate_version(version)
        if success:
            click.echo(f"✅ Version {version} activated")
        else:
            click.echo(f"❌ Version {version} not found")

    except Exception as e:
        click.echo(f"❌ Failed to activate version: {e}")
        ctx.exit(1)


@cli.command()
@click.argument("version")
@click.pass_context
def rollback_version(ctx, version):
    """Rollback to specific configuration version"""

    pipeline = ctx.obj["pipeline"]
    version_manager = ctx.obj["version_manager"]

    try:
        success = version_manager.rollback_to_version(version)
        if success:
            # Update pipeline with rolled back config
            rolled_back_version = version_manager.get_version(version)
            if rolled_back_version:
                pipeline.config = PipelineConfig(**rolled_back_version.config_data)
                click.echo(f"✅ Rolled back to version {version}")
        else:
            click.echo(f"❌ Version {version} not found")

    except Exception as e:
        click.echo(f"❌ Failed to rollback version: {e}")
        ctx.exit(1)


@cli.command()
@click.pass_context
def resilience_stats(ctx):
    """Show resilience manager statistics"""

    resilience_manager = ctx.obj["resilience_manager"]

    try:
        stats = resilience_manager.get_stats()

        click.echo(f"\n🛡️ Resilience Manager Statistics:")
        click.echo("=" * 50)

        click.echo(f"Timeout Config:")
        click.echo(f"  Connect: {stats['timeout']['connect_timeout']}s")
        click.echo(f"  Read: {stats['timeout']['read_timeout']}s")
        click.echo(f"  Write: {stats['timeout']['write_timeout']}s")
        click.echo(f"  Total: {stats['timeout']['total_timeout']}s")

        click.echo(f"\nRetry Strategy:")
        click.echo(f"  Strategy: {stats['retry']['strategy']}")
        click.echo(f"  Max retries: {stats['retry']['max_retries']}")
        click.echo(f"  Base delay: {stats['retry']['base_delay']}s")
        click.echo(f"  Max delay: {stats['retry']['max_delay']}s")

        click.echo(f"\nCircuit Breaker:")
        click.echo(f"  State: {stats['circuit_breaker']['state']}")
        click.echo(
            f"  Failure threshold: {stats['circuit_breaker']['failure_threshold']}"
        )
        click.echo(
            f"  Recovery timeout: {stats['circuit_breaker']['recovery_timeout']}s"
        )

        click.echo(f"\nRate Limiter:")
        click.echo(
            f"  Status: {'DRY RUN' if stats['rate_limiter']['dry_run'] else 'ACTIVE'}"
        )
        click.echo(f"  Requests/sec: {stats['rate_limiter']['requests_per_second']}")
        click.echo(f"  Burst size: {stats['rate_limiter']['burst_size']}")

    except Exception as e:
        click.echo(f"❌ Failed to get resilience stats: {e}")
        ctx.exit(1)


@cli.command()
@click.pass_context
def reset_resilience(ctx):
    """Reset resilience manager statistics"""

    resilience_manager = ctx.obj["resilience_manager"]

    try:
        resilience_manager = get_resilience_manager()
        click.echo("✅ Resilience manager reset")

    except Exception as e:
        click.echo(f"❌ Failed to reset resilience manager: {e}")
        ctx.exit(1)


def main():
    """Main entry point for the CLI"""
    cli()


if __name__ == "__main__":
    main()
