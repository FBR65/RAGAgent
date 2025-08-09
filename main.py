def main():
    """Example usage of the Agentic RAG system"""

    from ragagent import create_pipeline, ProcessingRequest

    # Create pipeline
    pipeline = create_pipeline()

    # Example document text (in practice, you'd load from a file)
    document_text = """
    Patent applications must be filed within one year of public disclosure.
    The filing fee for a basic patent application is $1,600 for large entities.
    Small entities qualify for a 50% reduction in most patent fees.
    Trademark applications can be filed on an intent-to-use basis.
    Trademark registration provides protection for 10 years, renewable indefinitely.
    """

    # Create a temporary file for testing
    import tempfile
    import os

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(document_text)
        temp_file = f.name

    try:
        # Create processing request
        request = ProcessingRequest(
            question="What are the patent filing fees?", document_path=temp_file
        )

        # Process the request
        result = pipeline.process_request(request)

        if result.success:
            response = result.data
            print(f"Question: {response.question}")
            print(f"Answer: {response.answer}")
            print(f"Citations: {response.citations}")
            print(f"Confidence: {response.confidence_score:.2f}")
            print(f"Processing time: {response.processing_time:.2f}s")
        else:
            print(f"Error: {result.error.message}")

    finally:
        # Clean up temporary file
        os.unlink(temp_file)


if __name__ == "__main__":
    main()
