"""Basic text chunking example.

Demonstrates how to use the RecursiveChunker to split text into chunks.
"""

from gweta.ingest.chunkers import RecursiveChunker


def main():
    # Sample long text to chunk
    text = """
    Python is a high-level, general-purpose programming language. Its design 
    philosophy emphasizes code readability with the use of significant indentation.
    
    Python is dynamically typed and garbage-collected. It supports multiple 
    programming paradigms, including structured, object-oriented and functional 
    programming.
    
    Python was conceived in the late 1980s by Guido van Rossum at Centrum 
    Wiskunde & Informatica (CWI) in the Netherlands. The language was released 
    in 1991 as Python 0.9.0, featuring classes with inheritance, exception 
    handling, and functions.
    
    Python consistently ranks as one of the most popular programming languages.
    It is used in web development, data science, artificial intelligence, 
    scientific computing, and many other domains.
    
    The Python Package Index (PyPI) hosts over 400,000 packages covering a wide 
    range of functionality. Popular packages include NumPy for numerical computing,
    Pandas for data analysis, and Django for web development.
    """

    # Create chunker with custom settings
    chunker = RecursiveChunker(
        chunk_size=200,
        chunk_overlap=50,
    )

    print("Chunking text with RecursiveChunker")
    print(f"Settings: size={chunker.chunk_size}, overlap={chunker.chunk_overlap}")
    print("=" * 50)
    print()

    # Split text into chunks
    chunks = chunker.chunk(text, source="python_intro.txt")

    # Display chunks
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i+1} ({len(chunk.text)} chars):")
        print(f"  {chunk.text[:100]}...")
        print()

    print(f"Total: {len(chunks)} chunks created")


if __name__ == "__main__":
    main()
