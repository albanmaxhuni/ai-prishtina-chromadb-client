# AI Prishtina VectorDB Client

![AI Prishtina Logo](assets/png/ai-prishtina.jpeg)

## Overview

The AI Prishtina VectorDB Client is a powerful library designed to facilitate seamless interaction with vector databases. It provides robust functionality for loading, processing, and streaming data from various sources, including cloud storage services like Amazon S3, Google Cloud Storage, Azure Blob Storage, and MinIO.

## Features

- **Data Source Handling**: Support for multiple data formats including text, JSON, CSV, Excel, Word documents, PDFs, images, audio, and video files.
- **Cloud Storage Integration**: Direct integration with major cloud storage providers for efficient data streaming.
- **Embedding Functions**: Built-in support for generating embeddings for text, images, audio, and video data.
- **Error Handling**: Comprehensive error handling and logging for robust data processing.

## Installation

To install the AI Prishtina VectorDB Client, run:

```bash
pip install ai-prishtina-vectordb
```

## Usage

### Basic Usage

```python
from ai_prishtina_vectordb.data_sources import DataSource

# Initialize the data source
source = DataSource(source_type="text")

# Load data from a file
data = source.load_data(source="path/to/your/file.txt", text_column="content", metadata_columns=["author", "date"])

# Stream data from a cloud storage
for batch in source.stream_data(source="s3://my-bucket/data/", text_column="text", metadata_columns=["source", "bucket"], batch_size=100):
    print(batch)
```

### Cloud Storage

The library supports streaming data from various cloud storage services. Ensure you have the necessary credentials configured in your environment or passed as parameters.

#### Example: Streaming from S3

```python
source.stream_data(source="s3://my-bucket/data/", text_column="text", metadata_columns=["source", "bucket"], batch_size=100, aws_access_key_id="your_access_key", aws_secret_access_key="your_secret_key")
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For support, please:
1. Check the [documentation](https://docs.ai-prishtina.com/ai-prishtina-chromadb-client)
2. Search [existing issues](https://github.com/ai-prishtina-chromadb-client/issues)
3. Create a new issue if needed

## Roadmap

- [ ] Multi-modal search
- [ ] Distributed deployment
- [ ] Advanced caching strategies
- [ ] More embedding models
- [ ] Performance optimizations

## Citation

If you use AIPrishtina VectorDB in your research, please cite:

```bibtex
@software{ai_prishtina_vectordb,
  author = {Alban Maxhuni, PhD},
  title = {AIPrishtina ChromaDB},
  year = {2024},
  url = {https://github.com/albanmaxhuni/ai-prishtina-chromadb-client.git}
}
```