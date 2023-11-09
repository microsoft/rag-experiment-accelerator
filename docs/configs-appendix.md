# Understanding the config files

## Prerequisites
Familiarity with [ReadMe configuration of elements](/README.md#Description-of-configuration-elements)

## Configuration links for more reading. 
- Search Types
    - [Semantic Search][semantic search]
    - [Vector Search][vector search]
    - [Hybrid Search][hybrid search]
- Chunking Strategies
    - [Size][Chunk Size]
    - [Overlap][Overlap]
- [Embedding][Embeddings]
    - Models: The accelerator uses [Sentence Transformer][Sentence Transformer] to generate the embeddings which utilizes [Pre-Trained Models][Transformer Models] based on embedding dimensions.
    - Dimensions: Each valid value maps to different models for embedding.
        - 384: [all-MiniLM-L6-v2][all-MiniLM-L6-v2]
        - 768: [all-mpnet-base-v2][all-mpnet-base-v2]
        - 1024:[bert-large-nli-mean-tokens][bert-large-nli-mean-tokens]
- LLM Metrics calculated using scikit-learn in combination with `Math.mean`
    - [Precision][precision score]
    - [Recall][recall score]
- [Prompt Engineering][prompts]


<!--- Link references --->
[Chunk Size]: https://learn.microsoft.com/en-us/azure/search/vector-search-how-to-chunk-documents#common-chunking-techniques
[Overlap]: https://learn.microsoft.com/en-us/azure/search/vector-search-how-to-chunk-documents#content-overlap-considerations
[Embeddings]: https://learn.microsoft.com/en-us/azure/ai-services/openai/concepts/understand-embeddings
[Sentence Transformer]: https://www.sbert.net/
[Transformer Models]: https://www.sbert.net/docs/pretrained_models.html
[all-MiniLM-L6-v2]: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
[all-mpnet-base-v2]: https://huggingface.co/sentence-transformers/all-mpnet-base-v2
[bert-large-nli-mean-tokens]: https://huggingface.co/sentence-transformers/bert-large-nli-mean-tokens
[prompts]: https://learn.microsoft.com/en-us/azure/ai-services/openai/concepts/advanced-prompt-engineering?pivots=programming-language-chat-completions
[recall score]: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html
[precision score]: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html#sklearn.metrics.precision_score
[vector search]: https://learn.microsoft.com/en-us/azure/search/vector-search-overview
[hybrid search]: https://learn.microsoft.com/en-us/azure/search/hybrid-search-overview
[semantic search]: https://learn.microsoft.com/en-us/azure/search/semantic-search-overview