"""
Implementing LLMs

To understand how Large Language Models (LLMs) operate, let's examine a simplified pseudocode example that outlines the tokenization and embedding process.

This process is fundamental to how LLMs interpret and generate text, and it draws parallels to various software development practices.

"""

import numpy as np
from typing import List, Dict, Tuple
import re


class SimpleLLMPipeline:
    """
    A simplified implementation demonstrating core LLM processing steps:
    1. Tokenization
    2. Embedding
    3. Positional Encoding
    4. Attention Mechanism (simplified)
    5. Text Generation
    """

    def __init__(self, vocab_size: int = 10000, embedding_dim: int = 512):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        # Initialize vocabulary and embeddings
        self.vocab = {}  # token -> id mapping
        self.reverse_vocab = {}  # id -> token mapping
        self.embeddings = np.random.randn(vocab_size, embedding_dim) * 0.1

        # Special tokens
        self.PAD_TOKEN = "<PAD>"
        self.UNK_TOKEN = "<UNK>"
        self.START_TOKEN = "<START>"
        self.END_TOKEN = "<END>"

    # Tokenize Function
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenization: Convert text into tokens

        The tokenize function is the starting point, where a given text is broken down into individual tokens.

        This process is like parsing in programming languages, where a string of code is divided into manageable components for analysis. Each token represents a basic unit of meaning, similar to how a parser identifies keywords, operators, and identifiers in code

        Real LLMs use subword tokenization (BPE, SentencePiece),but this example uses word-level tokenization for clarity.
        """
        # Basic preprocessing
        text = text.lower().strip()

        # Split on whitespace and punctuation (simplified)
        tokens = re.findall(r"\b\w+\b|[^\w\s]", text)

        # Add special tokens
        tokens = [self.START_TOKEN] + tokens + [self.END_TOKEN]

        return tokens

    def build_vocabulary(self, texts: List[str]) -> None:
        """Build vocabulary from training texts"""
        all_tokens = set()

        # Add special tokens first
        special_tokens = [
            self.PAD_TOKEN,
            self.UNK_TOKEN,
            self.START_TOKEN,
            self.END_TOKEN,
        ]
        for token in special_tokens:
            all_tokens.add(token)

        # Collect all tokens from texts
        for text in texts:
            tokens = self.tokenize(text)
            all_tokens.update(tokens)

        # Create vocab mappings
        for i, token in enumerate(sorted(all_tokens)):
            self.vocab[token] = i
            self.reverse_vocab[i] = token

    def tokens_to_ids(self, tokens: List[str]) -> List[int]:
        """Convert tokens to numerical IDs"""
        unk_id = self.vocab.get(self.UNK_TOKEN, 0)
        return [self.vocab.get(token, unk_id) for token in tokens]

    def ids_to_tokens(self, ids: List[int]) -> List[str]:
        """Convert numerical IDs back to tokens"""
        return [self.reverse_vocab.get(id, self.UNK_TOKEN) for id in ids]

    def embed_tokens(self, token_ids: List[int]) -> np.ndarray:
        """
        Embedding: Convert token IDs to dense vectors

        Once tokenized, the next step involves converting these tokens into embeddings.

        Embeddings are numerical vectors that capture the semantic essence of each token, much like how data structures such as arrays or hashmaps organize and store data for efficient retrieval. This vectorization process allows LLMs to understand context and relationships between words, enabling more nuanced text generation.

        Each token is mapped to a learned vector representation that captures semantic relationships.
        """
        embeddings = []
        for token_id in token_ids:
            if token_id < len(self.embeddings):
                embeddings.append(self.embeddings[token_id])
            else:
                # Use UNK embedding for out-of-vocab tokens
                embeddings.append(self.embeddings[self.vocab[self.UNK_TOKEN]])

        return np.array(embeddings)

    def add_positional_encoding(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Positional Encoding: Add position information to embeddings

        Since attention mechanisms don't inherently understand sequence order,
        we add positional information to embeddings.
        """
        seq_len, d_model = embeddings.shape
        pos_encoding = np.zeros((seq_len, d_model))

        # Create sinusoidal positional encodings
        for pos in range(seq_len):
            for i in range(0, d_model, 2):
                pos_encoding[pos, i] = np.sin(pos / (10000 ** (i / d_model)))
                if i + 1 < d_model:
                    pos_encoding[pos, i + 1] = np.cos(pos / (10000 ** (i / d_model)))

        return embeddings + pos_encoding

    def simple_attention(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Simplified Attention Mechanism

        Real transformers use multi-head self-attention, but this
        demonstrates the core concept of weighted token relationships.
        """
        seq_len, d_model = embeddings.shape

        # Compute attention scores (simplified)
        # In practice, this uses Query, Key, Value matrices
        attention_scores = np.dot(embeddings, embeddings.T)

        # Apply softmax to get attention weights
        attention_weights = self.softmax(attention_scores)

        # Apply attention weights to embeddings
        attended_embeddings = np.dot(attention_weights, embeddings)

        return attended_embeddings

    def softmax(self, x: np.ndarray) -> np.ndarray:
        """Softmax activation function"""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    def generate_next_token_probabilities(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Generate probability distribution over vocabulary for next token

        This is a simplified version - real LLMs use complex neural networks
        """
        # Use the last token's embedding to predict next token
        last_embedding = embeddings[-1]

        # Simple linear projection to vocabulary size
        # In practice, this is a learned linear layer
        vocab_scores = np.dot(last_embedding, self.embeddings.T)

        # Apply softmax to get probabilities
        probabilities = self.softmax(vocab_scores)

        return probabilities

    def process_text(self, text: str) -> Dict:
        """
        Complete pipeline: text -> tokens -> embeddings -> attention -> probabilities

        Finally, the process_text function uses these embeddings to predict and construct coherent text.

        This step is analogous to auto-completion features in Integrated Development Environments (IDEs), where the system suggests the next logical word or phrase based on context. Think "Smart Completions" for the IntelliJ IDEA and "IntelliSense" for Visual Studio. By leveraging probability-based models, LLMs can generate text that is contextually relevant and grammatically correct.
        """
        print(f"Input text: '{text}'")

        # Step 1: Tokenization
        tokens = self.tokenize(text)
        print(f"Tokens: {tokens}")

        # Step 2: Convert to IDs
        token_ids = self.tokens_to_ids(tokens)
        print(f"Token IDs: {token_ids}")

        # Step 3: Embedding
        embeddings = self.embed_tokens(token_ids)
        print(f"Embeddings shape: {embeddings.shape}")

        # Step 4: Add positional encoding
        pos_embeddings = self.add_positional_encoding(embeddings)
        print(f"With positional encoding: {pos_embeddings.shape}")

        # Step 5: Apply attention
        attended_embeddings = self.simple_attention(pos_embeddings)
        print(f"After attention: {attended_embeddings.shape}")

        # Step 6: Generate next token probabilities
        next_token_probs = self.generate_next_token_probabilities(attended_embeddings)

        # Get top predictions
        top_indices = np.argsort(next_token_probs)[-5:][::-1]
        top_tokens = [self.ids_to_tokens([idx])[0] for idx in top_indices]
        top_probs = [next_token_probs[idx] for idx in top_indices]

        return {
            "tokens": tokens,
            "token_ids": token_ids,
            "embeddings_shape": embeddings.shape,
            "top_predictions": list(zip(top_tokens, top_probs)),
        }


# Example usage and demonstration
def demonstrate_llm_pipeline():
    """Demonstrate the LLM pipeline with example data"""

    # Initialize the pipeline
    llm = SimpleLLMPipeline(vocab_size=1000, embedding_dim=128)

    # Sample training texts to build vocabulary
    training_texts = [
        "The cat sat on the mat",
        "A dog ran in the park",
        "She reads books every day",
        "Programming is fun and challenging",
        "Machine learning models process data",
    ]

    print("Building vocabulary from training texts...")
    llm.build_vocabulary(training_texts)
    print(f"Vocabulary size: {len(llm.vocab)}")

    # Process a sample text
    print("\n" + "=" * 50)
    print("PROCESSING PIPELINE DEMONSTRATION")
    print("=" * 50)

    sample_text = "The cat likes programming"
    result = llm.process_text(sample_text)

    print(f"\nTop 5 predicted next tokens:")
    for token, prob in result["top_predictions"]:
        print(f"  {token}: {prob:.4f}")

    print("\n" + "=" * 50)
    print("KEY CONCEPTS EXPLAINED")
    print("=" * 50)

    concepts = [
        (
            "Tokenization",
            "Breaking text into manageable units (words, subwords, characters)",
        ),
        (
            "Embedding",
            "Converting tokens to dense numerical vectors that capture meaning",
        ),
        ("Positional Encoding", "Adding sequence position information to embeddings"),
        (
            "Attention",
            "Allowing tokens to 'attend' to other relevant tokens in the sequence",
        ),
        (
            "Generation",
            "Predicting probability distributions over vocabulary for next tokens",
        ),
    ]

    for concept, explanation in concepts:
        print(f"\n{concept}:")
        print(f"  {explanation}")


if __name__ == "__main__":
    demonstrate_llm_pipeline()
