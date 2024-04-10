# LLMatic

### Abstract:
_Large Language Models (LLMs) have emerged as powerful tools capable of accomplishing a broad spectrum of tasks. Their abilities span numerous areas, and one area where they have made a significant impact is in the domain of code generation. Here, we propose using the coding abilities of LLMs to introduce meaningful variations to code defining neural networks. Meanwhile, Quality-Diversity (QD) algorithms are known to discover diverse and robust solutions. By merging the code-generating abilities of LLMs with the diversity and robustness of QD solutions, we introduce \texttt{LLMatic}, a Neural Architecture Search (NAS) algorithm. While LLMs struggle to conduct NAS directly through prompts, \texttt{LLMatic} uses a procedural approach, leveraging QD for prompts and network architecture to create diverse and high-performing networks. We test \texttt{LLMatic} on the CIFAR-10 and NAS-bench-201 benchmarks, demonstrating that it can produce competitive networks while evaluating just $2,000$ candidates, even without prior knowledge of the benchmark domain or exposure to any previous top-performing models for the benchmark._

### To run experiments:

```python search_qd.py```
