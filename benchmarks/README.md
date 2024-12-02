## Benchmarks
We validate the benchmark results provided in [HippoRAG](https://arxiv.org/abs/2405.14831), as well as comparing with other methods:
- NaiveRAG using the embedder `text-embedding-3-small`
- [LightRAG](https://github.com/HKUDS/LightRAG) 

The scripts in this directory will generate and evaluate the 2wikimultihopqa datasets on a subsets of 51 and 101 queries with the same methodology as in the HippoRAG paper. In particular, we evaluate the retrieval capabilities of each method, mesauring the percentage of queries for which all the required evidence was retrieved. We preloaded the results so it is enough to run `evaluate.xx` to get the numbers. You can also run `create_dbs.xx` to regenerate the databases for the different methods (you will need to set OPENAI_API_KEY, LightRAG could take a while to process).

The output should looks similar at follow (the exact numbers could vary based on your graph configuration)
```
Evaluation of the performance of different RAG methods on 2wikimultihopqa (51 queries)

VectorDB
Loading dataset...
[all questions] Percentage of queries with perfect retrieval: 0.49019607843137253
[multihop only] Percentage of queries with perfect retrieval: 0.32432432432432434

LightRAG
Loading dataset...
Percentage of queries with perfect retrieval: 0.47058823529411764
[multihop] Percentage of queries with perfect retrieval: 0.32432432432432434

Circlemind
Loading dataset...
[all questions] Percentage of queries with perfect retrieval: 0.9607843137254902
[multihop only] Percentage of queries with perfect retrieval: 0.9459459459459459


Evaluation of the performance of different RAG methods on 2wikimultihopqa (101 queries)

VectorDB
Loading dataset...
[all questions] Percentage of queries with perfect retrieval: 0.4158415841584158
[multihop only] Percentage of queries with perfect retrieval: 0.2318840579710145

LightRAG [local]
Loading dataset...
Percentage of queries with perfect retrieval: 0.44554455445544555
[multihop] Percentage of queries with perfect retrieval: 0.2753623188405797

Circlemind
Loading dataset...
[all questions] Percentage of queries with perfect retrieval: 0.9306930693069307
[multihop only] Percentage of queries with perfect retrieval: 0.8985507246376812
```

We also quickly benchmarked on the HotpotQA dataset (we will soon release the code for that as well). Here's a preview of the results (101 queries):

```
VectorDB: 0.78

LightRAG [local mode]: 0.55

Circlemind: 0.84
```
