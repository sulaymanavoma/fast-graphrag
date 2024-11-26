## Benchmarks
We validate the benchmark results provided in [HippoRAG](https://arxiv.org/abs/2405.14831).

The scripts in this directory will generate and evaluate the 2wikimultihopqa datasets on a subsets of 51 and 101 queries with the same methodology as in the paper above. We preloaded the results so its is enough to run `evaluate.xx` to get the numbers. You can also run `create_dbs.xx` to regenerate the vector and graph databases.

The output should looks similar at follow (the exact numbers could vary based on your graph configuration)
```
Evaluation of the performance of the VectorDB and Circlemind on the same data (51 queries)

VectorDB
Loading dataset...
[all questions] Percentage of queries with perfect retrieval: 0.49019607843137253
[multihop only] Percentage of queries with perfect retrieval: 0.32432432432432434

Circlemind
Loading dataset...
[all questions] Percentage of queries with perfect retrieval: 0.9607843137254902
[multihop only] Percentage of queries with perfect retrieval: 0.9459459459459459


Evaluation of the performance of the VectorDB and Circlemind on the same data (101 queries)

VectorDB
Loading dataset...
[all questions] Percentage of queries with perfect retrieval: 0.4158415841584158
[multihop only] Percentage of queries with perfect retrieval: 0.2318840579710145

Circlemind
Loading dataset...
[all questions] Percentage of queries with perfect retrieval: 0.9306930693069307
[multihop only] Percentage of queries with perfect retrieval: 0.8985507246376812
```