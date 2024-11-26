echo "Evaluation of the performance of the VectorDB and Circlemind on the same data (51 queries)";
echo;
echo "VectorDB";
python vdb_benchmark.py -n 51 -s
echo;
echo "Circlemind"
python graph_benchmark.py -n 51 -s

echo "Evaluation of the performance of the VectorDB and Circlemind on the same data (101 queries)";
echo;
echo "VectorDB";
python vdb_benchmark.py -n 101 -s
echo;
echo "Circlemind";
python graph_benchmark.py -n 101 -s