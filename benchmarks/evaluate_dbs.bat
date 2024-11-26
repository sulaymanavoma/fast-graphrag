@echo off
echo Evaluation of the performance of the VDB and GraphDB on the same data (51 queries)
echo.
echo VDB
python vdb_benchmark.py -n 51 -s
echo.
echo Graph
python graph_benchmark.py -n 51 -s

echo.
echo.
echo Evaluation of the performance of the VDB and GraphDB on the same data (101 queries)
echo.
echo VDB
python vdb_benchmark.py -n 101 -s
echo.
echo Graph
python graph_benchmark.py -n 101 -s
