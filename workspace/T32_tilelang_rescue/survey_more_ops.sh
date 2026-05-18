#!/bin/bash
TLPATH=/home/z00637938/workspace/tilelang-mlir-ascend
echo "=== TESTING_NPUIR_COUNT ==="
find $TLPATH/testing/npuir -name "*.py" -type f | wc -l
echo "=== UNITTEST_NPUIR_COUNT ==="
find $TLPATH/unittest/npuir -name "*.py" -type f | wc -l
echo "=== TESTING_NPUIR_BY_CATEGORY ==="
find $TLPATH/testing/npuir -name "*.py" -type f -printf "%P\n" | awk -F/ '{print $1}' | sort | uniq -c | sort -rn
echo "=== UNITTEST_BY_TOPIC (sample 20) ==="
find $TLPATH/unittest/npuir -name "*.py" -type f -printf "%f\n" | head -20
