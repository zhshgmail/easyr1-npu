#!/bin/bash
# Run each unittest test in its own subprocess sequentially.
TLPATH=/home/z00637938/workspace/tilelang-mlir-ascend
BISHENGIR_PKGS=$TLPATH/3rdparty/AscendNPU-IR/build/install/python_packages
export PYTHONPATH=$TLPATH:$BISHENGIR_PKGS/mlir_core:$BISHENGIR_PKGS/bishengir:$PYTHONPATH
export PATH=$TLPATH/3rdparty/AscendNPU-IR/build/install/bin:$PATH
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cd $TLPATH

pass=0
fail=0
hang=0
fails=""
hangs=""
for t in $(find unittest/npuir -name 'test_*.py' -type f | sort); do
  out=$(timeout 90 python3 -m pytest "$t" --tb=line -q --no-header 2>&1 | tail -10)
  rc=$?
  shortname=$(basename "$t")
  if [ $rc -eq 124 ]; then
    echo "HANG  $shortname (timeout)"
    hang=$((hang+1))
    hangs="$hangs $shortname"
  elif echo "$out" | grep -q " passed"; then
    pass=$((pass+1))
  else
    fmsg=$(echo "$out" | grep -iE 'fail|error:|assertion' | head -1)
    echo "FAIL  $shortname : $fmsg"
    fail=$((fail+1))
    fails="$fails $shortname"
  fi
done
echo
echo "=========="
echo "Summary: $pass PASS, $fail FAIL, $hang HANG of $((pass+fail+hang)) total"
[ -n "$fails" ] && echo "Failing: $fails"
[ -n "$hangs" ] && echo "Hanging: $hangs"
