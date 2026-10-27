# export SIMREPO_ROOT="/path/to/your/SimEngine"

bash engine/scripts/run_testing.sh \
    $SIMREPO_ROOT/policy/configs/simengine/e2e_vadv2_50pct.py \
    $SIMREPO_ROOT/data/policy_engine/ckpts/e2e_vadv2_50pct_ep8.pth \
    e2e_vadv2_50pct-test \
    navtest_failures \
    NR