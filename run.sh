#!/bin/bash
set -e

MODEL_PATH="${MODEL_PATH:-$HOME/Mirai/Models/Qwen3-0.6B}"
SEED=16
MESSAGE="Tell me briefly about Navier–Stokes equations"

BASELINE_BIN="$HOME/Mirai/2gram_k8.bin"
BASELINE_K32_BIN="$HOME/Mirai/2gram_k32.bin"
TAGGED_BIN="$HOME/Mirai/tagged_ngram_k8.bin"
TAGGED_K32_BIN="$HOME/Mirai/tagged_ngram_k32.bin"
KN_BIN="$HOME/Mirai/compress_time_kn_order4.bin"
KN_K32_BIN="$HOME/Mirai/compress_time_kn_k32.bin"

CLI=./target/release/cli

if [ ! -f "$CLI" ]; then
    echo "Building..."
    BINDGEN_EXTRA_CLANG_ARGS="-isysroot $(xcrun --show-sdk-path)" cargo build --release -p cli
fi

echo "=== Baseline (no speculator) ==="
$CLI run "$MODEL_PATH" \
    --no-thinking \
    --seed $SEED \
    --message "$MESSAGE"

for DEPTH in 1 2 3 4; do
    echo ""
    echo "=== Baseline 2-gram K=8 (depth=$DEPTH) ==="
    $CLI run "$MODEL_PATH" \
        --no-thinking \
        --seed $SEED \
        --speculator "${BASELINE_BIN}:${DEPTH}" \
        --message "$MESSAGE"
done

for DEPTH in 1 2 3 4; do
    echo ""
    echo "=== Baseline 2-gram K=32 (depth=$DEPTH) ==="
    $CLI run "$MODEL_PATH" \
        --no-thinking \
        --seed $SEED \
        --speculator "${BASELINE_K32_BIN}:${DEPTH}" \
        --message "$MESSAGE"
done

for DEPTH in 1 2 3 4; do
    echo ""
    echo "=== Tagged Multi-Table K=8 (order=4, depth=$DEPTH) ==="
    $CLI run "$MODEL_PATH" \
        --no-thinking \
        --seed $SEED \
        --speculator "tagged:${TAGGED_BIN}:${DEPTH}" \
        --message "$MESSAGE"
done

for DEPTH in 1 2 3 4; do
    echo ""
    echo "=== Tagged Multi-Table K=32 (order=4, depth=$DEPTH) ==="
    $CLI run "$MODEL_PATH" \
        --no-thinking \
        --seed $SEED \
        --speculator "tagged:${TAGGED_K32_BIN}:${DEPTH}" \
        --message "$MESSAGE"
done

for DEPTH in 1 2 3 4; do
    echo ""
    echo "=== KN Multi-Table K=8 (order=4, depth=$DEPTH) ==="
    $CLI run "$MODEL_PATH" \
        --no-thinking \
        --seed $SEED \
        --speculator "kn:${KN_BIN}:${DEPTH}" \
        --message "$MESSAGE"
done

for DEPTH in 1 2 3 4; do
    echo ""
    echo "=== KN Multi-Table K=32 (order=4, depth=$DEPTH) ==="
    $CLI run "$MODEL_PATH" \
        --no-thinking \
        --seed $SEED \
        --speculator "kn:${KN_K32_BIN}:${DEPTH}" \
        --message "$MESSAGE"
done

TAU=0.3

for DEPTH in 1 2 3 4; do
    echo ""
    echo "=== Tagged K=8 τ=$TAU (depth=$DEPTH) ==="
    $CLI run "$MODEL_PATH" \
        --no-thinking \
        --seed $SEED \
        --speculator "tagged:${TAGGED_BIN}:${DEPTH}:${TAU}" \
        --message "$MESSAGE"
done

for DEPTH in 1 2 3 4; do
    echo ""
    echo "=== Tagged K=32 τ=$TAU (depth=$DEPTH) ==="
    $CLI run "$MODEL_PATH" \
        --no-thinking \
        --seed $SEED \
        --speculator "tagged:${TAGGED_K32_BIN}:${DEPTH}:${TAU}" \
        --message "$MESSAGE"
done

for DEPTH in 1 2 3 4; do
    echo ""
    echo "=== KN K=8 τ=$TAU (depth=$DEPTH) ==="
    $CLI run "$MODEL_PATH" \
        --no-thinking \
        --seed $SEED \
        --speculator "kn:${KN_BIN}:${DEPTH}:${TAU}" \
        --message "$MESSAGE"
done

for DEPTH in 1 2 3 4; do
    echo ""
    echo "=== KN K=32 τ=$TAU (depth=$DEPTH) ==="
    $CLI run "$MODEL_PATH" \
        --no-thinking \
        --seed $SEED \
        --speculator "kn:${KN_K32_BIN}:${DEPTH}:${TAU}" \
        --message "$MESSAGE"
done
