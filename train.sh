#!/bin/sh
# Orus-OCR-0.1-alpha — full training recipe (see docs/04_training.md for what each stage does and why).
#
#   ./train.sh <stage> [<stage> ...]      stages, in order:
#     data     unify raw datasets -> data/ocr/pages/*.jsonl, cut 32 px line crops -> data/ocr/rec/*  (mmap stores)
#     synth    word bank + synthetic font lines (5 x 50k) + stitched real-word lines (4 x 50k)
#     lm       character 5-gram LM                                   -> runs/orus_lm.npz
#     A        stage A: Indra init, 30k steps, lr 1.5e-3              -> runs/orus_rec_a.pt
#     B1       stage B1: from A, all data incl. IIIT + stitched, 40k  -> runs/orus_rec_b1.pt
#     B3       stage B3: from B1, lower LR, 30k                       -> runs/orus_rec_b3.pt
#     distill  teacher (PaddleOCR-VL-1.6) labels for unlabeled English pages, agreement filter -> distill stores
#     C        stage C: from B3 + distill lines, 12k                  -> runs/orus_rec_c.pt  (= released model)
#     pic      picture-region net                                     -> runs/orus_pic.onnx
#     export   fp32 ONNX of C                                         -> runs/orus_rec.onnx
#     bench    line benchmark, page evaluation, error analysis of runs/orus_rec.onnx -> runs/*.json
#     B2       (optional) size sweep: 2.77M model from scratch, 45k steps -> runs/orus_rec_b2.pt
#
# Outputs go to runs/; models/ holds the frozen release and is never written. Copy a result into models/ by hand
# only to make a new release. Every heavy step runs in a memory-capped systemd scope (the machine has 15.6 GB RAM
# shared with other work; an OOM there once killed the whole desktop session).
set -e
cd "$(dirname "$0")"
P=${PYTHON:-~/Apps/miniconda3/envs/torch/bin/python}
cap() { systemd-run --user --scope -q -p MemoryMax=${MEM:-6G} -p MemorySwapMax=1G --unit=orus01-$(date +%H%M%S%N) "$@"; }
T="cap $P -u orus/ocr_train.py rec --init indra"

for stage in "$@"; do
  echo "== stage $stage ($(date +%H:%M))"
  case $stage in
    data)
      cap $P -u orus/ocr_data.py pages
      cap $P -u orus/ocr_data.py crops ;;
    synth)
      cap $P -u orus/ocr_synth.py words
      MEM=4G cap $P -u orus/ocr_synth.py lines font 3000 val 0
      MEM=4G cap $P -u orus/ocr_synth.py lines font 5000 test 0
      for s in 0 1 2 3 4; do MEM=4G cap $P -u orus/ocr_synth.py lines font 50000 train $s; done
      MEM=4G cap $P -u orus/ocr_synth.py lines stitch 3000 val 0
      MEM=4G cap $P -u orus/ocr_synth.py lines stitch 5000 test 0
      for s in 0 1 2 3; do MEM=4G cap $P -u orus/ocr_synth.py lines stitch 50000 train $s; done ;;
    lm)
      cap $P -u orus/ocr_lm.py build 5 6e6 ;;
    A)
      $T --name orus_rec_a --steps 30000 --lr 1.5e-3 --eval_every 3000 --workers 3 2>&1 | tee runs/orus_rec_a.log ;;
    B1)
      $T --name orus_rec_b1 --resume runs/orus_rec_a.pt --steps 40000 --lr 8e-4 --eval_every 4000 --workers 3 2>&1 | tee runs/orus_rec_b1.log ;;
    B3)
      $T --name orus_rec_b3 --resume runs/orus_rec_b1.pt --steps 30000 --lr 4e-4 --eval_every 5000 --workers 2 2>&1 | tee runs/orus_rec_b3.log ;;
    distill)
      MEM=3G cap $P -u orus/ocr_distill.py lines 120
      MEM=5G cap $P -u orus/ocr_distill.py teach paddlevl
      MEM=3G cap $P -u orus/ocr_distill.py store runs/orus_rec_b1.pt 0.3 ;;
    C)
      $T --name orus_rec_c --resume runs/orus_rec_b3.pt --steps 12000 --lr 3e-4 --eval_every 3000 --workers 3 2>&1 | tee runs/orus_rec_c.log ;;
    pic)
      MEM=4G cap $P -u orus/ocr_pic.py data 4000
      MEM=4G cap $P -u orus/ocr_pic.py train 4000 2>&1 | tee runs/orus_pic.log ;;
    export)
      $P orus/ocr.py export runs/orus_rec_c.pt runs/orus_rec.onnx ;;
    bench)
      if [ "$REMAKE" = 1 ]; then MEM=3G cap $P -u orus/ocr_bench.py make; fi    # bench/ holds the frozen test sets
      MEM=3G cap $P -u orus/ocr_bench.py run orus runs/orus_rec.onnx
      MEM=3G cap $P -u orus/ocr_bench.py run indra
      MEM=3G cap $P -u orus/ocr_eval.py orus runs/orus_rec.onnx 30
      MEM=3G cap $P -u orus/ocr_eval.py indra - 30
      MEM=3G cap $P -u orus/ocr_errors.py collect runs/orus_rec_c.pt
      $P -u orus/ocr_errors.py analyze runs/preds_orus_rec_c.json ;;
    B2)
      cap $P -u orus/ocr_train.py rec --name orus_rec_b2 --init scratch --c 192 --e 384 --depth 12 --steps 45000 \
        --lr 1.5e-3 --eval_every 5000 --workers 2 2>&1 | tee runs/orus_rec_b2.log ;;
    *) echo "unknown stage $stage"; exit 1 ;;
  esac
done
