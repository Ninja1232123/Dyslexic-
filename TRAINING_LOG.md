# π/2 Training Log

## Phase 1 - Complete (2025-12-23)

### Model A
- **Checkpoint**: `output_model_a/checkpoints/hebbian_step_5000A.pt`
- **Data**: `phase1_combined.jsonl`
- **Steps**: 5000
- **Final Loss**: 0.0977
- **Consistency**: 0.0586

### Model B
- **Checkpoint**: `output_model_b/checkpoints/hebbian_step_5000B.pt`
- **Data**: `phase1_combined.jsonl`
- **Steps**: 5000
- **Final Loss**: 0.0909
- **Consistency**: 0.0527

---

## Phase 2 - In Progress (Started 2025-12-23 14:02)

### Model A (GPU 0)
```bash
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
.venv/bin/python train_hebbian.py \
    --data "Training PI/Phase2(A-TalkSmart)/pi_encoded_phase2.jsonl" \
    --size 1.57B \
    --steps 5000 \
    --batch 2 \
    --lr 0.001 \
    --quantize \
    --resume output_model_a/checkpoints/hebbian_step_5000A.pt \
    --output ./output_model_a_phase2
```
- **Resume From**: Phase 1 step 5000A (loss: 0.0978)
- **GPU Memory**: ~9.2GB
- **Output**: `./output_model_a_phase2/`
- **Log**: `tail -f output_model_a_phase2.log`

### Model B (GPU 1)
```bash
CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
.venv/bin/python train_hebbian.py \
    --data "Training PI/Phase2(A-TalkSmart)/pi_encoded_phase2.jsonl" \
    --size 1.57B \
    --steps 5000 \
    --batch 2 \
    --lr 0.001 \
    --quantize \
    --resume output_model_b/checkpoints/hebbian_step_5000B.pt \
    --output ./output_model_b_phase2
```
- **Resume From**: Phase 1 step 5000B
- **GPU Memory**: ~9.0GB
- **Output**: `./output_model_b_phase2/`
- **Log**: `tail -f output_model_b_phase2.log`

### Phase 2 Data
- **Original**: `Training PI/Phase2(A-TalkSmart)/pi_encoded_phase2.jsonl` (11,716 samples)
- **Recovered**: `phase2_recovered_tokenized.jsonl` (354 samples from PDFs + frontierscience)
- **Combined**: `phase2_combined.jsonl` (12,070 samples, 149MB)

#### Recovered Data Sources:
- Philosophy: Kuhn, Sartre, Sutton memory traces
- Science: relativity, AI, frontierscience (160 physics problems)
- Programming: code.pdf, DSA, Programming-Fundamentals
- Legal/thesis papers

### Training Settings
| Parameter | Value |
|-----------|-------|
| Model Size | 1.57B |
| Hidden Size | 2304 |
| Layers | 24 |
| Heads | 18 |
| Batch Size | 2 (reduced from 4 for memory) |
| Learning Rate | 0.001 |
| Steps | 5000 |
| Quantized | Yes (π/2 phase encoding) |
| Training Type | Hebbian (no optimizer) |

---

## Curriculum Plan

| Phase | Data Type | Target % | Status |
|-------|-----------|----------|--------|
| Phase 1 | Fractals, Symphonies, Code | 20% | ✅ Complete |
| Phase 2 | Coherent Imagery, Audio, Advanced Educational Text | 60% | 🔄 In Progress |
| Phase 3 | Metacognition (consciousness, paradoxes, creativity) | 20% | ⏳ Pending |
| Phase 4 | Shell/terminal training | CLI assistant | ⏳ Pending |

---

## Phase 2 Combined Dataset (Ready for Next Run)

**File**: `phase2_all_combined.jsonl` (220MB, 63,956 samples)

| Modality | Samples | Source |
|----------|---------|--------|
| Text | 12,070 | Educational textbooks (quantum, music theory, math, coding) |
| Images | 50,000 | CIFAR-10, MNIST, Fashion-MNIST, SVHN, Food101, Cats vs Dogs |
| Audio | 1,886 | music-genre MP3 dataset |
| **Total** | **63,956** | |

### Pending (not yet tokenized)
| Type | Count | Location |
|------|-------|----------|
| MOV videos | 1,192 | 3DFDReal |
| PLY 3D clouds | 1,090 | point cloud data |

---

## Notes

- Both models training in parallel on separate GPUs
- Using Hebbian learning (rotation consistency) - no backprop optimizer
- Phase 2 runs in multiple passes: text first, then images/audio/video
