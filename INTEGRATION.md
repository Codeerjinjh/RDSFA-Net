# Integration Guide

This document lists the **complete delta** that RDSFA-Net adds on top of
`yolo-multimodal` / Ultralytics, so that you can port the proposed modules into an
existing multimodal fork instead of cloning the whole repository.

## 1. Files changed by this work

| File | Status | Purpose |
|---|---|---|
| `ultralytics/nn/AddModules/RDSFA.py` | **new** | All proposed modules (ECMI, RDSFA, EDRF, CSFR) and their helpers |
| `ultralytics/nn/AddModules/__init__.py` | modified | One line: `from .RDSFA import *` |
| `ultralytics/nn/tasks.py` | modified | Channel/shape registration inside `parse_model()` |
| `improve_multimodal/rtdetr/rtdetr-l-midfusion-RDSFA-Pro.yaml` | **new** | Paper model: ECMI + RDSFA + EDRF + CSFR, 500 queries, 8 decoder layers |
| `improve_multimodal/rtdetr/rtdetr-l-midfusion-RDSFA.yaml` | **new** | Lite variant: RDRFusion at P3–P5 only, no ECMI / CSFR |
| `datalvip.yaml`, `datam3fd.yaml`, `datadvtod.yaml` | **new** | Dataset configurations for the three benchmarks |
| `train.py`, `train_pro.py`, `val.py`, `val_pro.py`, `predict.py` | **new** | Training / validation / inference entry points |

Everything else under `ultralytics/` and `improve_multimodal/` belongs to the
upstream projects listed in the README acknowledgements.

## 2. Paper module → code map

| Paper name | Class in `RDSFA.py` | Notes |
|---|---|---|
| ECMI | `CrossModalInteraction` | Outputs the concatenation of both modalities, hence `c2 = 2 × C` |
| — | `SplitRGB`, `SplitIR` | Splits the ECMI output back into the two branches |
| Modality-specific backbone | `ModalityHGBlock_RGB`, `ModalityHGBlock_IR` | `HGBlock` with `SpatialAttention` / `ChannelSEAttention` respectively |
| RDSFA | `MultiScaleDeformableAlignment` | Three dilated offset branches + SE weighting + bidirectional symmetric sampling |
| EDRF | `EnhancedDegradationEstimator` + `DynamicModalityGate` + `EnhancedRecoveryGuidedFusion` | Assembled inside `RDRFusion` |
| CSFR | `CrossScaleFusion` | Coarse-to-fine refinement, `(channels_in, channels_out)` |
| — | `RDRFusion`, `RDSFusion` | Wrappers used directly in the YAML; `RDSFusion` is a subclass of `RDRFusion` |

Helper blocks: `SpatialAttention(kernel_size=3)`, `ChannelSEAttention(channels,
reduction=16)`, `ConvDW(c1, c2, k=3, stride=1, padding=1, bias=False)`.

Main module signatures:

```python
CrossModalInteraction(channels)                # ECMI
MultiScaleDeformableAlignment(channels)        # RDSFA
EnhancedDegradationEstimator(channels)         # EDRF: five per-pixel degradation factors
DynamicModalityGate(channels)                  # EDRF: shared / private recombination
EnhancedRecoveryGuidedFusion(channels)         # EDRF: degradation-gated recovery
CrossScaleFusion(channels_in, channels_out)    # CSFR
RDRFusion(Channel)                             # RDSFA + EDRF, used in the YAML
```

## 3. `tasks.py` registration

Add the following branches inside `parse_model()` (place them next to the other
custom fusion modules):

```python
elif m in {RDSFusion, RDRFusion}:
    c2 = ch[f[0]]                 # shared + private projected back to C
    args = [ch[f[0]]]
elif m is CrossModalInteraction:
    c2 = ch[f[0]] * 2             # concatenated RGB + IR = 2C
    args = [ch[f[0]]]
elif m in {SplitRGB, SplitIR}:
    c2 = ch[f] // 2               # half the channels
```

and make sure the package exports the new symbols:

```python
# ultralytics/nn/AddModules/__init__.py
from .RDSFA import *
```

## 4. Porting checklist

1. Copy `ultralytics/nn/AddModules/RDSFA.py` into your fork.
2. Add the import line to `ultralytics/nn/AddModules/__init__.py`.
3. Insert the three `elif` branches above into `ultralytics/nn/tasks.py`.
4. Copy the YAML configuration you need and adjust `nc` to your dataset.
5. Check the model builds before training:

```bash
python -c "from ultralytics import RTDETR; m = RTDETR('improve_multimodal/rtdetr/rtdetr-l-midfusion-RDSFA-Pro.yaml'); print(sum(p.numel() for p in m.model.parameters())/1e6, 'M')"
```

## 5. Configuration overview

| | `rtdetr-l-midfusion-RDSFA-Pro.yaml` | `rtdetr-l-midfusion-RDSFA.yaml` |
|---|---|---|
| Scale | `m` (max channels 1536) | `l` (max channels 1024) |
| ECMI at stage 1 | yes | no |
| Fusion | `RDRFusion` at P3/P4/P5 | `RDRFusion` at P3/P4/P5 |
| CSFR | `CrossScaleFusion` P3→P4→P5 | no |
| Decoder | `RTDETRDecoder(nc, 384, 500, 8)` | `RTDETRDecoder(nc)` |
| Reported params | ≈ 120 M | ≈ 97 M |

## 6. Notes

- Both modalities must be strictly paired by file name; the dataloader derives the
  infrared path with the `ir_replace` rule in the dataset YAML.
- `_cast_like()` inside `RDSFA.py` handles the AMP/half-precision mismatch that can
  appear when validating a half-precision checkpoint; keep it if you use `amp=True`.
- The offset predictor shares its weights across scales; changing the recurrence
  depth `T` and the maximum offset scale `s` is controlled inside
  `MultiScaleDeformableAlignment`.