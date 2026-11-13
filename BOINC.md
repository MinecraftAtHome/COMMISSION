# COMMISSION BOINC app

## Build

```
make BOINC=1 ARCH=all [LARGE_BIOMES=1] [UNBOUND=1]   # main-{sb,lb}[u]-boinc
```

- **Variants:** each `LARGE_BIOMES`/`UNBOUND` combination is a separate search
  and needs its own app.
- **Linux:** the binary needs glibc 2.34+ when built in the CI image (Ubuntu 22.04).

## Command line

```
--start <seed> --end <seed> [--size <min_size>]
```

- **Range:** `[start, end)` in signed 64-bit seeds.
- **Alignment:** use multiples of 2^28 (one GPU batch). Nothing is reported
  twice across workunits either way.
- **GPU:** taken from `init_data.xml`. The app runs one GPU per task.

## Output (`out.txt`)

```
<seed> <x> <z> <area>
...
done <start> <end> <min_size> <sb|lb> <bounded|unbound>
```

- **Islands:** sorted, with duplicates removed.
- **`done` line:** always written, so an empty workunit still has output.
- **Validation:** Windows and Linux produce byte-identical files, so
  `sample_bitwise_validator` works. The one exception: the GPU pre-filter uses
  `-use_fast_math`, so different GPU architectures could in theory disagree on
  a borderline candidate.

## Server notes

- **Resources:** NVIDIA, about 3 GB of VRAM, 1 GPU + 1 CPU per task (the CUDA
  sync spin-waits).
- **Checkpoints:** written at the volunteer's checkpoint interval and resumed
  automatically.
- **Workunit size:** an RTX 5070 does about 15.9 × 10^9 seeds/s. That's ~18 min
  for 2^44 seeds. Size workunits for slower GPUs.

## Not done yet

- **CUDA errors** fail the task instead of calling `boinc_temporary_exit`.
- **Real client:** only tested in standalone mode, not under an actual BOINC
  client.
