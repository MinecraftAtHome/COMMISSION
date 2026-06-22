# COMMISSION

## Definition
CUDA Optimized Massive Mushroom Island Standalone Search Including Only Necessities

CUDA app that uses various clever tricks to produce mushroom islands of adequate size.

## System Requirements

- Nvidia GPU (or AMD if you're lucky and you can transpile to HIP (currently unsupported)).
  - Tested Maxwell (sm_52) through Blackwell (sm_100)
  - 3GB+ VRAM (2976 MiB)
  - CUDA 12+ (tested, could work for older perhaps but that's untested.)
## Usage
- Clone the repo 
  `git clone https://github.com/MinecraftAtHome/COMMISSION.git`
- Enter the folder
  `cd COMMISSION`
- Make the program
  `make`
- Run the program
  `./main --devices 0,1,...`
### Runtime arguments
```
```
./main [--device <device>,<device>,...] [--threads <threads>] [--client <server_address>] [--server <listen_address>] [--output <output_file>] [--start <start_seed>] [--size <min_size>]
```

- client and server options aren't often used, they were created to network together multiple computers running this app.

```
### Info about device numbers
- These are just numbers given by the driver to identify your GPUs. You'll probably need to do this via trial and error if you have multiple gpus and you only want some of them running this.
- You could also omit the `--devices` option altogether. Then it'll just use `device 0`.
### Makefile info
- As of writing, the Makefile uses nvidia-smi to determine which gpu you have.
 - This is only necessary because Blackwell (sm\_100, or 50 series) performs worse when you compile native for it. So we compile targeting sm\_89 if you're on Lovelace (40 series) or newer, rather than compile for sm_100 for Blackwell.
 - If your GPU isn't 40 series or newer, it compiles with `-arch=native` by default. You can override this with `ARCH=<insert architecture>` i.e. `ARCH=sm_80` for 30 series.
- You can run the makefile without supplying any options, as all of the options have defaults.
- By default, your settings will be:
  - Small (normal) Biomes
  - Bound to world border (aka bounded)
  - Print interval at 4096
#### Options
 - ARCH
   - Change GPU target for PTX compilation
     - `ARCH=native` is default for GPUs 30 series or older
     - `ARCH=sm_89` is default for GPUs 40 series or newer
 - PRINT_INTERVAL (Warning: This is going to change in 1.4.1 to 256 by default!)
   - Change how often the program prints benchmarking info
     - Default is 4096, meaning every 4096 iterations of the full GPU pipeline, it will print the table with stats.
   - Example usage
     `make -B PRINT_INTERVAL=256`
     `make -B PRINT_INTERVAL=4096` (default)
 - UNBOUND
   - Change whether the app will check for mushroom islands that fall outside of the world border (30'000'000 in each direction)
   - Example usage
     `make -B UNBOUND=1`
   - Default is 0, which keeps it within the world border.
 - LARGE_BIOMES
   - Changes whether you're using normal (small) biomes, or large biomes.
   - Example usage
     `make -B LARGE_BIOMES=1` for large biomes
     `make -B LARGE_BIOMES=0` for small biomes (default)
