from dataclasses import MISSING, dataclass
import typing
import multiprocessing as mp

from hydra.core.config_store import ConfigStore
from map_elites import common as cm

@dataclass
class Config:
    """This class provides a "schema" for the config file, validating types."""

    RANDOM_INIT_NETS: int = 1#6 # INIT Nets to be accepted in archive
    INIT_NUM_NETS: int = 1#4 # INIT Nets created per generation
    ROLL_OUTS: int = 1#6 # For GPU training. ROLL_OUTS * (INIT_NUM_NETS or NUM_NETS) = Total nets created in each generation
    NUM_NETS: int = 1#4 # Mutation and crossover nets to be created
    START_FROM_CHECKPOINT: bool = False
    RANDOM_NETWORKS: bool = False

    MUTATION: str = "codegen-6B-mono"
    #MUTATION: str = "codex"

    GENERATIONS: int = 200
    NET_TRAINING_EPOCHS: int = 2
    TEMPERATURE: float = 0.0000000

    DEVICE: str = "cpu" # options: ["cuda", "cpu", "both"]

    NUM_PROCESSES: int = mp.cpu_count() - 1
    RAY: bool = True
    NUM_CPUS = NUM_PROCESSES
    NUM_GPUS = 1 # per task
    NUM_GPUS_TOTAL = 1 # total available
 
    # MAP-ELITES

    DIM_MAP: int = 2
    N_NICHES: int = 100
    

    #SAVE_DIR: str = "mnt/lustre/users/mnasir/NAS-LLM"

    SAVE_DIR: str = "./"
             

cs = ConfigStore.instance()
cs.store(name="base_config", node=Config)
