from dataclasses import dataclass


@dataclass
class DataConfig:
    comics_data_path: str
    vgg_feats_path: str
    vocab_path: str
    folds_dir: str
    megabatch_size: int
    num_workers: int
    pin_memory: bool


@dataclass
class TaskConfig:
    difficulty: str


@dataclass
class ModelConfig:
    name: str
    n_epochs: int
    batch_size: int
    iters_to_accumulate: int
    lr: float


@dataclass
class ExperimentConfig:
    data: DataConfig
    task: TaskConfig
    model: ModelConfig
