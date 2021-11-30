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
    difficulty: str = 'easy'


@dataclass
class ModelConfig:
    n_epochs: int = (10,)
    batch_size: int = (32,)
    iters_to_accumulate: int = (2,)
    lr: float = (1e-5,)


@dataclass
class ExperimentConfig:
    data: DataConfig
    task: TaskConfig
    model: ModelConfig
