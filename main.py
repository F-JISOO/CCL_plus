import os
import random
import argparse
import numpy as np
import torch

from utils.config import _C as cfg
from utils.logger import setup_logger


def get_trainer(algorithm_name):
    if algorithm_name == "finessl":
        from algorithm.finessl import Trainer
    elif algorithm_name == "fixmatch":
        from algorithm.fixmatch import Trainer
    elif algorithm_name == "freematch":
        from algorithm.freematch import Trainer
    elif algorithm_name == "softmatch":
        from algorithm.softmatch import Trainer
    elif algorithm_name == "flexmatch":
        from algorithm.flexmatch import Trainer
    elif algorithm_name == "abc":
        from algorithm.abc import Trainer
    elif algorithm_name == "acr":
        from algorithm.acr import Trainer
    elif algorithm_name == "supervised":
        from algorithm.supervised import Trainer
    elif algorithm_name == "debiaspl":
        from algorithm.debiaspl import Trainer
    elif algorithm_name == "daso":
        from algorithm.daso import Trainer
    elif algorithm_name == "CCL":
        from algorithm.CCL import Trainer
    elif algorithm_name == "Meta":
        from algorithm.Meta import Trainer
    elif algorithm_name == "SCAD":
        from algorithm.SCAD import Trainer
    elif algorithm_name == "CPG":
        from algorithm.CPG import Trainer
    elif algorithm_name == "CoLA":
        from algorithm.CoLA import Trainer
    else:
        raise ValueError(f"Unknown algorithm: {algorithm_name}")
    return Trainer


def main(args):
    cfg.defrost()
    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)
    # cfg.freeze()

    if cfg.seed is not None:
        seed = cfg.seed
        print("Setting fixed seed: {}".format(seed))
        random.seed(seed)
        np.random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if cfg.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

    imbl = cfg.DATA.IMB_L
    imbu = cfg.DATA.IMB_U
    numl = cfg.DATA.NUM_L
    print(cfg.algorithm)
    C = cfg.DATA.NUMBER_CLASSES
    if not hasattr(cfg, "cut1"):
        cfg.cut1 = C // 3
    if not hasattr(cfg, "cut2"):
        cfg.cut2 = (2 * C) // 3
    if cfg.output_dir is None:
        cfg.output_dir = os.path.join(
            "./output",
            os.path.basename(args.cfg).rstrip(".yaml"),
            cfg.algorithm,
            cfg.clip_type)
    else:
        cfg.output_dir = os.path.join(
            cfg.output_dir,
            cfg.DATA.NAME,
            cfg.algorithm,
            cfg.clip_type,
            f"NUML{numl}_imbl{imbl}_imbu{imbu}"
        )

    print("** Config **")
    print(cfg)

    setup_logger(cfg.output_dir)
    Trainer = get_trainer(cfg.algorithm)
    trainer = Trainer(cfg)
    if cfg.eval_only:
        cfg.model_dir = cfg.model_dir if cfg.model_dir is not None else cfg.output_dir
        cfg.load_epoch = cfg.load_epoch if cfg.load_epoch is not None else cfg.num_epochs
        trainer.load_model(cfg.model_dir, epoch=cfg.load_epoch)
        trainer.test_and_save_pseudo_dist("fixmatch_pseudo_stats_test_aves.pt")
        return

    
    if cfg.clip_type == "":
        print("Training white-box model...")
        trainer.train()
    else:
        print("Training black-box model...")
        trainer.train_black_model()

        # acc = trainer.offline_ensemble_test()
        # print(acc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="", help="path to config file")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER,
                        help="modify config options using the command-line")
    args = parser.parse_args()
    main(args)
