cfg = {
    "CKPT_DIR": "../artifacts/ckpts",
    "LOG_DIR": "../artifacts/logs",
    "IMG_OUT_DIR": "../artifacts/img",
    "MODEL": {
        "model_name": "GPTModel",
        "context_len": 32,
        "embed_size": 128,
        "transfomer_layers": 8,
        "num_heads": 4,
        "dropout": 0.1,
    },
    "DATA": {
        "root_dir": "../data/tiny_shakespeare",
        "target_url": "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
        "batch_size": 16,
    },
    "OPTIM": {
        "loss": "",
        "optimizer": "AdamW",
        "lr": 0.0003,
        "num_iterations_train": 1000,
        "num_iterations_val": 200,
        "num_epochs": 50,
        "eval_every": 5,
        "gradient_clip": True,
        "scheduler": "",
        "T_max": 100,
        "eta_min": 0.000001,
    },
}
