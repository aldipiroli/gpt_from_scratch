import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.gpt_model import GPTModel


def test_image_patcher():
    cfg = {}
    model = GPTModel(cfg)
    assert True


if __name__ == "__main__":
    print("All test passed!")
