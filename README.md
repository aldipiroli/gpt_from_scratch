# GPT from scratch
Implementing from scratch the paper ["Improving Language Understanding by Generative Pre-Training"](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) OpenAI GPT-1.

> Note: This implementation is inspired by the great [tutorial](https://www.youtube.com/watch?v=kCc8FmEb1nY) of Andrej Karpathy.
 
### Clone and install dependencies
``` 
git clone https://github.com/aldipiroli/gpt_from_scratch
pip install -r requirements.txt
``` 
### Train 
``` 
cd python 
python train.py config/gpt_config.yaml
```
### Inference
``` 
python inference.py --config config/gpt_config.yaml --ckpt path/to/ckpt --prompt "the prompt"
``` 