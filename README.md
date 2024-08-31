# photo generation using wikipedia corpus



## Environment:

```
python = 3.8.19
```



## QuickStart:

login your account of huggingface using access token first

```
pip install -r requirements.txt

python rendermulti.py
```



## tips:

Even if the code includes `os.environ['HF_ENDPOINT'] = https://hf-mirror.com`, you could also run `export HF_ENDPOINT=https://hf-mirror.com`.

The number of processes is represented in the rendermulti.py by the variable `num_processes`, which defaults to 1

The dataset loaded for the first time is saved in the .cache folder by default, and then loaded locally

 