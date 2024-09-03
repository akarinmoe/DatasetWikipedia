# Photo Generation Using Wikipedia Corpus



## Environment:

```
python = 3.8.19
```



## QuickStart:

Login your account of huggingface using access token first

```
pip install -r requirements.txt

python predownload.py
```

After downloading the datasets locally

```
mkdir outputimg

python rendermulti.py
```



## Tips:

Even if the code includes `os.environ['HF_ENDPOINT'] = https://hf-mirror.com`, you could also run `export HF_ENDPOINT=https://hf-mirror.com`.

The number of processes is represented in the `rendermulti.py` by the variable `num_processes`, which defaults to 1

 