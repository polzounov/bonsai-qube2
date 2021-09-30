## Run this:

If on raspberry pi:

```bash
docker load < qq-mc1.tar.gz
docker run -p 5000:5000 qq-mc1
python3 run.py -c brain -p 5000
```

<br>
If on a x64 computer:

```bash
docker load < qq-mc1-x64.tar.gz
docker run -p 5000:5000 qq-mc1-x64
python3 run.py -c brain -p 5000
```
