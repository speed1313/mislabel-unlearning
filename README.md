# mislabel-unlearning
Code for the domestic conference paper, "Improving Generalization Performance of Trained Models by Unlearning Mislabeled Data", [16th Forum on Data Engineering and Information Management (DEIM2024)](https://confit.atlas.jp/guide/event/deim2024/top?lang=en).


![proposed method](./images/proposed.png)



# How to use
- setup
```
$ rye sync
```

- Detection of mislabeled data
```
$ rye run python3 src/mnist_detection.py
```

- Unlearning mislabeled data
```
$ rye run python3 src/mnist_unlearning.py
```

- Unlearning detected data
```
$ rye run python3 src/mnist_pipelnie.py
```
