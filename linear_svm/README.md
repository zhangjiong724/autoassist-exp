The code is based on LIBSVM, download the datasets from:
https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/


algorithm = 0 (for SGD), 10 (for importance sampling) and 20 for shrinking

samplimg = 0 (uniform sampling), 1 (loss as imprtance weight) and 2 (grad norm as importance weight)

For example for SGD:

```
./train -s 23 -c 1.0\
        -n [num_updates] \
		-a 0  \
		-S 0   \
        -t [test_data]   \
        [training_data]
```


