# Diffusion-Based Variational Information Pursuit for Interpretable Generation

This is the offical repository for *Diffusion-Based Variational Information Pursuit for Interpretable Generation*. The codebase is in the development phase. For more information contact acomasma@gmail.com.


## Requirements
Please check out `requirements.txt` for detailed requirements. We also use `wandb` to moderate training and testing performance. One may remove lines related to `wandb` and switch to something different if they desire. 


## Training
There are two stages of training: *Initial Random Sampling (IRS)* and *Subsequent Biased Sampling (SBS)*.

To run IRS:

```
python3 main_mnist.py \
  --epochs 100 \
  --data mnist \
  --batch_size 128 \
  --max_queries 676 \
  --max_queries_test 21 \
  --lr 0.0001 \
  --tau_start 1.0 \
  --tau_end 0.2 \
  --sampling random \
  --seed 0 \
  --name mnist_random
```

To run SBS:

```
python3 main_mnist.py \
  --epochs 20 \
  --data mnist \
  --batch_size 128 \
  --max_queries 21 \
  --max_queries_test 21 \
  --lr 0.0001 \
  --tau_start 0.2 \
  --tau_end 0.2 \
  --sampling biased \
  --seed 0 \
  --ckpt_path <CKPT_PATH> \
  --name mnist_biased
```
where `<CKPT_PATH>` is the path to the pre-trained model using IRS.
## License

