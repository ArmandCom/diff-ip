#Train querier with cmi loss with attribute as the input

## Training diffusion model

```
python  main_clevr.py --num_workers 8 --batch_size 16 --run_name attr-CelebA_diff_only --name attr-CelebA_diff_only --sampling random --experiment_type attributes --data_dir img_align_celeba --save_dir clevr_diff_ip  --epoch 500 --loss_type l2
```

## Training querier
```
python  main_clevr.py --num_workers 8 --batch_size 16 --run_name attr-CelebA_diff_only --name attr-CelebA_querier_only --sampling biased --experiment_type attributes --data_dir img_align_celeba --save_dir clevr_diff_ip  --epoch 499 --loss_type l2 --cmi --train_querier --freeze_unet --sample_ids 9 19 29 39 
```

## License

