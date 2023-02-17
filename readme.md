<div align="center">This repository contains the implementation for the paper:<br />

[No-Reference Image Quality Assessment via Transformers, Relative Ranking, and Self-Consistency (WACV 2022)](https://arxiv.org/pdf/2108.06858.pdf) [Video](https://www.youtube.com/watch?v=Ph3TIqiIN34&ab_channel=AlirezaGolestaneh)
</div>


 <br />
  <br />
   <br />

<a href="https://replicate.com/arielreplicate/tres_iqa"><img src="https://replicate.com/arielreplicate/tres_iqa/badge"></a>


![wacv2021](https://user-images.githubusercontent.com/12434910/137831770-dd5d17da-fe83-431e-ac86-bebbe2810820.png)




## Creat Environment
This code is train and test on Ubuntu 16.04 while using  Anaconda, python `3.6.6`, and pytorch `1.8.0`.
To set up the evironment run:
`conda env create -f environment.yml`
after installing the virtuall env you should be able to run `python -c "import torch; print(torch.__version__)" ` in the terminal and see `1.8.0`




## Datasets
In this work we use 7 datasets for evaluation ([LIVE](https://live.ece.utexas.edu/research/quality/subjective.htm), [CSIQ](https://s2.smu.edu/~eclarson/csiq.html), [TID2013](http://www.ponomarenko.info/tid2013.htm), [KADID10K](http://database.mmsp-kn.de/kadid-10k-database.html), [CLIVE](https://live.ece.utexas.edu/research/ChallengeDB/), [KonIQ](http://database.mmsp-kn.de/koniq-10k-database.html), [LIVEFB](https://baidut.github.io/PaQ-2-PiQ/))

To start training please make sure to follow the correct folder structure  for each of the aformentioned datasets as provided bellow:

<details>
<summary>LIVE</summary>
 

```
live
    |--fastfading
    |    |  ...     
    |--blur
    |    |  ... 
    |--jp2k
    |    |  ...     
    |--jpeg
    |    |  ...     
    |--wn
    |    |  ...     
    |--refimgs
    |    |  ...     
    |--dmos.mat
    |--dmos_realigned.mat
    |--refnames_all.mat
    |--readme.txt
```
</details>

<details>
<summary>CSIQ</summary>

```
csiq
    |--dst_imgs_all
    |    |--1600.AWGN.1.png
    |    |  ... (you need to put all the distorted images here)
    |--src_imgs
    |    |--1600.png
    |    |  ...
    |--csiq.DMOS.xlsx
    |--csiq_label.txt
```

</details>



<details>
<summary>TID2013</summary>

```
tid2013
    |--distorted_images
    |--reference_images
    |--mos.txt
    |--mos_std.txt
    |--mos_with_names.txt
    |--readme
```
</details>



<details>
<summary>KADID10K</summary>

```
kadid10k
    |--distorted_images
    |    |--I01_01_01.png
    |    |  ...    
    |--reference_images
    |    |--I01.png
    |    |  ...    
    |--dmos.csv
    |--mv.sh.save
    |--mvv.sh
```
</details>


<details>
<summary>CLIVE</summary>


```
clive
    |--Data
    |    |--I01_01_01.png
    |    |  ...    
    |--Images
    |    |--I01.png
    |    |  ...    
    |--ChallengeDB_release
    |    |--README.txt
    |--dmos.csv
    |--mv.sh.save
    |--mvv.sh
```
</details>


<details>
<summary>KonIQ</summary>


 ```
fblive
    |--1024x768
    |    |  992920521.jpg 
    |    |  ... (all the images should be here)     
    |--koniq10k_scores_and_distributions.csv
```
</details>



<details>
<summary>LIVEFB</summary>


 ```
fblive
    |--FLIVE
    |    |  AVA__149.jpg    
    |    |  ... (all the images should be here)     
    |--labels_image.csv
```
</details>


## Training
The training scrips are provided in the `run.sh`. Please change the paths correspondingly. 
Please note that to achive the same performace the parameters should match the ones in the `run.sh` files.





## Pretrained models
The pretrain models are provided [here](https://drive.google.com/drive/folders/149CcTlnVX3fXmNFmnFRwUmX0PL8jd5vf?usp=sharing).

## Acknowledgement
This code is borrowed parts from [HyperIQA](https://github.com/SSL92/hyperIQA) and [DETR](https://github.com/facebookresearch/detr). 


## FAQs
<details>
- <summary>What is the difference between self-consistency and ensembling? and will the self-consistency increase the interface time?</summary>
In ensampling methods, we need to have several models (with different initializations) and ensemble the results during the training and testing, but in our self-consistency model, we enforce one model to have consistent performance for one network during the training while the network has an input with different transformations.
Our self-consistency model has the same interface time/parameters in the testing similar to the model without self-consistency. In other words, we are not adding any new parameters to the network and it won't affect the interface.
</details>

 
<details>
- <summary>What is the difference between self-consistency and augmentation?</summary>
In augmentation, we augment an input and send it to one network, so although the network will become robust to different augmentation, it will never have the chance of enforcing the outputs to be the same for different versions of an input at the same time. In our self-consistency approach, we force the network to have a similar output for an image with a different transformation (in our case horizontal flipping) which leads to more robust performance. Please also note that we still use augmentation during the training, so our model is benefiting from the advantages of both augmentation and self-consistency. Also, please see Fig. 1 in the main paper, where we showed that models that used augmentation alone are sensitive to simple transformations.
</details>


<details>
- <summary>Why does the relative ranking loss apply to the samples with the highest and lowest quality scores, why not applying it to all the samples?</summary>
1) We did not see a significant improvement by applying our ranking loss to all the samples within each batch compared to the case that we just use extreme cases. 
2) Considering more samples  lead to more gradient back-propagation and therefore more computation during the training which causes slower training.
</details>



## Citation
If you find this work useful for your research, please cite our paper:
```
@InProceedings{golestaneh2021no,
  title={No-Reference Image Quality Assessment via Transformers, Relative Ranking, and Self-Consistency},
  author={Golestaneh, S Alireza and Dadsetan, Saba and Kitani, Kris M},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  pages={3209--3218},
  year={2022}
}
```

If you have any questions about our work, please do not hesitate to contact isalirezag@gmail.com


