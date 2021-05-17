# memento_keras
Repository for the Memento Project, with Keras code.

## Data required
Running this code requires the Memento10k dataset. Request it through our website: http://memento.csail.mit.edu/ 

## Inference
You can perform simple inference with the Memento10k_inference_standalone.ipynb notebook. Instructions are provided in the notebook.

## Training
You can retrain our models with one of our training notebooks. We recommend using:
- Memento10k_training_with_captions if captions are available,
- Memento10k_training_with_alphas if decay information is available for your videos,
- Memento10k_training_NO_alphas if you just want to train on memorability scores.

## Citation
Please cite our paper when using our code: 
```
@article{DBLP:journals/corr/abs-2009-02568,
  author    = {Anelise Newman and
               Camilo Fosco and
               Vincent Casser and
               Allen Lee and
               Barry A. McNamara and
               Aude Oliva},
  title     = {Multimodal Memorability: Modeling Effects of Semantics and Decay on
               Video Memorability},
  journal   = {CoRR},
  volume    = {abs/2009.02568},
  year      = {2020},
  url       = {https://arxiv.org/abs/2009.02568},
  archivePrefix = {arXiv},
  eprint    = {2009.02568},
  timestamp = {Thu, 17 Sep 2020 09:01:51 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/abs-2009-02568.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
