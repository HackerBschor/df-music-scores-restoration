# Deep Learning Music Scores Restoration 🎼 🖥️

Topic: `Computer Vision` 
<br>
Type: `Image Generation / Image Transformation`

This project aims to restore damaged music scores using deep learning methods. 
We want to use **Autoencoders** & **Transformers** to perform several generativ AI tasks to perform 
**up-scaling**, **de-blurring** & **enriching** to transform damaged music score into a cleaner and more readable form. 

### Dataset 

To train the model, we intend to generate a huge dataset of 'perfect'
music scores by converting [MusicXML](https://de.wikipedia.org/wiki/MusicXML) 
files into images files using the [Verovio python interface](https://pypi.org/project/verovio/). 
Afterwards, 
we use some [image augmentation](https://albumentations.ai/) techniques or classic [image processing methods](https://pillow.readthedocs.io/en/stable/)
to 'damage' the music-score-images.

We think of generating scores of existing [open-source/ licence free](http://mscorelib.com/actree/) 
and randomly generated music (to avoid overfitting).

Using the Verovio library, we can generate images like this from a [MusicXML file](examples/) 
(67th page of the Don Giovanni Overture by Mozart):
![Mozart-Don_Giovanni_66](examples%2Frender%2FMozart-Don_Giovanni_66.pdf)


### Training

Using the damaged and the non-damaged music scores of the training set, 
we will train a model on the 'perfect' scores and use the damaged ones as input.
We combine [Autoencoders](https://www.researchgate.net/publication/356423394_Denoising_Text_Image_Documents_using_Autoencoders) to denoise 
the images with a [Hybrid Attention Transformer](https://arxiv.org/abs/2205.04437v3) for a super-resolution 
and de-blurring.
We intend to use PyTorch to implement the two methods. 
Furthermore, we want to combine both methods into one model and train it end-to-end.

### Evaluation

TODO


### Future Work
In the future, 
we think of fine-tuning the model using real scans of old used music sheets and brand-new ones.


## References
* [Hybrid Attention Transformer for Image Super-Resolution](https://arxiv.org/abs/2205.04437v3)
* [HAT Github](https://github.com/XPixelGroup/HAT)
* [Denoising Text Image Documents using Autoencoders](https://www.researchgate.net/publication/356423394_Denoising_Text_Image_Documents_using_Autoencoders)
* [Denoising GitHub](https://github.com/Surya-Prakash-Reddy/Denoising-Documents)

## Work Breakdown structure

The following table gives a very rough estimate on the tasks 
and the time needed to finish them: 

| Task                  | Estimated Time         |
|-----------------------|------------------------|
| Research              | 2 Days                 |
| Dataset Generation    | 1 Day                  |
| Model Creation        | 3 Days                 |
| Model Training        | 1 Day (no work for me) |
| Model Evaluation      | 1 Days                 |
| Integration           | 2 Days                 |
| Report & Presentation | 1 Day                  |


### Disclaimer
This repository is created by Nicolas Bschor as part of 
the Applied Deep Learning course at the Technical University of Vienna.