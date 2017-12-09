from imgaug import augmenters as iaa
from imgaug import parameters as iap
import numpy as np
import matplotlib.image as mpimg
from matplotlib import pyplot as plt


images = np.array(
    [mpimg.imread("test.jpg") for _ in range(5)],   
    dtype=np.uint8
)

seq = iaa.Sequential([
    iaa.Fliplr(0.1), # horizontal flips

    iaa.Crop(percent=(0, 0.15)), # random crops
    
    iaa.Sometimes(0.5,   
        iaa.GaussianBlur(sigma=(0, 0.5))
    ),

    # Strengthen or weaken the contrast in each image.
    iaa.ContrastNormalization((0.75, 1.5)),

    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.1),
    
    # Make some images brighter and some darker.
    iaa.Multiply((0.8, 1.2), per_channel=0.3),
    
    # Apply affine transformations to each image.
    iaa.Affine(
        scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
        translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
        rotate=(-30, 30),
        #shear
    )
], random_order=True) 

images_aug = seq.augment_images(images)

for i in range(5):
    plt.subplot(1,5,i+1)
    plt.imshow(images_aug[i])
    plt.tight_layout()

plt.show()


