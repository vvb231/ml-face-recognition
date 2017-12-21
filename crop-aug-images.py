{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "#Capture a video\n",
    "import cv2\n",
    "\n",
    "cap = cv2.VideoCapture(0)\n",
    "w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH ))\n",
    "h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT ))\n",
    "fourcc = cv2.VideoWriter_fourcc(*'MJPG')\n",
    "out = cv2.VideoWriter('video.avi',fourcc, 20.0, (w,h))\n",
    "\n",
    "while(True):\n",
    "    ret, frame = cap.read()\n",
    "#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)\n",
    "    try:\n",
    "        out.write(frame)\n",
    "    except:\n",
    "        print('ERROR - Not writting to file') \n",
    "    cv2.imshow('frame', frame)\n",
    "    if cv2.waitKey(1) & 0xFF == ord('p'):\n",
    "        break\n",
    "\n",
    "cap.release()\n",
    "out.release()\n",
    "cv2.destroyAllWindows()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 52,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Dataset Ready!\n"
     ]
    }
   ],
   "source": [
    "# Framing video into images\n",
    "import numpy as np\n",
    "import os\n",
    "\n",
    "# Playing video from file:\n",
    "cap = cv2.VideoCapture('video.avi')\n",
    "\n",
    "try:\n",
    "    if not os.path.exists('data1'):\n",
    "        os.makedirs('data1')\n",
    "except OSError:\n",
    "    print ('Error: Creating directory of data')\n",
    "\n",
    "currentFrame = 0\n",
    "nameCount = 0\n",
    "while(currentFrame < 200): \n",
    "    # Capture frame-by-frame\n",
    "    ret, frame = cap.read()\n",
    "\n",
    "    # Saves image of the current frame in jpg file\n",
    "    name = './data1/frame' + str(nameCount) + '.jpg'\n",
    "#     print ('Creating...' + name)\n",
    "    cv2.imwrite(name, frame)\n",
    "    nameCount +=1\n",
    "    # To stop duplicate images\n",
    "    currentFrame += 1\n",
    "print ('Dataset Ready!')\n",
    "# When everything done, release the capture\n",
    "cap.release()\n",
    "cv2.destroyAllWindows()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 47,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Populating the interactive namespace from numpy and matplotlib\n"
     ]
    }
   ],
   "source": [
    "import keras\n",
    "from imgaug import augmenters as iaa\n",
    "from imgaug import parameters as iap\n",
    "\n",
    "import matplotlib.image as mpimg\n",
    "from matplotlib import pyplot as plt\n",
    "%pylab inline"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 48,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(100, 720, 1280, 3)"
      ]
     },
     "execution_count": 48,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "images = np.array([mpimg.imread(\"data1/frame{}.jpg\".format(i)) for i in range(100)], dtype = np.uint8)\n",
    "images.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 80,
   "metadata": {},
   "outputs": [],
   "source": [
    "seq = iaa.Sequential([\n",
    "    iaa.Fliplr(0.2), # horizontal flips\n",
    "    \n",
    "    # Small gaussian blur with random sigma between 0 and 0.5.\n",
    "    iaa.Sometimes(0.8,\n",
    "        iaa.GaussianBlur(sigma=(0, 0.5))\n",
    "        \n",
    "    ),\n",
    "    iaa.Sometimes(0.7,\n",
    "         iaa.Sharpen(alpha=(0.0, 1.0), lightness=(0.75, 2.0))\n",
    "    ),\n",
    "    \n",
    "    # Strengthen or weaken the contrast in each image.\n",
    "    iaa.ContrastNormalization((0.5, 1.5)),\n",
    "    # Add gaussian noise.\n",
    "    # For 50% of all images, we sample the noise once per pixel.\n",
    "    # For the other 50% of all images, we sample the noise per pixel AND\n",
    "    # channel. This can change the color (not only brightness) of the\n",
    "    # pixels.\n",
    "    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.1),\n",
    "    \n",
    "    # Apply affine transformations to each image.\n",
    "    # Scale/zoom them, translate/move them, rotate them and shear them.\n",
    "    iaa.Sometimes(0.5,\n",
    "        iaa.Affine(\n",
    "            translate_percent={\"x\": (-0.1, 0.1), \"y\": (-0.1, 0.1)},\n",
    "            rotate=(-10, 10),\n",
    "        )\n",
    "     )\n",
    "], random_order=True) # apply augmenters in random order\n",
    "\n",
    "images_aug = seq.augment_images(images)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 54,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\ngg==\n",
      "text/plain": [
       "<matplotlib.figure.Figure at 0x11e37ffd0>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "for i in range(5):\n",
    "    plt.subplot(1,5,i+1)\n",
    "    plt.imshow(images[i])\n",
    "    plt.tight_layout()\n",
    "\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 82,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Augmented images cropped and saved in folder data\n"
     ]
    }
   ],
   "source": [
    "#Take augmented images from above detect faces and crop faces - saves cropped images in 'data' folder in \n",
    "#the same directory \n",
    "\n",
    "face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')\n",
    "try:\n",
    "    if not os.path.exists('data'):\n",
    "        os.makedirs('data')\n",
    "except OSError:\n",
    "    print ('Error: Creating directory of data')\n",
    "for i in range(99):\n",
    "    img= images_aug[i]\n",
    "    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)\n",
    "    \n",
    "    faces = face_cascade.detectMultiScale(gray, 1.1, 5)\n",
    "    for (x,y,w,h) in faces:\n",
    "        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)\n",
    "        roi_gray = gray[y:y+h, x:x+w]\n",
    "        roi_color = img[y:y+h, x:x+w]\n",
    "\n",
    "    crop_img = img[y: y + h, x: x + w]     \n",
    "\n",
    "    # Saves image of the current frame in jpg file\n",
    "    name = './data/frame' + str(currentFrame) + '.jpg'\n",
    "    #print ('Creating...' + name)\n",
    "    cv2.imwrite(name, crop_img)\n",
    "\n",
    "    # To stop duplicate images\n",
    "    currentFrame += 1\n",
    "    #plt.imshow(images_aug[i])\n",
    "print ('Augmented images cropped and saved in folder data')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.1"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
