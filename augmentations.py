from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import numpy as np
import torch
# pylint:disable=g-multiple-import
from PIL import ImageOps, ImageEnhance, ImageFilter, Image
from torchvision.transforms import transforms
# pylint:enable=g-multiple-import


IMAGE_SIZE = 32
# What is the dataset mean and std of the images on the training set
MEANS = [0.49139968, 0.48215841, 0.44653091]
STDS = [0.24703223, 0.24348513, 0.26158784]
PARAMETER_MAX = 10  # What is the max 'level' a transform could be predicted


def random_flip(x):
  """Flip the input x horizontally with 50% probability."""
  if np.random.rand(1)[0] > 0.5:
    return np.fliplr(x)
  return x


def zero_pad_and_crop(img, amount=4):
  """Zero pad by `amount` zero pixels on each side then take a random crop.
  Args:
    img: numpy image that will be zero padded and cropped.
    amount: amount of zeros to pad `img` with horizontally and verically.
  Returns:
    The cropped zero padded img. The returned numpy array will be of the same
    shape as `img`.
  """
  padded_img = np.zeros((img.shape[0] + amount * 2, img.shape[1] + amount * 2,
                         img.shape[2]))
  padded_img[amount:img.shape[0] + amount, amount:
             img.shape[1] + amount, :] = img
  top = np.random.randint(low=0, high=2 * amount)
  left = np.random.randint(low=0, high=2 * amount)
  new_img = padded_img[top:top + img.shape[0], left:left + img.shape[1], :]
  return new_img


def create_cutout_mask(img_height, img_width, num_channels, size):
  """Creates a zero mask used for cutout of shape `img_height` x `img_width`.
  Args:
    img_height: Height of image cutout mask will be applied to.
    img_width: Width of image cutout mask will be applied to.
    num_channels: Number of channels in the image.
    size: Size of the zeros mask.
  Returns:
    A mask of shape `img_height` x `img_width` with all ones except for a
    square of zeros of shape `size` x `size`. This mask is meant to be
    elementwise multiplied with the original image. Additionally returns
    the `upper_coord` and `lower_coord` which specify where the cutout mask
    will be applied.
  """
  assert img_height == img_width

  # Sample center where cutout mask will be applied
  height_loc = np.random.randint(low=0, high=img_height)
  width_loc = np.random.randint(low=0, high=img_width)

  # Determine upper right and lower left corners of patch
  upper_coord = (max(0, height_loc - size // 2), max(0, width_loc - size // 2))
  lower_coord = (min(img_height, height_loc + size // 2),
                 min(img_width, width_loc + size // 2))
  mask_height = lower_coord[0] - upper_coord[0]
  mask_width = lower_coord[1] - upper_coord[1]
  assert mask_height > 0
  assert mask_width > 0

  mask = np.ones((img_height, img_width, num_channels))
  zeros = np.zeros((mask_height, mask_width, num_channels))
  mask[upper_coord[0]:lower_coord[0], upper_coord[1]:lower_coord[1], :] = (
      zeros)
  return mask, upper_coord, lower_coord


def cutout_numpy(img, size=16):
  """Apply cutout with mask of shape `size` x `size` to `img`.
  The cutout operation is from the paper https://arxiv.org/abs/1708.04552.
  This operation applies a `size`x`size` mask of zeros to a random location
  within `img`.
  Args:
    img: Numpy image that cutout will be applied to.
    size: Height/width of the cutout mask that will be
  Returns:
    A numpy tensor that is the result of applying the cutout mask to `img`.
  """
  img_height, img_width, num_channels = (img.shape[0], img.shape[1],
                                         img.shape[2])
  assert len(img.shape) == 3
  mask, _, _ = create_cutout_mask(img_height, img_width, num_channels, size)
  return img * mask


def float_parameter(level, maxval):
  """Helper function to scale `val` between 0 and maxval .
  Args:
    level: Level of the operation that will be between [0, `PARAMETER_MAX`].
    maxval: Maximum value that the operation can have. This will be scaled
      to level/PARAMETER_MAX.
  Returns:
    A float that results from scaling `maxval` according to `level`.
  """
  return float(level) * maxval / PARAMETER_MAX


def int_parameter(level, maxval):
  """Helper function to scale `val` between 0 and maxval .
  Args:
    level: Level of the operation that will be between [0, `PARAMETER_MAX`].
    maxval: Maximum value that the operation can have. This will be scaled
      to level/PARAMETER_MAX.
  Returns:
    An int that results from scaling `maxval` according to `level`.
  """
  return int(level * maxval / PARAMETER_MAX)


def pil_wrap(img):
  """Convert the `img` numpy tensor to a PIL Image."""
  return Image.fromarray(
      np.uint8((img * STDS + MEANS) * 255.0)).convert('RGBA')


def pil_unwrap(pil_img):
  """Converts the PIL img to a numpy array."""
  pic_array = (np.array(pil_img.getdata()).reshape((32, 32, 4)) / 255.0)
  i1, i2 = np.where(pic_array[:, :, 3] == 0)
  pic_array = (pic_array[:, :, :3] - MEANS) / STDS
  pic_array[i1, i2] = [0, 0, 0]
  return pic_array


def apply_policy(policy, img):
  """Apply the `policy` to the numpy `img`.
  Args:
    policy: A list of tuples with the form (name, probability, level) where
      `name` is the name of the augmentation operation to apply, `probability`
      is the probability of applying the operation and `level` is what strength
      the operation to apply.
    img: Numpy image that will have `policy` applied to it.
  Returns:
    The result of applying `policy` to `img`.
  """
  pil_img = pil_wrap(img)
  for xform in policy:
    assert len(xform) == 3
    name, probability, level = xform
    xform_fn = NAME_TO_TRANSFORM[name].pil_transformer(probability, level)
    pil_img = xform_fn(pil_img)
  return pil_unwrap(pil_img)


class TransformFunction(object):
  """Wraps the Transform function for pretty printing options."""

  def __init__(self, func, name):
    self.f = func
    self.name = name

  def __repr__(self):
    return '<' + self.name + '>'

  def __call__(self, pil_img):
    return self.f(pil_img)


class TransformT(object):
  """Each instance of this class represents a specific transform."""

  def __init__(self, name, xform_fn):
    self.name = name
    self.xform = xform_fn

  def pil_transformer(self, probability, level):

    def return_function(im):
      if random.random() < probability:
        im = self.xform(im, level)
      return im

    name = self.name + '({:.1f},{})'.format(probability, level)
    return TransformFunction(return_function, name)

  def do_transform(self, image, level):
    f = self.pil_transformer(PARAMETER_MAX, level)
    return f(image)


################## Transform Functions ##################
aug_ohl_list = []
aug_name_ls = []
identity = TransformT('identity', lambda pil_img, level: pil_img)
# identity_ohl = TransformT('identity_ohl', lambda pil_img: pil_img)
identity_ohl = lambda pil_img: identity.do_transform(pil_img, 0)
aug_ohl_list.append(identity_ohl)
aug_name_ls.append('identity.')
flip_lr = TransformT(
    'FlipLR',
    lambda pil_img, level: pil_img.transpose(Image.FLIP_LEFT_RIGHT))

flip_lr_ohl = lambda pil_img: flip_lr.do_transform(pil_img, 0)
aug_ohl_list.append(flip_lr_ohl)
aug_name_ls.append('FlipLR.')
flip_ud = TransformT(
    'FlipUD',
    lambda pil_img, level: pil_img.transpose(Image.FLIP_TOP_BOTTOM))
aug_name_ls.append('FlipUD.')
flip_ud_ohl = lambda pil_img: flip_ud.do_transform(pil_img, 0)
aug_ohl_list.append(flip_ud_ohl)

# pylint:disable=g-long-lambda
auto_contrast = TransformT(
    'AutoContrast',
    lambda pil_img, level: ImageOps.autocontrast(
        pil_img.convert('RGB')).convert('RGBA'))
auto_contrast_ohl = lambda pil_img: auto_contrast.do_transform(pil_img, 0)
aug_ohl_list.append(auto_contrast_ohl)
aug_name_ls.append('AutoContrast.')
equalize = TransformT(
    'Equalize',
    lambda pil_img, level: ImageOps.equalize(
        pil_img.convert('RGB')).convert('RGBA'))
equalize_ohl = lambda pil_img: equalize.do_transform(pil_img, 0)
aug_ohl_list.append(equalize_ohl)
aug_name_ls.append('Equalize.')
invert = TransformT(
    'Invert',
    lambda pil_img, level: ImageOps.invert(
        pil_img.convert('RGB')).convert('RGBA'))
invert_ohl = lambda pil_img: invert.do_transform(pil_img, 0)
aug_ohl_list.append(invert_ohl)
aug_name_ls.append('Invert.')
# pylint:enable=g-long-lambda
blur = TransformT(
    'Blur', lambda pil_img, level: pil_img.filter(ImageFilter.BLUR))
blur_ohl = lambda pil_img: blur.do_transform(pil_img, 0)
aug_ohl_list.append(blur_ohl)
aug_name_ls.append('Blur.')
smooth = TransformT(
    'Smooth',
    lambda pil_img, level: pil_img.filter(ImageFilter.SMOOTH))
smooth_ohl = lambda pil_img: smooth.do_transform(pil_img, 0)
aug_ohl_list.append(smooth_ohl)
aug_name_ls.append('Smooth.')
aug_ohl_list_rotate=[]
def _rotate_impl(pil_img, level):
  """Rotates `pil_img` from -30 to 30 degrees depending on `level`."""
  degrees = int_parameter(level, 30)
  if random.random() > 0.5:
    degrees = -degrees
  return pil_img.rotate(degrees)

M_list = [0, 2, 10, 14]
#M_list = [5]
rotate = TransformT('Rotate', _rotate_impl)
for m in M_list:
  mop = lambda pil_img: rotate.do_transform(pil_img, m)
  aug_ohl_list.append(mop)
  aug_ohl_list_rotate.append(mop)
  aug_name_ls.append('Rotate.'+str(m))

def _posterize_impl(pil_img, level):
  """Applies PIL Posterize to `pil_img`."""
  level = int_parameter(level, 4)
  return ImageOps.posterize(pil_img.convert('RGB'), 4 - level).convert('RGBA')

posterize = TransformT('Posterize', _posterize_impl)
for m in M_list:
  mop = lambda pil_img: posterize.do_transform(pil_img, m)
  aug_ohl_list.append(mop)
  aug_name_ls.append('Posterize.'+str(m))

def _shear_x_impl(pil_img, level):
  """Applies PIL ShearX to `pil_img`.
  The ShearX operation shears the image along the horizontal axis with `level`
  magnitude.
  Args:
    pil_img: Image in PIL object.
    level: Strength of the operation specified as an Integer from
      [0, `PARAMETER_MAX`].
  Returns:
    A PIL Image that has had ShearX applied to it.
  """
  level = float_parameter(level, 0.3)
  if random.random() > 0.5:
    level = -level
  return pil_img.transform((32, 32), Image.AFFINE, (1, level, 0, 0, 1, 0))


shear_x = TransformT('ShearX', _shear_x_impl)
for m in M_list:
  mop = lambda pil_img:shear_x.do_transform(pil_img, m)
  aug_ohl_list.append(mop)
  aug_name_ls.append('ShearX.'+str(m))


def _shear_y_impl(pil_img, level):
  """Applies PIL ShearY to `pil_img`.
  The ShearY operation shears the image along the vertical axis with `level`
  magnitude.
  Args:
    pil_img: Image in PIL object.
    level: Strength of the operation specified as an Integer from
      [0, `PARAMETER_MAX`].
  Returns:
    A PIL Image that has had ShearX applied to it.
  """
  level = float_parameter(level, 0.3)
  if random.random() > 0.5:
    level = -level
  return pil_img.transform((32, 32), Image.AFFINE, (1, 0, 0, level, 1, 0))


shear_y = TransformT('ShearY', _shear_y_impl)
for m in M_list:
  mop = lambda pil_img:shear_y.do_transform(pil_img, m)
  aug_ohl_list.append(mop)
  aug_name_ls.append('ShearY.'+str(m))


def _translate_x_impl(pil_img, level):
  """Applies PIL TranslateX to `pil_img`.
  Translate the image in the horizontal direction by `level`
  number of pixels.
  Args:
    pil_img: Image in PIL object.
    level: Strength of the operation specified as an Integer from
      [0, `PARAMETER_MAX`].
  Returns:
    A PIL Image that has had TranslateX applied to it.
  """
  level = int_parameter(level, 10)
  if random.random() > 0.5:
    level = -level
  return pil_img.transform((32, 32), Image.AFFINE, (1, 0, level, 0, 1, 0))


translate_x = TransformT('TranslateX', _translate_x_impl)
for m in M_list:
  mop = lambda pil_img:translate_x.do_transform(pil_img, m)
  aug_ohl_list.append(mop)
  aug_name_ls.append('TranslateX.'+str(m))


def _translate_y_impl(pil_img, level):
  """Applies PIL TranslateY to `pil_img`.
  Translate the image in the vertical direction by `level`
  number of pixels.
  Args:
    pil_img: Image in PIL object.
    level: Strength of the operation specified as an Integer from
      [0, `PARAMETER_MAX`].
  Returns:
    A PIL Image that has had TranslateY applied to it.
  """
  level = int_parameter(level, 10)
  if random.random() > 0.5:
    level = -level
  return pil_img.transform((32, 32), Image.AFFINE, (1, 0, 0, 0, 1, level))


translate_y = TransformT('TranslateY', _translate_y_impl)
for m in M_list:
  mop = lambda pil_img:translate_y.do_transform(pil_img, m)
  aug_ohl_list.append(mop)
  aug_name_ls.append('TranslateY.'+str(m))


def _crop_impl(pil_img, level, interpolation=Image.BILINEAR):
  """Applies a crop to `pil_img` with the size depending on the `level`."""
  cropped = pil_img.crop((level, level, IMAGE_SIZE - level, IMAGE_SIZE - level))
  resized = cropped.resize((IMAGE_SIZE, IMAGE_SIZE), interpolation)
  return resized


crop_bilinear = TransformT('CropBilinear', _crop_impl)


def _solarize_impl(pil_img, level):
  """Applies PIL Solarize to `pil_img`.
  Translate the image in the vertical direction by `level`
  number of pixels.
  Args:
    pil_img: Image in PIL object.
    level: Strength of the operation specified as an Integer from
      [0, `PARAMETER_MAX`].
  Returns:
    A PIL Image that has had Solarize applied to it.
  """
  level = int_parameter(level, 256)
  return ImageOps.solarize(pil_img.convert('RGB'), 256 - level).convert('RGBA')


solarize = TransformT('Solarize', _solarize_impl)
for m in M_list:
  mop = lambda pil_img:solarize.do_transform(pil_img, m)
  aug_ohl_list.append(mop)
  aug_name_ls.append('Solarize.'+str(m))


def _cutout_pil_impl(pil_img, level):
  """Apply cutout to pil_img at the specified level."""
  size = int_parameter(level, 20)
  if size <= 0:
    return pil_img
  img_height, img_width, num_channels = (32, 32, 3)
  _, upper_coord, lower_coord = (
      create_cutout_mask(img_height, img_width, num_channels, size))
  pixels = pil_img.load()  # create the pixel map
  for i in range(upper_coord[0], lower_coord[0]):  # for every col:
    for j in range(upper_coord[1], lower_coord[1]):  # For every row
      pixels[i, j] = (125, 122, 113, 0)  # set the colour accordingly
  return pil_img

cutout = TransformT('Cutout', _cutout_pil_impl)


def _enhancer_impl(enhancer):
  """Sets level to be between 0.1 and 1.8 for ImageEnhance transforms of PIL."""
  def impl(pil_img, level):
    v = float_parameter(level, 1.8) + .1  # going to 0 just destroys it
    return enhancer(pil_img).enhance(v)
  return impl


color = TransformT('Color', _enhancer_impl(ImageEnhance.Color))

for m in M_list:
  mop = lambda pil_img:color.do_transform(pil_img, m)
  aug_ohl_list.append(mop)
  aug_name_ls.append('Color.'+str(m))

contrast = TransformT('Contrast', _enhancer_impl(ImageEnhance.Contrast))

for m in M_list:
  mop = lambda pil_img:contrast.do_transform(pil_img, m)
  aug_ohl_list.append(mop)
  aug_name_ls.append('Contrast.'+str(m))

brightness = TransformT('Brightness', _enhancer_impl(
    ImageEnhance.Brightness))

for m in M_list:
  mop = lambda pil_img:brightness.do_transform(pil_img, m)
  aug_ohl_list.append(mop)
  aug_name_ls.append('Brightness.'+str(m))

sharpness = TransformT('Sharpness', _enhancer_impl(ImageEnhance.Sharpness))

for m in M_list:
  mop = lambda pil_img:sharpness.do_transform(pil_img, m)
  aug_ohl_list.append(mop)
  aug_name_ls.append('Sharpness.'+str(m))

ALL_TRANSFORMS = [
    identity,
    auto_contrast,
    equalize,
    rotate,
    posterize,
    solarize,
    color,
    contrast,
    brightness,
    sharpness,
    shear_x,
    shear_y,
    translate_x,
    translate_y,
]

random_policy_ops = [
    'Identity', 'AutoContrast', 'Equalize', 'Rotate',
    'Solarize', 'Color', 'Contrast', 'Brightness',
    'Sharpness', 'ShearX', 'TranslateX', 'TranslateY',
    'Posterize', 'ShearY'
]

def augment_list(): 
    l = [
        identity,
        auto_contrast,
        equalize,
        rotate,
        posterize,
        solarize,
        color,
        contrast,
        brightness,
        sharpness,
        shear_x,
        shear_y,
        translate_x,
        translate_y]
    return l

def augment_mag_stage_list(): 
    l = [
        identity,
        auto_contrast,
        equalize,
        rotate,
        posterize,
        solarize,
        color,
        contrast,
        brightness,
        sharpness,
        shear_x,
        shear_y,
        translate_x,
        translate_y]
    amsl = []
    for m in M_list:
      tmp = []
      for op in l:
        tmp.append(lambda pil_img:op.do_transform(pil_img, m))
      amsl.append(tmp)
    return amsl
class Curriculum_Aug:
    def __init__(self, n, th):
        self.n = n
        self.aug_ohl_list = augment_mag_stage_list()
        self.sl = len(self.aug_ohl_list[0])
        self.th=th
        self.stage = 1

    def __call__(self, img):
      ss = np.random.choice(self.stage, self.n)
      ids = np.random.choice(self.sl, self.n)
      # print(idxs)
      if random.random()<self.th:
        for s, idx in zip(ss,ids):
          img = self.aug_ohl_list[s][idx](img)
          img = img.convert('RGB')
      return img
class Lighting(object):
    """Lighting noise(AlexNet - style PCA - based noise)"""

    def __init__(self, alphastd, eigval, eigvec):
        self.alphastd = alphastd
        self.eigval = torch.Tensor(eigval)
        self.eigvec = torch.Tensor(eigvec)

    def __call__(self, img):
        if self.alphastd == 0:
            return img

        alpha = img.new().resize_(3).normal_(0, self.alphastd)
        rgb = self.eigvec.type_as(img).clone() \
            .mul(alpha.view(1, 3).expand(3, 3)) \
            .mul(self.eigval.view(1, 3).expand(3, 3)) \
            .sum(1).squeeze()

        return img.add(rgb.view(3, 1, 1).expand_as(img))


class CutoutDefault(object):
    """
    Reference : https://github.com/quark0/darts/blob/master/cnn/utils.py
    """
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


class RandAugment:
    def __init__(self, n, m):
        self.n = n
        self.m = m      # [0, 30]
        self.augment_list = augment_list()

    def __call__(self, img):
        ops = random.choices(self.augment_list, k=self.n)
        for op in ops:
            img = op.do_transform(img, self.m)
        img = img.convert('RGB')
        return img
class RandAugment_th:
    def __init__(self, n, m):
        self.n = n
        self.m = m      # [0, 30]
        self.augment_list = augment_list()
        self.th = 1

    def __call__(self, img):
        ops = random.choices(self.augment_list, k=self.n)
        if random.random()<self.th:
          for op in ops:
              img = op.do_transform(img, self.m)
          img = img.convert('RGB')
        return img
def augment_list_G():  
    l = [
        identity,
        rotate,
        shear_x,
        shear_y,
        translate_x,
        translate_y]
    return l
class RandAugment_G:
    def __init__(self, n, m):
        self.n = n
        self.m = m      # [0, 30]
        self.augment_list = augment_list_G()

    def __call__(self, img):
        ops = random.choices(self.augment_list, k=self.n)
        for op in ops:
            #val = (float(self.m) / 30) * float(maxval - minval) + minval
            img = op.do_transform(img, self.m)
        img = img.convert('RGB')
        return img
def augment_list_C():  
    l = [
        identity,
        auto_contrast,
        equalize,
        posterize,
        solarize,
        color,
        contrast,
        brightness,
        sharpness,
        ]

    return l
class RandAugment_C:
    def __init__(self, n, m):
        self.n = n
        self.m = m      # [0, 30]
        self.augment_list = augment_list_C()

    def __call__(self, img):
        ops = random.choices(self.augment_list, k=self.n)
        for op in ops:
            #val = (float(self.m) / 30) * float(maxval - minval) + minval
            img = op.do_transform(img, self.m)
        img = img.convert('RGB')
        return img
#For Augmentation search
class RWAug_Search:
    def __init__(self, n, idxs):
        self.n = n
        #idxs is the operation id
        self.idxs = idxs      
        self.aug_ohl_list = aug_ohl_list

    def __call__(self, img):
      assert len(self.idxs) == self.n
      #print(self.idxs)
      for idx in self.idxs:
        img = aug_ohl_list[idx](img)
        img = img.convert('RGB')
      return img
#For Augmentation policy apply
class RWAug_Train:
    def __init__(self, n, p = None):
        self.n = n
        if p == None:
          p = [1 for i in range(len(aug_ohl_list))]
        self.p = np.array(p)/sum(p)      # [0, 30]
        self.aug_ohl_list = aug_ohl_list
        self.th=1

    def __call__(self, img):
      assert len(self.p) == len(self.aug_ohl_list)
      idxs = np.random.choice(range(len(self.aug_ohl_list)),size = 2,p = self.p)
      # print(idxs)
      if random.random()<self.th:
        for idx in idxs:
          img = aug_ohl_list[idx](img)
          img = img.convert('RGB')
      return img
# May 6th add for baseline "Total Random!"
class RandAugment_ohl:
    def __init__(self, n):
        self.n = n
        self.augment_list = aug_ohl_list
        self.th=1

    def __call__(self, img):
        ops = random.choices(self.augment_list, k=self.n)
        if random.random()<self.th:
          for op in ops:
              #val = (float(self.m) / 30) * float(maxval - minval) + minval
              img = op(img)
          img = img.convert('RGB')
        return img
M_list_cifar = [2, 6, 10, 14]
M_list_imagenet = [5, 13, 20, 28]
aug_color_cifar = []
aug_color_imagenet = []
aug_geo_cifar = []
aug_geo_imagenet = []
aug_all_cifar = []
aug_all_imagenet = []
aug_list_C = augment_list_C()
for op in aug_list_C:
  for m in  M_list_cifar:
    mop = lambda pil_img:op.do_transform(pil_img, m)
    aug_color_cifar.append(mop)
    aug_all_cifar.append(mop)
  for m in  M_list_imagenet:
    mop = lambda pil_img:op.do_transform(pil_img, m)
    aug_color_imagenet.append(mop)
    aug_all_imagenet.append(mop)
aug_list_G = augment_list_G()
for op in aug_list_G:
  for m in  M_list_cifar:
    mop = lambda pil_img:op.do_transform(pil_img, m)
    aug_geo_cifar.append(mop)
    aug_all_cifar.append(mop)
  for m in  M_list_imagenet:
    mop = lambda pil_img:op.do_transform(pil_img, m)
    aug_geo_imagenet.append(mop)
    aug_all_imagenet.append(mop)
aug_dict={}
aug_dict['cifarall'] = aug_all_cifar
aug_dict['cifarcolor'] = aug_color_cifar
aug_dict['cifargeo'] = aug_geo_cifar
aug_dict['imagenetall'] = aug_all_imagenet
aug_dict['imagenetcolor'] = aug_color_imagenet
aug_dict['imagenetgeo'] = aug_geo_imagenet
class RandOhl:
    def __init__(self, n, dataset = 'cifar',mode = 'all'):
        self.n = n
        self.augment_list = aug_dict[dataset+mode]
        self.th=1

    def __call__(self, img):
        ops = random.choices(self.augment_list, k=self.n)
        if random.random()<self.th:
          for op in ops:
              #val = (float(self.m) / 30) * float(maxval - minval) + minval
              img = op(img)
          img = img.convert('RGB')
        return img
if __name__ == '__main__':
  print(aug_name_ls)
  print(len(aug_name_ls))
