# A Quick Note about this DATA directory

Since this is a project trying two different methods - segmentation vs. detection, this DATA directory is used as a shared data corpus labelled manually with two kinds of labeling.
- **Segmentation**
  - input: `imgs/*.jpg`
  - label: `masks/*.jpg`

- **Detection**
  - input: `imgs/*.jpg`
  - label: bbox represented as left_top and right_bottom corner coordinates in `annotations/*.xml`

**Attention**:
- NOT all `imgs/*.jpg` has a corresponding `masks/*.jpg` because there does not exist target veins in the image or we cannot manually recognize any veins in it.
- The `annotations/*.xml` contains not only detection bbox info but also segmentation region info, so we need to process them correctly into YOLO project. (The segmentation info in xml file is not used for UNet training).