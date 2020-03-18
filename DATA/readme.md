# A Quick Note about this DATA directory

Since this is a project trying two different methods - segmentation vs. detection, this DATA directory is used as a shared data corpus labelled manually with two kinds of labeling.
- **Segmentation**
  - input: `imgs/*.jpg`
  - label: `masks/*.jpg`

- **Detection**
  - input: `imgs/*.jpg`
  - label: bbox represented as left_top and right_bottom corner coordinates in `annotations/*.xml`

**Attention PLS**:
- NOT all `imgs/*.jpg` has a corresponding `masks/*.jpg` because there probably does not exist target veins or we cannot manually recognize any veins in it.
- The `annotations/*.xml` contains not only detection bbox info but also segmentation region info and the segmentation info in xml file is not used for UNet training.

Therefore, we need to do a bit correct preprocessing before using them for training in each method:
- **Segmentation**: extract only images who do have their corresponding masks.
- **Detection**: extract only images who do have their corresponding annotation xmls.
- Note: The amount of images for two methods should be same.