"""dataloader_test.py

Unittest for dataloader

"""
import glob
import unittest
import numpy as np
from dataloader import *

TESTFILE = {"class1":["/mnt/dataset/experiment/cat_dog/train/training/cat.1.jpg",
                      "/mnt/dataset/experiment/cat_dog/train/training/cat.2.jpg",
                      "/mnt/dataset/experiment/cat_dog/train/training/cat.3.jpg",
                      "/mnt/dataset/experiment/cat_dog/train/training/cat.4.jpg",
                      "/mnt/dataset/experiment/cat_dog/train/training/cat.5.jpg",
                      "/mnt/dataset/experiment/cat_dog/train/training/cat.6.jpg",
                      "/mnt/dataset/experiment/cat_dog/train/training/cat.7.jpg",
                      "/mnt/dataset/experiment/cat_dog/train/training/cat.8.jpg",
                      "/mnt/dataset/experiment/cat_dog/train/training/cat.9.jpg",],
            "class2":["/mnt/dataset/experiment/cat_dog/train/training/dog.1.jpg",
                      "/mnt/dataset/experiment/cat_dog/train/training/dog.2.jpg",
                      "/mnt/dataset/experiment/cat_dog/train/training/dog.3.jpg",
                      "/mnt/dataset/experiment/cat_dog/train/training/dog.4.jpg",
                      "/mnt/dataset/experiment/cat_dog/train/training/dog.5.jpg",
                      "/mnt/dataset/experiment/cat_dog/train/training/dog.6.jpg",
                      "/mnt/dataset/experiment/cat_dog/train/training/dog.7.jpg",
                      "/mnt/dataset/experiment/cat_dog/train/training/dog.8.jpg",
                      "/mnt/dataset/experiment/cat_dog/train/training/dog.9.jpg",]
            }
TEST_CLASSMAP = {"class1": 0,
                 "class2": 1}

class GetDatasetTest(unittest.TestCase):
    def setUp(self):
        self.dataset = GetDataset(datapath_map=TESTFILE,
                                  classid_map=TEST_CLASSMAP,
                                  preproc_fn=None,
                                  augment_fn=None,
                                  image_size=(256,256))

    def test_lenth_of_dataset(self):
        dataset_length = len(self.dataset)

        self.assertEqual(dataset_length, len(TESTFILE['class1'])+len(TESTFILE['class2']) )

    def test_getitem(self):
        img, target = self.dataset[0]

        self.assertIsInstance(img, np.ndarray)
        self.assertIsInstance(target, np.ndarray)
        self.assertEqual(img.shape, (256, 256, 3))
        self.assertEqual(target.shape, (len(TEST_CLASSMAP),))
        print(target)

    def test_iteration(self):
        img, target = next(self.dataset)

        self.assertIsInstance(img, np.ndarray)
        self.assertIsInstance(target, np.ndarray)
        self.assertEqual(img.shape, (256, 256, 3))
        self.assertEqual(target.shape, (len(TEST_CLASSMAP),))

    def tearDown(self):
        pass


class DataLoaderTest(unittest.TestCase):
    def setUp(self):
        self.batch_size = 32
        self.image_size = (256, 256, 3)
        dataset = GetDataset(datapath_map=TESTFILE,
                             classid_map=TEST_CLASSMAP,
                             preproc_fn=None,
                             augment_fn=None,
                             image_size=self.image_size)
        self.dataloader = DataLoader(dataset, num_classes=len(TEST_CLASSMAP), batch_size=self.batch_size)

    def test_getitem(self):
        imgs, targets = self.dataloader[0]

        self.assertIsInstance(imgs, np.ndarray)
        self.assertEqual(len(imgs), self.batch_size)
        self.assertEqual(len(targets), self.batch_size)
        self.assertEqual(imgs[0].shape, self.image_size)

    def tearDown(self):
        pass

if __name__ == "__main__":
    unittest.main()