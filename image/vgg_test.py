import numpy as np
import tensorflow as tf

from image import vgg


class VGGTest(tf.test.TestCase):
    def test_vgg16(self):
        model = vgg.vgg16(224, 224, 1000)
        inp_tensor = tf.random.uniform(shape=(4, 3, 224, 224), dtype=tf.float32)
        out = model(inp_tensor)
        self.assertShapeEqual(np.random.uniform(0, 1, (4, 1000)), out)
        self.assertEqual(model.count_params(), 138357544)

    def test_vgg19(self):
        model = vgg.vgg19(224, 224, 1000)
        self.assertEqual(model.count_params(), 143667240)

    def test_vgg13(self):
        model = vgg.vgg13(224, 224, 1000)
        self.assertEqual(model.count_params(), 133047848)

    def test_vgg11(self):
        model = vgg.vgg11(224, 224, 1000)
        self.assertEqual(model.count_params(), 132863336)


if __name__ == '__main__':
    tf.test.main()
