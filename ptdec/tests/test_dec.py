import torch
from torch.autograd import Variable
from unittest import TestCase

from ptdec.ae import StackedAutoEncoder
from ptdec.dec import DEC


class TestAutoEncoder(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ae = StackedAutoEncoder([100, 50, 5])
        cls.dec = DEC(2, 100, 5, cls.ae)

    def test_dimension(self):
        """
        Basic tests that check that given an input tensor the output and encoded tensors are of the
        expected size.
        """
        input_tensor = Variable(torch.Tensor(1, 100).fill_(1.0))
        output_tensor = self.dec(input_tensor)
        self.assertEquals(
            tuple(output_tensor.size()),
            (1, 2)
        )
