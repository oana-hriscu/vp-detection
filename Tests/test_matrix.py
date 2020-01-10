from unittest import TestCase

from Kalman_Filter import Matrix


class TestMatrix(TestCase):

    def setUp(self):
        self.matrix = Matrix([[10000., 0.], [0., 10000.]])

    def test_zero(self):
        self.zero = self.matrix.zero(5, 10)
        self.assertIsNone(self.zero)

    def test_identity(self):
        self.identity = self.matrix.identity(500)
        self.assertIsNone(self.identity)

    def test_show(self):
        self.show = self.matrix.show()
        self.assertIsNone(self.show)

    def test_transpose(self):
        self.transpose = self.matrix.transpose()
        self.assertEquals(self.transpose, [[10000.0, 0.0], [0.0, 10000.0]])
        self.assertIsNot(self.transpose, [0.0, 10000.0])

    def test_Cholesky(self):
        self.cholesky = self.matrix.Cholesky()
        self.assertEquals(self.cholesky, [[100.0, 0.0], [0, 100.0]])
    #
    # def test_CholeskyInverse(self):
    #     self.fail()
    #
    # def test_inverse(self):
    #     self.fail()
