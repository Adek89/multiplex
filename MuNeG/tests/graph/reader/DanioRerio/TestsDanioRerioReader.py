__author__ = 'Adrian'
import unittest
from graph.reader.DanioRerio.DanioRerioReader import DanioRerioReader
class TestsDanioRerioReader(unittest.TestCase):


    def test_read(self):
        reader = DanioRerioReader()
        reader.read()

