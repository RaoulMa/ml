import unittest
from fibonacci import fibonacci

class test(unittest.TestCase):
    def test_result(self):
        self.assertEqual(fibonacci(0), 1, 'edge n=0 case incorrect')
        self.assertEqual(fibonacci(1), 1, 'edge n=1 case incorrect')
        self.assertEqual(fibonacci(2), 2, 'edge n=2 case incorrect')
        self.assertEqual(fibonacci(3), 3, 'edge n=3 case incorrect')

def __main__():
    unittest.main()