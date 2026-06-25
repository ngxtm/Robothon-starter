import unittest
import os
import tempfile
import sys

# Add scripts dir to path for importing
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'scripts'))

from config_manager import load_config, check_dependencies

class TestConfigManager(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.TemporaryDirectory()
        
    def tearDown(self):
        self.test_dir.cleanup()

    def test_load_config_valid(self):
        config_path = os.path.join(self.test_dir.name, 'config.json')
        with open(config_path, 'w') as f:
            f.write('{"test": "value"}')
            
        config = load_config(config_path)
        self.assertEqual(config, {"test": "value"})

    def test_load_config_not_found(self):
        with self.assertRaises(FileNotFoundError):
            load_config("nonexistent.json")

    def test_check_dependencies_valid(self):
        req_path = os.path.join(self.test_dir.name, 'reqs.txt')
        with open(req_path, 'w') as f:
            f.write('json\nos\n') # Built-ins should always pass
            
        self.assertTrue(check_dependencies(req_path))

if __name__ == '__main__':
    unittest.main()
