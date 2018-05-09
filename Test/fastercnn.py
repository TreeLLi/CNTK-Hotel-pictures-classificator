import unittest
import os, sys

abs_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(abs_path, ".."))
sys.path.append(os.path.join(abs_path, "../Detection/FasterRCNN"))

from FasterRCNN import set_global_vars, load_model, CloneMethod

model_path = os.path.join(abs_path, "../PretrainedModels/VGG16.model")

from FasterRCNN import clone_model, clone_conv_layers
model = load_model(model_path)
method = CloneMethod.freeze
begin = ["data"]
end = ["relu5_3"]

set_global_vars()
    
class Clone(unittest.TestCase):
    
    def test_model(self):
        print ("clone model")
        cloned = clone_model(model, begin, end, method)
        self.assertTrue(cloned)
        
    def test_conv_layers(self):
        print ("convolutional layers")
        cloned = clone_conv_layers(model)
        self.assertTrue(cloned)

if __name__ == "__main__":
    unittest.main()
