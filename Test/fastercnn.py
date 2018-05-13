import unittest
import os, sys, gc

abs_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(abs_path, ".."))
sys.path.append(os.path.join(abs_path, "../Detection/FasterRCNN"))

from FasterRCNN import set_global_vars, load_model, CloneMethod

model_path = os.path.join(abs_path, "../PretrainedModels/AlexNet.model")

from FasterRCNN import clone_model, clone_conv_layers
model = load_model(model_path)
method = CloneMethod.freeze
begin = ["features"]
end = ["conv5.y"]

eval = None

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


class Train(unittest.TestCase):

    def test_e2e(self):
        from FasterRCNN import train_faster_rcnn_e2e

        gc.collect()
        
        eval = train_faster_rcnn_e2e(model_path, True)

    def test_alternating(self):
        from FasterRCNN import train_faster_rcnn_alternating

        gc.collect()
        
        eval = train_faster_rcnn_alternating(model_path, True)

class Evaluation(unittest.TestCase):

    def test_evaluate(self):
        from FasterRCNN import train_faster_rcnn_e2e, eval_faster_rcnn_mAP

        gc.collect()
        
        eval = train_faster_rcnn_e2e(model_path, True)
        eval_faster_rcnn_mAP(eval)
        
if __name__ == "__main__":
    unittest.main()
