import unittest
import os

from DataSets.HotailorPOC2.download_HotailorPOC2_dataset import download_dataset
from PretrainedModels.models_util import download_model, download_model_by_name


class TestDownload(unittest.TestCase):

    def test_data_download(self):
        print ("Testing dataset HotailorPOC2 download function:")

        download_dataset()

        # check if downloaded dataset exists at the right directory
        # dataset_folder = os.path.dirname("DataSets/HotailorPOC/")
        # if not (os.path.exists(os.path.join(dataset_folder, "positive"))
        #         and os.path.exists(os.path.join(dataset_folder, "positive"))
        #         and os.path.exists(os.path.join(dataset_folder, "positive"))):


    def test_model_download_by_name(self):
        print ("Testing model download function:")

        check_path = "PretrainedModels/"
        filename = "AlexNet.model"
        download_model_by_name(filename)
        self.assertTrue(os.path.exists(check_path + filename))

    def test_model_download(self):
        modelNameToUrl = {
        'AlexNet.model':   'https://www.cntk.ai/Models/AlexNet/AlexNet.model',
        'AlexNetBS.model': 'https://www.cntk.ai/Models/AlexNet/AlexNetBS.model',
        'ResNet_18.model': 'https://www.cntk.ai/Models/ResNet/ResNet_18.model',
        'Fast-RCNN_grocery100.model' : 'https://www.cntk.ai/Models/FRCN_Grocery/Fast-RCNN_grocery100.model',
        'VGG16.model':'https://www.cntk.ai/Models/Caffe_Converted/VGG16_ImageNet_Caffe.model',
        'VGG19.model':'https://www.cntk.ai/Models/Caffe_Converted/VGG19_ImageNet_Caffe.model'
        }

        check_path = "PretrainedModels/"
        filename = "AlexNet.model"
        url = modelNameToUrl[filename]
        download_model(filename, url)
        self.assertTrue(os.path.exists(check_path + filename))
        
if __name__ == "__main__":
    unittest.main()
