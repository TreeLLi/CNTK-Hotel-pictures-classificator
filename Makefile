make:
	python Detection/FasterRCNN/FasterRCNN.py

data:
	python Detection/FasterRCNN/install_data_and_model.py

train:
	rm -f Detection/FasterRCNN/Output/*.model
	make

clean:
	rm -f -r Detection/FasterRCNN/Output Detection/FasterRCNN/__pycache__
