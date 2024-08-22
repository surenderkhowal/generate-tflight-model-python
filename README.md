# Navigate to your project directory or create new directory
cd path/to/your/project


# install tensorflow sdk

pip install tensorflow


# Create a virtual environment (you can name it 'venv')
python -m venv venv

# create the virtual environment using Python
python3 -m venv venv

# Activate the virtual environment
# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate


# Once your virtual environment is activated and the required packages are installed, you can run the Python script. Use the following command:
python train_model.py

# If your script requires specific arguments, you can pass them like this:
python tuned_model.py arg1 arg2


# If you face an issue related to **protobuf**  make sure you'r using correct version of python
pip3 install protobuf
or
pip install protobuf


# ImportError: Could not import PIL.Image. The use of load_img requires PIL. then install pillow lib also

pip install Pillow

# After the installation, you can verify that Pillow is installed correctly by running the following command:
pip show Pillow



# images dataset structure like this 

project_name/
└── dataset/
    ├── bottles/
    │   ├── bottle-1.png
    │   ├── bottle-2.png
    │   ├── bottle-3.png
    ├── utensils/
    │   ├── utensil-1.png
    │   ├── utensil-2.png
        .
        .
        .

# Subdirectory for Classes: Each class of images (e.g., bottles, utensils) should have its own subdirectory within the dataset directory.

# Make sure the directory sequence match with your app code where you'r using the tflite model after generating 

       // Ensure the order of labels matches the order of classes during training
        val labels = arrayOf(
            "Bottle",
            "Utensils",
            .
            .
            .
        )

# Once you create model successfully you can use this model in android app
# run android androidExample->TensorflowTest
# add your model in  generate-tflight-model-python/androidExample/TensorFlowTest/app/src/main/assets
# make sure your model name match with which used in PlasticClassifier.kt 