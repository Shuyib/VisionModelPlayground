# Multimodel OCR for Insurance Documents
As DigiNsure Inc. embarks on a mission to digitize historical insurance claim documents, we are excited to introduce our latest project: a multi-modal Optical Character Recognition (OCR) model. This model is designed to enhance the efficiency of processing claims and customer service interactions by accurately identifying primary and secondary IDs from scanned documents.

This is a datacamp project that can be found here: [Datacamp Project](https://app.datacamp.com/learn/projects/2215).

## Installation
Start by making a virtual environment and installing the required packages. You can do this using pip:

```bash
python3 -m venv env
source env/bin/activate  # On Windows use `env\Scripts\activate`
pip install -r requirements.txt
```

You can now open the jupyter notebook and start working on the project.

```bash
jupyter notebook
```

## Usage
We use a multi-modal approach to train an OCR model that takes images of scanned documents and their corresponding insurance types as input. The model is trained to classify the documents into two categories: primary and secondary IDs. We use a residual block architecture to build the model, which allows for better feature extraction and classification. Another simple model is also provided for comparison.

Accuracy for the residual block model is around 90% on the test set, while the simple model achieves over 95% accuracy. The residual block model is more complex and requires more computational resources, but needs more pruning to achieve the same accuracy as the simple model.

## Contributing
We welcome contributions to this project! If you have suggestions for improvements or new features, please open an issue or submit a pull request. Please make sure to follow the code style and testing guidelines.


