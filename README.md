# 🎳 Bowling Segmentation Stat Manager (BSSM) 🎳

This is the repository for BSSM.

## 💻 Getting Started

First, clone this repository and navigate to the project folder:

```bash
git clone https://github.com/gwmic/bssm
cd bssm
```

Next, set up a Python environment and install the required project dependencies:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Next, set up a Roboflow Inference Docker container. This Docker container will manage inference for the gaze detection system. [Learn how to set up an Inference Docker container](https://inference.roboflow.com/quickstart/docker/).

## 🔑 keys

Before running the inference script, ensure that the `API_KEY` is set as an environment variable. This key provides access to the inference API.

- For Unix/Linux:

    ```bash
    export API_KEY=your_api_key_here
    ```

- For Windows:

    ```bash
    set API_KEY=your_api_key_here
    ```
  
Replace `your_api_key_here` with your Roboflow API key. [Learn how to retrieve your Roboflow API key](https://docs.roboflow.com/api-reference/authentication#retrieve-an-api-key).

## 🎬 Run BSSM

To use bssm run the following command:

```bash
python3 bssm.py
```

After five strike shot's have been recorded an oil ditribution map will be generated as `oildistribution.pkl`. Run the following command to view the map:

```bash
python3 viewoil.py oildistribution.pkl
```