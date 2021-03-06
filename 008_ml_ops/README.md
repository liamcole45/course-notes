# Notes from Lectures, Labs and Readings

# Working with Cloud Build
#### Linux/bash Commands
- `cat` displays file in terminal 
- `nano` editor for file

## Building Containers with DockerFile and Cloud Build

1. Activate Cloud Shell.
2. Create an empty `quickstart.sh` file using the nano text editor.
```
nano quickstart.sh
```
3. Add the following lines in to the quickstart.sh file:
```
#!/bin/sh
echo "Hello, world! The time is $(date)."
```
Save the file and close nano
4. Create an empty Dockerfile file using the nano text editor.
```
nano Dockerfile
```
add
```
FROM alpine
COPY quickstart.sh /
CMD ["/quickstart.sh"]
```
5. In Cloud Shell, run the following command to make the quickstart.sh script executable.
```
chmod +x quickstart.sh
```
6. In Cloud Shell, run the following command to build the Docker container image in Cloud Build.
```
gcloud builds submit --tag gcr.io/${GOOGLE_CLOUD_PROJECT}/quickstart-image .
```
Important:
- Don't miss the dot (".") at the end of the command. The dot specifies that the source code is in the current working directory at build time
7. In the Google Cloud Console, on the Navigation menu (Navigation menu), click Container Registry > Images.

## Building Containers with a build configuration file and Cloud Build
1. Create a soft link as a shortcut to the working [directory](https://github.com/GoogleCloudPlatform/training-data-analyst/tree/master/courses/ak8s/v1.1/Cloud_Build/a)
```
ln -s ~/training-data-analyst/courses/ak8s/v1.1 ~/ak8s
```
2. Change to the directory that contains the sample files for this lab.
```
cd ~/ak8s/Cloud_Build/a
```
3. In Cloud Shell, execute the following command to start a Cloud Build using cloudbuild.yaml as the build configuration file:
```
gcloud builds submit --config cloudbuild.yaml .
```

### Compute Solution Comparisons
<img src="./pictures/compute_solution_comparisons.png" alt="drawing" width="600"/>

### Compute Solution Summary
<img src="./pictures/compute_solutions_summary.png" alt="drawing" width="600"/>

### Kubernetes Control Plane
<img src="./pictures/kubernetes_control_plane.png" alt="drawing" width="600"/>

### GKE Zonal vs Regional Cluster
<img src="./pictures/zonal_vs_regional_cluster.png" alt="drawing" width="600"/>

### GKE Private Cluster
<img src="./pictures/private_cluster.png" alt="drawing" width="400"/>

### Ways to create deployments
Describe Deployment Command/Linux

<img src="./pictures/describe_deployment.png" alt="drawing" width="400"/>

Deployment Config Output Command/Linux

<img src="./pictures/output_deployment_config.png" alt="drawing" width="400"/>

Autoscaling

<img src="./pictures/Autoscaling.png" alt="drawing" width="400"/>


### AI Platform Pipelines

Tech stack

<img src="./pictures/ai_platforms_tech_stack.png" alt="drawing" width="400"/>

Implementation Strategy

<img src="./pictures/ai_platforms_implementation_strategy.png" alt="drawing" width="400"/>

Instance

<img src="./pictures/ai_platforms_instance.png" alt="drawing" width="400"/>

Power of Reusable Pipelines

<img src="./pictures/power_of_reusable_pipelines.png" alt="drawing" width="400"/>

System Overview

<img src="./pictures/system_overview.png" alt="drawing" width="400"/>

## Using custom containers with AI Platform Training
### Enable Cloud Services
1. In Cloud Shell, to set the project ID to your Google Cloud Project, run the following command:
```
export PROJECT_ID=$(gcloud config get-value core/project)
gcloud config set project $PROJECT_ID
```
2. To enable the required Cloud services, run the following commands:
```
gcloud services enable \
cloudbuild.googleapis.com \
container.googleapis.com \
cloudresourcemanager.googleapis.com \
iam.googleapis.com \
containerregistry.googleapis.com \
containeranalysis.googleapis.com \
ml.googleapis.com \
dataflow.googleapis.com
```
3. Add the Editor permission for your Cloud Build service account:
```
PROJECT_NUMBER=$(gcloud projects describe $PROJECT_ID --format="value(projectNumber)")
CLOUD_BUILD_SERVICE_ACCOUNT="${PROJECT_NUMBER}@cloudbuild.gserviceaccount.com"
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member serviceAccount:$CLOUD_BUILD_SERVICE_ACCOUNT \
  --role roles/editor
```
### Create an instance of AI Platform Pipelines
1. Go to [Pipelines](https://console.cloud.google.com/marketplace/kubernetes/config/google-cloud-ai-platform/kubeflow-pipelines) and create a GKE Cluster if there is not already one there
2. Then click `Deploy` 

### Create an instance of Vertex AI Platform Notebooks
An instance of Vertex AI Platform Notebooks is used as a primary experimentation/development workbench. The instance is configured using a custom container image that includes all Python packages required for this lab.
1. In Cloud Shell, create a folder in your home directory:
```
cd
mkdir tmp-workspace
cd tmp-workspace
```
2. Create a requirements file with the Python packages to install in the custom image:
```
gsutil cp gs://cloud-training/OCBL203/requirements.txt .
```
3. Create a Dockerfile that defines your custom container image:
```
gsutil cp gs://cloud-training/OCBL203/Dockerfile .  
```
4. Build the image and push it to your project's Container Registry:
```
IMAGE_NAME=kfp-dev
TAG=latest
IMAGE_URI="gcr.io/${PROJECT_ID}/${IMAGE_NAME}:${TAG}"
gcloud builds submit --timeout 15m --tag ${IMAGE_URI} .
```

5. Create an instance of Vertex AI Platform Notebooks:
```
ZONE=us-central1-a
INSTANCE_NAME=ai-notebook
```
If you want to use a different ZONE and INSTANCE_NAME, replace us-central1-a with the zone of your choice as [YOUR_ZONE] and replace ai-notebook with the instance name of your choice as [YOUR_INSTANCE_NAME].
```
IMAGE_FAMILY="common-container"
IMAGE_PROJECT="deeplearning-platform-release"
INSTANCE_TYPE="n1-standard-4"
METADATA="proxy-mode=service_account,container=$IMAGE_URI"
gcloud compute instances create $INSTANCE_NAME \
  --zone=$ZONE \
  --image-family=$IMAGE_FAMILY \
  --machine-type=$INSTANCE_TYPE \
  --image-project=$IMAGE_PROJECT \
  --maintenance-policy=TERMINATE \
  --boot-disk-device-name=${INSTANCE_NAME}-disk \
  --boot-disk-size=100GB \
  --boot-disk-type=pd-ssd \
  --scopes=cloud-platform,userinfo-email \
  --metadata=$METADATA
```

This may take up to 5 minutes to complete.

After five minutes, in the Cloud Console, on the Navigation menu, click Vertex AI > Notebooks. The notebook instance you created in the previous step will be listed.

Click the Open Jupyterlab link.

### Clone the mlops-on-gcp repo within your Vertex AI Platform Notebooks instance
```
cd home/jupyter
```

```
git clone https://github.com/GoogleCloudPlatform/mlops-on-gcp
```


### Notebook Location
[lab-01.ipynb](./labs/lab-01.ipynb)
Downloaded from [here](https://github.com/GoogleCloudPlatform/mlops-on-gcp/blob/master/on_demand/kfp-caip-sklearn/lab-01-caip-containers/exercises/lab-01.ipynb)

- **In this lab you learned how to develop a training application, package it as a Docker image, and run it on AI Platform Training.**
- **Really good detail, please refer to Notebook Location to see notes**


Using custom containers with AI Platform Training
Learning Objectives:

- Learn how to create a train and a validation split with BigQuery
- Learn how to wrap a machine learning model into a Docker container and train in on AI Platform
- Learn how to use the hyperparameter tunning engine on Google Cloud to find the best - hyperparameters
- Learn how to deploy a trained machine learning model Google Cloud as a rest API and query it
- In this lab, you develop a multi-class classification model, package the model as a docker image, and run on AI Platform Training as a training application. The training application trains a multi-class classification model that predicts the type of forest cover from cartographic data. The dataset used in the lab is based on Covertype Data Set from UCI Machine Learning Repository.

Scikit-learn is one of the most useful libraries for machine learning in Python. The training code uses scikit-learn for data pre-processing and modeling.

The code is instrumented using the `hypertune` package so it can be used with AI Platform hyperparameter tuning job in searching for the best combination of hyperparameter values by optimizing the metrics you specified.

## Kubeflow Components
Lecture diagrams

### ML Devops
<img src="./pictures/ml_devops.png" alt="drawing" width="600"/>

### Kubeflow Dag
<img src="./pictures/kfp_dag.png" alt="drawing" width="600"/>

### Defining Dag
<img src="./pictures/defining_dag.png" alt="drawing" width="600"/>

## 1st Main Type of Kubeflow Pipeline - Prebuilt Components

`component.yaml`

<img src="./pictures/component_yaml.png" alt="drawing" width="600"/>

Loading Prebuild Components
<img src="./pictures/loading_prebuild_components.png" alt="drawing" width="600"/>

Component `bigquery/query`
<img src="./pictures/component_bigquery_query.png" alt="drawing" width="600"/>

Component `ml_engine/train`
<img src="./pictures/component_ml_engine_train.png" alt="drawing" width="600"/>

Component `ml_engine/deploy`
<img src="./pictures/component_ml_engine_deploy.png" alt="drawing" width="600"/>

Component `hypertuning`
<img src="./pictures/component_hypertuning.png" alt="drawing" width="600"/>

## 2nd Main Type of Kubeflow Pipeline - Lightweight Python Components
Wraping Python in Kubeflow
<img src="./pictures/wraping_python_in_kf.png" alt="drawing" width="600"/>

## 3rd Main Type of Kubeflow Pipeline - Custom Components
Step 1 - `main.py`

<img src="./pictures/1_main_py.png" alt="drawing" width="600"/>

Step 2 - Package into docker
<img src="./pictures/2_package_into_docker.png" alt="drawing" width="600"/>

Step 3 - Component description
<img src="./pictures/3_component_description.png" alt="drawing" width="600"/>

Step 4 - Load component into pipeline
<img src="./pictures/4_load_component_into_pipeline.png" alt="drawing" width="600"/>

Step 5 - Run pipeline components
<img src="./pictures/5_run_pipeline_components.png" alt="drawing" width="600"/>

## Compile, upload and run

Step 1 - Build and push base container
<img src="./pictures/1_build_push_base_container.png" alt="drawing" width="600"/>

Step 2 - Build and push trainer container
<img src="./pictures/2_build_push_trainer_container.png" alt="drawing" width="600"/>

Step 3 - Compile Kubeflow
<img src="./pictures/3_compile_kubeflow.png" alt="drawing" width="600"/>

Step 4 - Upload pipeline to Kubeflow cluster
<img src="./pictures/4_upload_pipeline_to_kf_cluster.png" alt="drawing" width="600"/>

Step 5 - Run pipeline
<img src="./pictures/5_run_pipeline.png" alt="drawing" width="600"/>



## Continuous Training Pipeline with Kubeflow Pipeline and Cloud AI Platform

**Learning Objectives:**
1. Learn how to use Kubeflow Pipeline(KFP) pre-build components (BiqQuery, AI Platform training and predictions)
1. Learn how to use KFP lightweight python components
1. Learn how to build a KFP with these components
1. Learn how to compile, upload, and run a KFP with the command line


In this lab, you will build, deploy, and run a KFP pipeline that orchestrates **BigQuery** and **AI Platform** services to train, tune, and deploy a **scikit-learn** model.

Cloning repo in notebook (after setting up Kubeflow pipeline)
```
cd home/jupyter/
git clone https://github.com/GoogleCloudPlatform/mlops-on-gcp
```

[lab-02.ipynb](./labs/lab-02.ipynb)
Downloaded from [here](https://github.com/GoogleCloudPlatform/mlops-on-gcp/blob/72a48faf875f06abe667e3e3bb2eafacc60594c5/on_demand/kfp-caip-sklearn/lab-02-kfp-pipeline/lab-02.ipynb)

In this lab, you learned how to build, deploy, and run a KFP that orchestrates BigQuery and AI Platform services to train, tune, and deploy a Scikit-learn model.


## Full CI/CD Stack for ML Systems
<img src="./pictures/full_ci_cd_stack_for_ml_systems.png" alt="drawing" width="600"/>

## Prepackaged configuration actions that are available to you (see Github URL)
<img src="./pictures/cloud_builder_examples.png" alt="drawing" width="600"/>

# Cloud Builder
### Example
<img src="./pictures/cloud_builder_example.png" alt="drawing" width="600"/>

### Custom Example
<img src="./pictures/custom_cloud_builder_example.png" alt="drawing" width="600"/>

### Cloud Build Config Example
<img src="./pictures/cloud_build_build_config_example.png" alt="drawing" width="600"/>

### Cloud Build Persistent Directory
<img src="./pictures/cloud_build_persistence_dir.png" alt="drawing" width="600"/>

### Cloud Build Substitutions Example
<img src="./pictures/cloud_build_substitutions.png" alt="drawing" width="600"/>

## Lab

In this lab, you develop a Cloud Build CI/CD workflow that automatically builds and deploys a Kubeflow Pipeline (KFP). You also integrate your workflow with GitHub by setting up a trigger that starts the workflow when a new tag is applied to the GitHub repo that hosts the pipeline's code.

Objectives:
- Create a custom Cloud Build builder to pilot AI Platform Pipelines.
- Write a Cloud Build config file to build and push all the artifacts for a KFP.
- Set up a Cloud Build Github trigger to rebuild the KFP.


[lab-03.ipynb](./labs/lab-03.ipynb)
Downloaded from [here](https://github.com/GoogleCloudPlatform/mlops-on-gcp/blob/de86e262cb5a79a201f7c719c3f1b0173cecaf52/on_demand/kfp-caip-sklearn/lab-03-kfp-cicd/lab-03.ipynb)

In this lab, you developed a Cloud Build CI/CD workflow that automatically builds and deploys a Kubeflow Pipeline (KFP). You also integrated your workflow with GitHub by setting up a trigger that starts the workflow when a new tag is applied to the GitHub repo that hosts the pipeline's code.