# Notes from Lectures, Labs and Readings

### Demo Readme Template Code
<img src="./pictures/ml_code_minority.png" alt="drawing" width="600"/>

### Demo Readme Template Code
[serving_ml_prediction.ipynb](./labs/serving_ml_prediction.ipynb)
Downloaded from [here](https://github.com/GoogleCloudPlatform/training-data-analyst/blob/master/courses/machine_learning/deepdive2/production_ml/labs/serving_ml_prediction.ipynb)


# TensorFlow Extended (TFX) Overview
<img src="./pictures/tfx_overview.png" alt="drawing" width="600"/>

<img src="./pictures/tfx_overview_services.png" alt="drawing" width="600"/>

### TensorFlow Pipelines
<img src="./pictures/tfx_pipelines.png" alt="drawing" width="600"/>

# TFX Library Overview
 <img src="./pictures/tfx_libraries.png" alt="drawing" width="600"/>

## TensorFlow Standard Data Components
1. <img src="./pictures/example_gen.png" alt="drawing" width="600"/>
2. <img src="./pictures/statistics_gen.png" alt="drawing" width="600"/>
3. <img src="./pictures/schema_gen.png" alt="drawing" width="600"/>
4. <img src="./pictures/example_validator.png" alt="drawing" width="600"/>
5. <img src="./pictures/transform_1.png" alt="drawing" width="600"/>
6. <img src="./pictures/transform_2.png" alt="drawing" width="600"/>

## TensorFlow Standard Model Components
1. <img src="./pictures/trainer.png" alt="drawing" width="600"/>
2. <img src="./pictures/trainer_2.png" alt="drawing" width="600"/>
3. <img src="./pictures/tuner.png" alt="drawing" width="600"/>
4. <img src="./pictures/evaluator.png" alt="drawing" width="600"/>
5. <img src="./pictures/infravalidator.png" alt="drawing" width="600"/>
6. <img src="./pictures/pusher.png" alt="drawing" width="600"/>
7. <img src="./pictures/bulkinferer.png" alt="drawing" width="600"/>

TensorFlow Model Analysis (TFMA)
<img src="./pictures/tf_model_analysis_tfma.png" alt="drawing" width="600"/>

## TensorFlow Custom Components
[ComponentSpec](https://www.tensorflow.org/tfx/guide/custom_component#componentspec)

The ComponentSpec class defines the component contract by defining the input and output artifacts to a component as well as the parameters that are used for the component execution. There are three parts in it:
- INPUTS: A dictionary of typed parameters for the input artifacts that are into the component executor. Normally input artifacts are the outputs from upstream components and thus share the same type.
- OUTPUTS: A dictionary of typed parameters for the output artifacts which the component produces.
- PARAMETERS: A dictionary of additional ExecutionParameter items that will be passed into the component executor. These are non-artifact parameters that we want to define flexibly in the pipeline DSL and pass into execution.

## TFX Standard Components Walkthrough
[lab-01.ipynb](./labs/lab-01.ipynb). Need to run this to create files used in this lab
```
cd mlops-on-gcp/workshops/tfx-caip-tf23
./install.sh
```

## Learning Objectives

1.  Develop a high level understanding of TFX pipeline components.
2.  Learn how to use a TFX Interactive Context for prototype development of TFX pipelines.
3.  Work with the Tensorflow Data Validation (TFDV) library to check and analyze input data.
4.  Utilize the Tensorflow Transform (TFT) library for scalable data preprocessing and feature transformations.
5.  Employ the Tensorflow Model Analysis (TFMA) library for model evaluation.

In this lab, you will work with the [Covertype Data Set](https://github.com/jarokaz/mlops-labs/blob/master/datasets/covertype/README.md) and use TFX to analyze, understand, and pre-process the dataset and train, analyze, validate, and deploy a multi-class classification model to predict the type of forest cover from cartographic features.

You will utilize  **TFX Interactive Context** to work with the TFX components interactivelly in a Jupyter notebook environment. Working in an interactive notebook is useful when doing initial data exploration, experimenting with models, and designing ML pipelines. You should be aware that there are differences in the way interactive notebooks are orchestrated, and how they access metadata artifacts. In a production deployment of TFX on GCP, you will use an orchestrator such as Kubeflow Pipelines, or Cloud Composer. In an interactive mode, the notebook itself is the orchestrator, running each TFX component as you execute the notebook cells. In a production deployment, ML Metadata will be managed in a scalabe database like MySQL, and artifacts in apersistent store such as Google Cloud Storage. In an interactive mode, both properties and payloads are stored in a local file system of the Jupyter host.

**Setup Note:**
Currently, TFMA visualizations do not render properly in JupyterLab. It is recommended to run this notebook in Jupyter Classic Notebook. To switch to Classic Notebook select *Launch Classic Notebook* from the *Help* menu.

## Next steps

This concludes your introductory walthrough through TFX pipeline components. In the lab, you used TFX to analyze, understand, and pre-process the dataset and train, analyze, validate, and deploy a multi-class classification model to predict the type of forest cover from cartographic features. You utilized a TFX Interactive Context for prototype development of a TFX pipeline directly in a Jupyter notebook. Next, you worked with the TFDV library to modify your dataset schema to add feature constraints to catch data anamolies that can negatively impact your model's performance. You utilized TFT library for feature proprocessing for consistent feature transformations for your model at training and serving time. Lastly, using the TFMA library, you added model performance constraints to ensure you only push more accurate models than previous runs to production.

The next labs in the series will guide through developing a TFX pipeline, deploying and running the pipeline on **AI Platform Pipelines** and automating the pipeline build and deployment processes with **Cloud Build**.

## Pipeline Orchestration with TFX
<img src="./pictures/why_orchestrate.png" alt="drawing" width="600"/>
<img src="./pictures/tfx_orchestration_on_a_notebook.png" alt="drawing" width="600"/>

## Apache Beam TFX Relationship 
<img src="./pictures/apache_beam_scales_tfx.png" alt="drawing" width="600"/>
<img src="./pictures/how_tfx_uses_beam.png" alt="drawing" width="600"/>

## TFX on Google Cloud
<img src="./pictures/high_level_of_tfx_on_google_cloud.png" alt="drawing" width="600"/>
<img src="./pictures/high_level_of_tfx_on_google_cloud_2.png" alt="drawing" width="600"/>
<img src="./pictures/high_level_of_tfx_on_google_cloud_3.png" alt="drawing" width="600"/>
<img src="./pictures/high_level_of_tfx_on_google_cloud_4.png" alt="drawing" width="600"/>

## TFX on Cloud AI Platform Pipelines 
[lab-02.ipynb](./labs/lab2/lab-02.ipynb). Need to run this to create files used in this lab
```
cd mlops-on-gcp/workshops/tfx-caip-tf23
./install.sh
```

In this lab, you learned how to manually build and deploy a TFX pipeline to AI Platform Pipelines and trigger pipeline runs from a notebook

### Overview
In this lab, you use utilize the following tools and services to deploy and run a TFX pipeline on Google Cloud that automates the development and deployment of a TensorFlow 2.3 WideDeep Classifer to predict forest cover from cartographic data:
- The TFX CLI utility to build and deploy a TFX pipeline.
- A hosted AI Platform Pipeline instance (Kubeflow Pipelines) for TFX pipeline orchestration.
- Dataflow jobs for scalable, distributed data processing for TFX components.
- A AI Platform Training job for model training and flock management for parallel tuning trials.
- AI Platform Prediction as a model server destination for blessed pipeline model versions.
- CloudTuner and AI Platform Vizier for advanced model hyperparameter tuning using the Vizier algorithm.
- You will then create and monitor pipeline runs using the TFX CLI as well as the KFP UI.

### Objectives
- Use the TFX CLI to build a TFX pipeline.
- Deploy a TFX pipeline version without tuning to a hosted AI Platform Pipelines instance.
- Create and monitor a TFX pipeline run using the TFX CLI.
- Deploy a new TFX pipeline version with tuning enabled to a hosted AI Platform Pipelines instance.
- Create and monitor another TFX pipeline run directly in the KFP UI.

## TFX Pipeline Design Pattern
<img src="./pictures/tfx_pipeline_design_pattern.png" alt="drawing" width="600"/>

## TFX Pipeline as Docker Container
<img src="./pictures/tfx_pipelines_as_docker_container.png" alt="drawing" width="600"/>

## CI/CD for training TFX Pipelines on Google Cloud
<img src="./pictures/ci_cd_training_tfx_pipelines_on_google_cloud.png" alt="drawing" width="600"/>

## Pipeline Automation Steps
### Level 0 - TFX Pipeline Notebook Prototyping
<img src="./pictures/level_0_tfx_pipeline_notebook_prototyping.png" alt="drawing" width="600"/>

### Level 1 TFX Pipeline Continous Training
<img src="./pictures/level_1_tfx_pipeline_continous_training.png" alt="drawing" width="600"/>

### Level 2 TFX CI/CD Pipelines
<img src="./pictures/level_2_ml_development_automation_tfx_ci_cd_pipelines.png" alt="drawing" width="600"/>

### end_to_end_tfx_mlops_workflow
<img src="./pictures/end_to_end_tfx_mlops_workflow.png" alt="drawing" width="600"/>

### future_developments_feature_store
<img src="./pictures/future_developments_feature_store.png" alt="drawing" width="600"/>


## Git [forks](https://docs.github.com/en/get-started/quickstart/fork-a-repo)

Most commonly, forks are used to either propose changes to someone else's project or to use someone else's project as a starting point for your own idea. You can fork a repository to create a copy of the repository and make changes without affecting the upstream repository. For more information, see "Working with forks."
Propose changes to someone else's project

For example, you can use forks to propose changes related to fixing a bug. Rather than logging an issue for a bug you've found, you can:
- Fork the repository.
- Make the fix.
- Submit a pull request to the project owner.

## [Building repositories](https://cloud.google.com/build/docs/automating-builds/build-repos-from-github) from GitHub 

GitHub triggers enable you to automatically build on Git pushes and pull requests and view your build results on GitHub and Cloud Console. Additionally, GitHub triggers support all the features supported by the existing GitHub triggers and use the Cloud Build GitHub app to configure and authenticate to GitHub.

This page explains how to create GitHub triggers and build on GitHub using the Cloud Build GitHub app.

## TFX on Cloud AI Platform Pipelines 
[lab-03.ipynb](./labs/lab-03.ipynb). Need to run this to create files used in this lab
```
cd mlops-on-gcp/workshops/tfx-caip-tf23
./install.sh
```

Objectives:
- Develop a CI/CD workflow with Cloud Build to build and deploy a machine learning pipeline.
- Integrate with Github to trigger workflows with pipeline source repository changes.

In this lab, you walked through authoring a Cloud Build CI/CD workflow that automatically builds and deploys a TFX pipeline. You also integrated your TFX workflow with GitHub by setting up a Cloud Build trigger. In the next lab, you will walk through inspection of TFX metadata and pipeline artifacts created during TFX pipeline runs.