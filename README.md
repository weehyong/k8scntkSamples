# k8scntkSamples
Samples showing how to use CNTK with Kurbenetes

## Instructions
To get CNTK and Kurbenetes running on Azure, acs-engine will be the key ingredient to get the Kurbenetes cluster up and running quickly.

The following sites/resources helped me a lot to get it up and running on Azure. A Big Thank you to William Buchwalter for helping me get started.
* https://github.com/Azure/acs-engine
* [William's article on Medium.com](https://medium.com/@wbuchwalter/creating-a-kubernetes-cluster-with-gpu-support-on-azure-for-ml-training-and-predictions-with-a551a19b8859)

Once the K8S cluster is up and running, I used the following CNTK images which I prepared for running
* Serving - https://hub.docker.com/r/weehyong/cntkresnetgpu/
* Training - https://hub.docker.com/r/weehyong/cifar-10/

In the sample YAML files provided, you will see how we write the various files (checkpoints, logs, and model) to Azure Files. This allows you to use serve the models after the training job completes, and the pod is reclaimed.

## Notes
[This repo is work-in-progress]

