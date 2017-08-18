# k8scntkSamples
Samples showing how to use CNTK with Kubernetes

## Credits
A Big Thank you to William Buchwalter for helping me get started on acs-engine.

## Instructions
To get CNTK and Kubernetes running on Azure, acs-engine is the key ingredient to get the cluster up and running.

The following sites/resources helped me a lot in getting the Kurbenetes cluster up and running on Azure. 
* https://github.com/Azure/acs-engine
* [William's article on Medium.com](https://medium.com/@wbuchwalter/creating-a-kubernetes-cluster-with-gpu-support-on-azure-for-ml-training-and-predictions-with-a551a19b8859)

Once the K8S cluster is up and running, I used the following CNTK images:
* Serving - https://hub.docker.com/r/weehyong/cntkresnetgpu/
* Training - https://hub.docker.com/r/weehyong/cifar-10/

In the sample YAML files provided, you will see how we write the various files (checkpoints, logs, and model) to Azure Files. This allows you to use the models after the training job completes, and the pod is reclaimed.

## Notes
[This repo is work-in-progress]

