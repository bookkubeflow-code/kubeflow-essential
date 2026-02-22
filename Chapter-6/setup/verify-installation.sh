#!/bin/bash
# Verify Training Operator v1 and v2 installations

echo "=== Testing v1 (TFJob) ==="
kubectl apply -f - <<EOF
apiVersion: kubeflow.org/v1
kind: TFJob
metadata:
  name: test-tfjob
spec:
  tfReplicaSpecs:
    Worker:
      replicas: 1
      template:
        spec:
          containers:
          - name: tensorflow
            image: tensorflow/tensorflow:2.13.0
            command: ["python", "-c", "print('TFJob test successful')"]
EOF

echo "Waiting for TFJob..."
sleep 10
kubectl get TFJob test-tfjob

echo ""
echo "=== Cleaning up test TFJob ==="
kubectl delete tfjob test-tfjob

echo ""
echo "=== Testing v2 (TrainJob via Python SDK) ==="
echo "Run: python test-v2-trainjob.py (see examples/test-v2-trainjob.py)"
