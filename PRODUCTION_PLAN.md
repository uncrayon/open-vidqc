# vidqc Production Deployment Plan

## Status

**Deferred until v0.1 validation gates are passed** (see SPEC.md §11.2).

## Required Gates

Before implementing this plan:

1. v0.1 success criteria (§0.5) are met on real AI-generated clips
2. End-to-end latency benchmarked and acceptable (< 60s per clip on CPU)
3. Model validated on at least 10 hold-out clips

## Architecture Overview

Target: 500 concurrent clip processing executions on AWS.

### Components

1. **Input**: S3 bucket for incoming clips
2. **Queue**: SQS for task distribution
3. **Compute**: ECS Fargate tasks (auto-scaling)
4. **Storage**:
   - S3 for clips and results
   - DynamoDB for job status tracking
5. **Model artifacts**: S3 bucket with trained model + scaler

### Workflow

```
1. User uploads clip → S3 bucket (input/)
2. S3 event → Lambda → SQS message
3. ECS Fargate task polls SQS
4. Task downloads clip from S3
5. Task runs vidqc predict
6. Task writes result to S3 (output/) and updates DynamoDB
7. Optional: CloudWatch metrics for monitoring
```

### Scaling Parameters

- ECS service: 5-500 tasks (auto-scale on SQS queue depth)
- Task size: 2 vCPU, 4GB RAM (for EasyOCR CPU inference)
- SQS visibility timeout: 120s (allows for <60s inference + overhead)

### Cost Estimate

(To be calculated after latency benchmarking)

## Next Steps

1. Validate v0.1 model performance
2. Benchmark single-task latency in Docker container
3. Implement Terraform configuration for AWS resources
4. Deploy to staging environment
5. Load test with synthetic workload
6. Production rollout

---

**Note:** This is a skeleton. Full implementation details will be added in Milestone 7 after core ML pipeline is validated.
