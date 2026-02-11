# Production Deployment Plan — vidqc on AWS

## Status

Skeleton plan for v0.1. Full implementation details to be added after validation gates are passed (see SPEC.md §0.5).

## Required Gates

Before deploying to production:

1. v0.1 success criteria (§0.5) are met on real AI-generated clips
2. End-to-end latency benchmarked and acceptable (< 60s per clip on CPU)
3. Model validated on at least 10 hold-out clips

## Architecture

```
                          ┌──────────────┐
                          │  API Gateway │
                          │  (REST API)  │
                          └──────┬───────┘
                                 │
                          ┌──────▼───────┐
                          │   Lambda     │
                          │  (Router)    │
                          └──────┬───────┘
                                 │
                    ┌────────────┼────────────┐
                    │            │            │
              ┌─────▼────┐ ┌────▼─────┐ ┌────▼─────┐
              │  S3      │ │  ECS     │ │  DynamoDB│
              │  (Clips) │ │ (Worker) │ │ (Results)│
              └──────────┘ └──────────┘ └──────────┘
```

## Components

### 1. API Gateway + Lambda (Request Router)

- REST API for clip submission and result retrieval
- `POST /predict` — accepts video upload, queues processing job
- `GET /predict/{job_id}` — returns prediction result
- `POST /batch` — accepts list of S3 URIs for batch processing
- Lambda validates input, stores clip to S3, sends SQS message

### 2. S3 (Clip Storage)

- Bucket: `vidqc-clips-{env}`
- Lifecycle: delete clips after 30 days (configurable)
- Presigned URLs for direct upload from client (bypasses Lambda size limits)
- Separate prefix for models: `s3://vidqc-models-{env}/`

### 3. SQS (Job Queue)

- Standard queue for decoupling API from workers
- Dead letter queue for failed jobs (max 3 retries)
- Visibility timeout: 120s (matches max processing time)
- Message format: `{job_id, s3_key, config_overrides}`

### 4. ECS Fargate (Worker)

- Container image from `Dockerfile` (pushed to ECR)
- Task definition: 2 vCPU, 4GB RAM (CPU-only, no GPU)
- Auto-scaling: 0–10 tasks based on SQS queue depth
- Scale-to-zero when idle (cost optimization)
- Worker loop:
  1. Poll SQS for job
  2. Download clip from S3
  3. Run `vidqc predict --input clip.mp4`
  4. Write result to DynamoDB
  5. Delete SQS message

### 5. ECR (Container Registry)

- Repository: `vidqc`
- Image tagged with git SHA + `latest`
- CI/CD pushes new image on merge to `main`
- Image size target: < 3GB (EasyOCR models are the main contributor)

### 6. DynamoDB (Results Store)

- Table: `vidqc-results`
- Partition key: `job_id` (UUID)
- TTL: 90 days
- Schema: `{job_id, status, prediction, created_at, completed_at, error}`

### 7. CloudWatch (Monitoring)

- Metrics: job count, latency p50/p95/p99, error rate
- Alarms: error rate > 5%, p99 latency > 90s, DLQ depth > 0
- Log groups: `/ecs/vidqc-worker`, `/lambda/vidqc-router`

## CI/CD Pipeline

```
GitHub Push → GitHub Actions → Build + Test → Push to ECR → Deploy to ECS
```

1. `ci.yml` runs lint + tests on every PR
2. On merge to `main`:
   - Build Docker image
   - Push to ECR
   - Update ECS task definition
   - Rolling deployment (min 1 healthy task)

## Cost Estimate (Low Traffic: ~100 clips/day)

| Component | Monthly Cost |
|-----------|-------------|
| ECS Fargate (scale-to-zero) | ~$30–50 |
| S3 (10GB stored) | ~$0.25 |
| SQS | ~$0 (free tier) |
| DynamoDB (on-demand) | ~$1 |
| API Gateway | ~$3.50 |
| ECR | ~$1 |
| CloudWatch | ~$5 |
| **Total** | **~$40–60/month** |

## Security

- API Gateway: API key or Cognito auth
- S3: bucket policy restricts access to Lambda/ECS roles
- ECS: task role with least-privilege IAM
- No secrets in container image — use SSM Parameter Store
- VPC: worker tasks in private subnet, NAT gateway for outbound

## Scaling Considerations

- **Bottleneck**: OCR processing (~30-60s per clip)
- **Horizontal scaling**: Add ECS tasks (each processes 1 clip at a time)
- **GPU acceleration** (v0.2): Switch to GPU ECS instances for EasyOCR
- **Batch optimization** (v0.2): Process multiple clips per task with `--workers`

## Migration Path from v0.1 to v0.2

1. Add GPU support to Dockerfile (CUDA base image)
2. Implement `--workers` for parallel batch processing
3. Add WebSocket support for real-time progress updates
4. Consider Step Functions for multi-stage pipeline orchestration

## Next Steps

- [ ] Create Terraform/CDK infrastructure code
- [ ] Set up staging and production environments
- [ ] Configure Cognito user pool for API authentication
- [ ] Set up CloudWatch dashboards
- [ ] Load test with realistic traffic patterns
- [ ] Document runbook for common operational tasks
