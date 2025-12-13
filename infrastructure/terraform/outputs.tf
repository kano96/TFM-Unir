output "s3_bucket_name" {
  value = aws_s3_bucket.data.bucket
}

output "sns_topic_arn" {
  value = aws_sns_topic.alerts.arn
}

output "ecr_repos" {
  value = {
    detector  = aws_ecr_repository.detector.repository_url
    predictor = aws_ecr_repository.predictor.repository_url
    rca       = aws_ecr_repository.rca.repository_url
    simulator = aws_ecr_repository.simulator.repository_url
  }
}
