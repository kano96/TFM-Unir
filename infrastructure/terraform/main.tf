locals {
  prefix = "${var.project_name}-${var.owner}"
}

# S3 bucket para datasets/modelos
resource "aws_s3_bucket" "data" {
  bucket = "${local.prefix}-data"
}

resource "aws_s3_bucket_versioning" "data_versioning" {
  bucket = aws_s3_bucket.data.id
  versioning_configuration {
    status = "Enabled"
  }
}

# ECR repos (uno por microservicio)
resource "aws_ecr_repository" "detector" {
  name = "${local.prefix}-detector"
}

resource "aws_ecr_repository" "predictor" {
  name = "${local.prefix}-predictor"
}

resource "aws_ecr_repository" "rca" {
  name = "${local.prefix}-rca"
}

resource "aws_ecr_repository" "simulator" {
  name = "${local.prefix}-simulator"
}

# SNS topic para alertas
resource "aws_sns_topic" "alerts" {
  name = "${local.prefix}-alerts"
}
