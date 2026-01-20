data "aws_caller_identity" "current" {}

locals {
  prefix = "${var.project_name}-${var.owner}"
  suffix = data.aws_caller_identity.current.account_id
}

resource "aws_s3_bucket" "data" {
  bucket = "${local.prefix}-${local.suffix}-data"
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
