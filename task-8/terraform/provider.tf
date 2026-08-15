terraform {
  required_version = ">= 1.5"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

# Floci is a local AWS-API-compatible emulator (LocalStack-compatible
# protocol), same pattern FleetPulse's task-9 uses: fake credentials,
# every service endpoint pointed at Floci instead of real AWS, so this
# plan/apply never touches a real account.
provider "aws" {
  region                      = "us-east-1"
  access_key                  = "test"
  secret_key                  = "test"
  skip_credentials_validation = true
  skip_metadata_api_check     = true
  skip_requesting_account_id  = true

  endpoints {
    s3  = "http://localhost.floci.io:4566"
    iam = "http://localhost.floci.io:4566"
    sts = "http://localhost.floci.io:4566"
  }
}
