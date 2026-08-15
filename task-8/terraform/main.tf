# A small, self-contained IAM least-privilege exercise (not a full
# deployment like FleetPulse's task-9): an S3 bucket for tutoring-session
# transcripts, and an IAM role for a hypothetical "session uploader"
# Lambda that writes transcripts there. The role starts deliberately
# over-broad (see git history/README for the before/after), checkov
# flags it, then it's tightened to least privilege, applied to Floci, and
# an out-of-scope action is confirmed to get AccessDenied.

resource "aws_s3_bucket" "session_transcripts" {
  bucket = "tutorloop-session-transcripts"
}

resource "aws_iam_role" "session_uploader" {
  name = "tutorloop-session-uploader"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "lambda.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy" "session_uploader_policy" {
  name = "tutorloop-session-uploader-policy"
  role = aws_iam_role.session_uploader.id

  # Tightened from the original Action="*"/Resource="*" (see git history
  # and README for the before/after checkov scan): a session uploader
  # only ever needs to write and read transcript objects in this one
  # bucket, nothing else. No iam:*, no s3:DeleteBucket, no access to any
  # other bucket, scoped to specific actions on this bucket's ARN and its
  # objects, not a wildcard resource.
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Sid    = "SessionTranscriptReadWrite"
      Effect = "Allow"
      Action = [
        "s3:PutObject",
        "s3:GetObject",
      ]
      Resource = "${aws_s3_bucket.session_transcripts.arn}/*"
    }]
  })
}
