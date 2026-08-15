#!/bin/bash
# Verifies the tightened session-uploader IAM role two ways, against a
# real Floci-emulated AWS: (1) live API calls with assumed-role
# credentials, and (2) iam:SimulatePrincipalPolicy, an authoritative
# policy evaluation Floci does implement even though it doesn't enforce
# IAM on live calls (see README's "A real Floci fidelity gap").
set -euo pipefail

export AWS_ENDPOINT_URL=http://localhost.floci.io:4566
export AWS_ACCESS_KEY_ID=test
export AWS_SECRET_ACCESS_KEY=test
export AWS_DEFAULT_REGION=us-east-1

echo "=== 1. Assume the tightened role ==="
CREDS=$(aws sts assume-role \
  --role-arn "arn:aws:iam::000000000000:role/tutorloop-session-uploader" \
  --role-session-name verify-session \
  --query "Credentials" --output json)
ASSUMED_KEY=$(echo "$CREDS" | python3 -c "import json,sys; print(json.load(sys.stdin)['AccessKeyId'])")
ASSUMED_SECRET=$(echo "$CREDS" | python3 -c "import json,sys; print(json.load(sys.stdin)['SecretAccessKey'])")
ASSUMED_TOKEN=$(echo "$CREDS" | python3 -c "import json,sys; print(json.load(sys.stdin)['SessionToken'])")
echo "assumed-role credentials obtained"

echo
echo "=== 2. In-scope call: s3:PutObject on the transcripts bucket ==="
echo "verify_iam.sh transcript" > /tmp/verify_session.txt
AWS_ACCESS_KEY_ID="$ASSUMED_KEY" AWS_SECRET_ACCESS_KEY="$ASSUMED_SECRET" AWS_SESSION_TOKEN="$ASSUMED_TOKEN" \
  aws s3api put-object --bucket tutorloop-session-transcripts --key verify_session.txt --body /tmp/verify_session.txt \
  && echo "PutObject: succeeded (expected, in scope)"

echo
echo "=== 3. iam:SimulatePrincipalPolicy, the tightened policy's real evaluation ==="
aws iam simulate-principal-policy \
  --policy-source-arn "arn:aws:iam::000000000000:role/tutorloop-session-uploader" \
  --action-names "s3:PutObject" "s3:DeleteBucket" "iam:CreateUser" \
  --resource-arns "arn:aws:s3:::tutorloop-session-transcripts/verify_session.txt" \
  --query "EvaluationResults[].{Action:EvalActionName,Decision:EvalDecision}" --output table

echo
echo "Done. See README.md \"A real Floci fidelity gap\" for why step 3's implicitDeny"
echo "is the authoritative check here, not a live out-of-scope API call."
