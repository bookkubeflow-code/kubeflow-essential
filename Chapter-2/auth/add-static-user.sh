#!/bin/bash
# Add a static user to Dex authentication
# This script demonstrates adding a new user (jane.doe@example.com)

# Step 1: Edit the Dex ConfigMap to add the new user
# In the config.yaml section, add to the staticPasswords array:
#
# staticPasswords:
# - email: user@example.com
#   hashFromEnv: DEX_USER_PASSWORD
#   username: user
#   userID: "15841185641784"
# - email: jane.doe@example.com
#   hashFromEnv: JANE_DOE_PASSWORD
#   username: jane-doe
#   userID: "15841185641785"

echo "Opening Dex ConfigMap for editing..."
kubectl edit configmap dex -n auth

# Step 2: Generate a password hash
# Install passlib and bcrypt if not already installed
# python3 -m venv ~/kubeflow-passlib-env
# source ~/kubeflow-passlib-env/bin/activate
# pip install passlib bcrypt

echo ""
echo "Generate password hash with:"
echo '  python3 -c "from passlib.hash import bcrypt; import getpass; print(bcrypt.using(rounds=12, ident=\"2y\").hash(getpass.getpass()))"'

# Step 3: Update the Dex passwords secret with the hash
# Replace the hash below with your generated hash
# kubectl patch secret dex-passwords -n auth --type='merge' -p='{
#   "stringData": {
#     "JANE_DOE_PASSWORD": "$2y$12$YOUR_GENERATED_HASH_HERE"
#   }
# }'

# Step 4: Restart Dex to apply changes
echo ""
echo "After updating the secret, restart Dex:"
echo "  kubectl rollout restart deployment dex -n auth"
