#!/bin/bash
# Change the default Kubeflow credentials for production security

# Generate a password hash
echo "Enter new password:"
HASH=$(python3 -c 'from passlib.hash import bcrypt; import getpass; print(bcrypt.using(rounds=12, ident="2y").hash(getpass.getpass()))')

echo "Generated hash: $HASH"

# Update the secret
kubectl create secret generic dex-passwords \
    --from-literal=DEX_USER_PASSWORD="$HASH" \
    -n auth \
    --dry-run=client -o yaml | kubectl apply -f -

# Restart Dex
kubectl rollout restart deployment dex -n auth

echo "Default credentials updated. Restart Dex deployment to apply."
