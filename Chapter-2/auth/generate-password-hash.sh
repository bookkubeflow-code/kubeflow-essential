#!/bin/bash
# Generate bcrypt password hashes for Dex static users
# Requires Python 3.11+ with passlib and bcrypt

# Create and activate virtual environment
python3 -m venv ~/kubeflow-passlib-env
source ~/kubeflow-passlib-env/bin/activate

# Install dependencies
pip install passlib bcrypt

# Generate hash
echo "Enter password for the new user:"
python3 -c "from passlib.hash import bcrypt; import getpass; print('Hash:', bcrypt.using(rounds=12, ident='2y').hash(getpass.getpass()))"

# Deactivate virtual environment
deactivate

echo ""
echo "Use the hash above to update the Dex passwords secret:"
echo '  kubectl patch secret dex-passwords -n auth --type='"'"'merge'"'"' -p='"'"'{'
echo '    "stringData": {'
echo '      "USER_PASSWORD": "<paste-hash-here>"'
echo '    }'
echo '  }'"'"''
echo ""
echo "Then restart Dex:"
echo "  kubectl rollout restart deployment dex -n auth"
