# setup.sh
#!/bin/bash
set -x  # Enable debug mode to show commands being executed

# Create streamlit config directory
mkdir -p ~/.streamlit/ || { echo "Error: Failed to create directory"; exit 1; }

# Log environment variables (excluding sensitive data)
echo "Checking environment..."
echo "PORT: ${PORT:-'not set'}"
echo "PYTHONPATH: ${PYTHONPATH:-'not set'}"

# Ensure PORT is set
if [ -z "$PORT" ]; then
    echo "Warning: PORT not set, using default 8501"
    PORT=8501
fi

# Write credentials file with error checking
if ! cat > ~/.streamlit/credentials.toml << EOL
[general]
email = "c-hi.yang@hotmail.sg"
EOL
then
    echo "Error: Failed to write credentials.toml"
    exit 1
fi

# Write config file with error checking
if ! cat > ~/.streamlit/config.toml << EOL
[server]
headless = true
enableCORS = false
port = ${PORT}
[logger]
level = "debug"
EOL
then
    echo "Error: Failed to write config.toml"
    exit 1
fi

echo "Streamlit configuration completed successfully"