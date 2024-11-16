#!/bin/bash

# Create streamlit config directory
mkdir -p ~/.streamlit/ || { echo "Failed to create directory"; exit 1; }

# Ensure PORT is set
if [ -z "$PORT" ]; then
    PORT=8501  # Default fallback port
fi

# Write credentials file
cat > ~/.streamlit/credentials.toml << EOL
[general]
email = "c-hi.yang@hotmail.sg"
EOL

# Write config file
cat > ~/.streamlit/config.toml << EOL
[server]
headless = true
enableCORS = false
port = ${PORT}
EOL

echo "Streamlit configuration completed successfully"