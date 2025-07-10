#!/bin/bash

# 1. Print the PID of the current script. '$$' is a special variable that holds the PID.
echo "Script running with PID: $$"

# 2. Prompt the user for the destination directory and read the input.
read -p "Please enter the destination directory name: " DEST_DIR

# Optional: Exit if the user did not enter a name.
if [ -z "$DEST_DIR" ]; then
    echo "Error: Destination directory name cannot be empty."
    exit 1
fi

# Replace with your Git repository URL
REPO_URL="https://github.com/your-username/your-repository.git"

echo "--- Starting clone/pull loop for directory '$DEST_DIR' ---"

# Infinite loop
while true
do
  # Check if the target directory exists
  if [ -d "$DEST_DIR" ]; then
    echo "Directory '$DEST_DIR' already exists, executing git pull..."
    # Navigate into the directory, pull, and navigate back out
    (cd "$DEST_DIR" && git pull)
  else
    echo "Directory '$DEST_DIR' not found, executing git clone..."
    git clone "$REPO_URL" "$DEST_DIR"
  fi

  echo "Operation complete, waiting for 10 seconds..."
  sleep 10
done