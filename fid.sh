#!/bin/bash

# Get the root folders from command line arguments
dir1=$1
dir2=$2

# Define the range of subdirectories
start=1
end=22

# Initialize the sum variable
sum=0

# Loop through the subdirectories
for ((i=start; i<=end; i++))
do
    # Run the command for each subdirectory and capture the output
    output=$(python -m pytorch_fid "$dir1/$i" "$dir2/$i")
    
    # Extract the float number at the end after 'FID'
    fid=$(echo $output | awk '{print $NF}')
    
    # Add the number to the sum
    sum=$(echo "$sum + $fid" | bc -l)
done

# Calculate the average
average=$(echo "$sum / $end" | bc -l)

echo "Average FID: $average"
