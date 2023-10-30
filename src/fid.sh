#!/bin/bash

# Get the root folders from command line arguments
dir1=$1
dir2=$2

# Define the range of subdirectories
start=1
end=22

# Initialize the sum variable
sum=0

# Initialize variables for max and min FID values and their corresponding subdirectories
max_fid=0
min_fid=1000000
max_subdir=""
min_subdir=""

# Loop through the subdirectories
for ((i=start; i<=end; i++))
do
    # Run the command for each subdirectory and capture the output
    output=$(python -m pytorch_fid "$dir1/$i" "$dir2/$i")
    
    # Extract the float number at the end after 'FID'
    fid=$(echo $output | awk '{print $NF}')
    
    # Add the number to the sum
    sum=$(echo "$sum + $fid" | bc -l)
    
    # Check if this is a new maximum or minimum FID value and update variables accordingly
    if (( $(echo "$fid > $max_fid" | bc -l) )); then
        max_fid=$fid
        max_subdir="$i"
    fi
    
    if (( $(echo "$fid < $min_fid" | bc -l) )); then
        min_fid=$fid
        min_subdir="$i"
    fi
    
done

# Calculate the average
average=$(echo "$sum / $end" | bc -l)

echo "Average FID: $average"
echo "Maximum FID: $max_fid (in subdirectory $max_subdir)"
echo "Minimum FID: $min_fid (in subdirectory $min_subdir)"