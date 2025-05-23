#!/bin/bash

# Navigate to your directory containing the subfolders
# cd /path/to/your/folder

for folder in valid_gaindata_*; do
  if [[ -d "$folder" ]]; then
    # Extract the parts after valid_gaindata_
    suffix=${folder#valid_gaindata_}
    
    # Replace cosinesched with cossched
    if [[ $suffix == cosinesched* ]]; then
      new_name="cossched${suffix#cosinesched}"
    else
      new_name=$suffix
    fi
    
    # Rename the folder
    mv "$folder" "$new_name"
    echo "Renamed: $folder -> $new_name"
  fi
done

# Note: nophys folders should remain unchanged