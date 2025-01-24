# youtube-jaywalking

## Overview


## Usage of the code
The code is open-source and free to use. It is aimed for, but not limited to, academic research. We welcome forking of this repository, pull requests, and any contributions in the spirit of open science and open-source code 😍😄 For inquiries about collaboration, you may contact Md Shadab Alam (md_shadab_alam@outlook.com) or Pavlo Bazilinskyy (pavlo.bazilinskyy@gmail.com).

## Getting Started
Tested with Python 3.9.19. To setup the environment run these two commands in a parent folder of the downloaded repository (replace `/` with `\` and possibly add `--user` if on Windows:


**Step 1:**

Clone the repository
```command line
git clone https://github.com/bazilinskyy/youtube-jaywalking.git
```

**Step 2:**

Create a new virtual environment
```command line
python -m venv venv
```

**Step 3:**

Activate the virtual environment
```command line
source venv/bin/activate
```

On Windows use
```command line
venv\Scripts\activate
```

**Step 4:**

Install dependencies
```command line
pip install -r requirements.txt
```
**Step 5:**

Ensure you have the required datasets in the data/ directory, including the mapping.csv file.

**Step 6:**

Run the code:
```command line
python3 analysis.py
```

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Results

Countries where the study has been conducted
[![Countries where the study has been conducted](figures/world_map.png)](https://htmlpreview.github.io/?https://github.com/Shaadalam9/youtube-pedestrian/blob/main/figures/world_map.html)

Count of pedestrian crossing the street in the presence of traffic lights or traffic signs
[![Count of pedestrian crossing the street in the presence of traffic lights or traffic signs](figures/count_of_pedestrian_crossing_with_traffic_equipment_avg.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/youtube-jaywalking/blob/main/figures/count_of_pedestrian_crossing_with_traffic_equipment_avg.html)

Count of pedestrian crossing the street in the presence of traffic lights or traffic signs (normalised with time)
[![Count of pedestrian crossing the street in the presence of traffic lights or traffic signs (normalised with time)](figures/count_of_pedestrian_crossing_with_traffic_equipment_norm.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/youtube-jaywalking/blob/main/figures/count_of_pedestrian_crossing_with_traffic_equipment_norm.html)

Count of pedestrian crossing the street in the absence of traffic lights or traffic signs
[![Count of pedestrian crossing the street in the absence of traffic lights or traffic signs](figures/count_of_pedestrian_crossing_without_traffic_equipment_avg.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/youtube-jaywalking/blob/main/figures/count_of_pedestrian_crossing_without_traffic_equipment_avg.html)

Count of pedestrian crossing the street in the absence of traffic lights or traffic signs (normalised with time)
[![Count of pedestrian crossing the street in the absence of traffic lights or traffic signs (normalised with time)](figures/count_of_pedestrian_crossing_without_traffic_equipment_norm.png?raw=true)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/youtube-jaywalking/blob/main/figures/count_of_pedestrian_crossing_without_traffic_equipment_norm.html)


## Contact
If you have any questions or suggestions, feel free to reach out to md_shadab_alam@outlook.com