"""
Author: Joel Weber
Date: 15/07/2023
Title: Proximal search
Description: Analyses the green points around a line to determine if it is actually in the center of a row also could catch if the line is not actually representative of a row

Input Data:
    - Set of averaged lines 
    - The width of the row

Strategy:
    1. Iterate through each averaged line and check the area around for the following conditions: Total green points, green point density map, 
    2. For the width+ an error margin create a cropped image
    3. Apply a 
"""

