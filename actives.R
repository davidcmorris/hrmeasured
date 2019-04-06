# Install tidyverse
install.packages("tidyverse")
# Load library(tidyverse)
# Import data set (make sure to set working directory)
staffing <- read_csv("20190406_hr_measured_staffing.csv")
# Check to see if it read in correctly
head(staffing)
