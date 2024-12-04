library(data.table) # Efficient for large CSV files

# Load the dataset
file_path <- "/home/reinaldo_medina_gcp/Documents/competencia_03_forecast_cleaned.csv"
data <- fread(file_path)

# Function to clean and standardize feature names
clean_feature_names <- function(names_vec) {
  names_vec <- gsub("[^[:alnum:]_]", "_", names_vec) # Replace non-alphanumeric characters with '_'
  names_vec <- gsub("__+", "_", names_vec)           # Replace multiple underscores with a single '_'
  names_vec <- gsub("^_|_$", "", names_vec)          # Remove leading and trailing underscores
  return(names_vec)
}

# Apply cleaning to column names
colnames(data) <- clean_feature_names(colnames(data))

# Save the cleaned dataset
cleaned_file_path <- gsub("\\.csv$", "_cleaned.csv", file_path)
fwrite(data, cleaned_file_path)

# Validate that no special JSON characters remain
validate_clean_names <- function(names_vec) {
  if (any(grepl("[^[:alnum:]_]", names_vec))) {
    stop("Some feature names still contain invalid characters.")
  } else {
    message("Feature names are clean and standardized.")
  }
}

# Run the validation
validate_clean_names(colnames(data))

# Output cleaned feature names
print(colnames(data))
dataset = data



library(lightgbm)   # For LightGBM modeling
# Assume 'edad' is the target, and other columns are features
target <- "edad"
features <- setdiff(colnames(data), target)

# Handle missing values (if any)
data[is.na(data)] <- -999 # Replace NAs with a placeholder value

# Split into training and test sets (80-20 split)
set.seed(123) # For reproducibility
train_idx <- sample(1:nrow(data), size = 0.008 * nrow(data))
train_data <- data[train_idx, ]
test_data <- data[-train_idx, ]

# Convert to LightGBM datasets
dtrain <- lgb.Dataset(data = as.matrix(train_data[, ..features]), 
                      label = train_data[[target]])
dtest <- as.matrix(test_data[, ..features])

params <- list(
  objective = "regression",
  metric = "rmse",
  boosting = "gbdt",
  learning_rate = 0.1,
  num_leaves = 31
)

# Train LightGBM model
model <- lgb.train(
  params = params,
  data = dtrain,
  nrounds = 100,
  verbose = 0
)
