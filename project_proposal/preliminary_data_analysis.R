library(ggplot2)
library(dplyr)
library(tidyr)
df <- read.csv("training_data_clean.csv")
# Converting ordinal data to actual numbers
df <- df %>%
  mutate(
    !!names(df)[3] := as.integer(substr(df[[3]], 1, 1)),
    !!names(df)[5] := as.integer(substr(df[[5]], 1, 1)),
    !!names(df)[8] := as.integer(substr(df[[8]], 1, 1)),
    !!names(df)[9] := as.integer(substr(df[[9]], 1, 1))
  )
# Changing name of some columns
names(df)[3] <- "use_academic" # name of 3rd column changed
names(df)[5] <- "sub_optimal_resp"
names(df)[8] <- "get_source"
names(df)[9] <- "verify"


# Initial attempt at categorizing the free text response data
df <- df %>%
  mutate(
    !!names(df)[2] := case_when(
      grepl("code|coding|programming", df[[2]], ignore.case = TRUE) ~ "coding",
      grepl("math", df[[2]], ignore.case = TRUE) ~ "math",
      grepl("study|explain|complicated|simplify|academic", df[[2]], ignore.case = TRUE) ~ "study",
      grepl("\\bsummar", df[[2]], ignore.case = TRUE) ~ "summarising",
      grepl("email|resume", df[[2]], ignore.case = TRUE) ~ "professional_document",
      grepl("writing|essay", df[[2]], ignore.case = TRUE) ~ "writing",
      TRUE ~ paste0("!", df[[2]])
    )
  )


# Seperating data set into 3 subsets depending on which model was used
df_gemin  <- df %>% filter(label == "Gemini")
df_chatgpt <- df %>% filter(label == "ChatGPT")
df_claude <- df %>% filter(label == "Claude")
# Based on the label, we create 3 bar charts on a single diagram. Each bar chart contains the mean value of some feature
# depending on the model
df_summary <- df %>%
  select(label, all_of(c(3, 5, 8, 9))) %>%
  pivot_longer(
    cols = -label,
    names_to = "Feature",
    values_to = "Value"
  ) %>%
  group_by(label, Feature) %>%
  summarise(mean_value = mean(Value, na.rm = TRUE), .groups = "drop")

ggplot(df_summary, aes(x = Feature, y = mean_value, fill = label)) +
  geom_col(position = "dodge") +
  labs(
    title = "Mean Value per Feature by Model Label",
    x = "Feature (Ordinal)",
    y = "Mean of Values"
  ) +
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 0, hjust = 1),
    plot.title = element_text(size = 14, face = "bold")
  )




