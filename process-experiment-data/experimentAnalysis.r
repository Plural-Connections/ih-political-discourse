library(lme4)
library(texreg)
library(WebPower)
library(broom)
library(expss)
library(dplyr)
library(ggsci)
library(jtools)
library(ggplot2)
library(patchwork)

df = read.csv("usersround2/processed_data/output-06-2025.csv")

df$social_cue_nudge <- factor(df$social_cue_nudge, ordered = FALSE, levels=c('True', 'False'))
df$social_cue_nudge <- relevel(df$social_cue_nudge, ref='False')
df$actually_treated <- factor(df$actually_treated, ordered = FALSE, levels=c('True', 'False'))
df$actually_treated <- relevel(df$actually_treated, ref='False')
df$environment_condtion <- factor(df$environment_condtion, ordered = FALSE, levels=c('IA', 'IH', 'NEUTRAL'))
df$environment_condtion <- relevel(df$environment_condtion, ref="NEUTRAL")
df$stance <- factor(df$stance, ordered = FALSE)
df$change_ih <- df$post_IH - df$pre_IH
df$scaled_change_ih <- 2 * (df$change_ih - min(df$change_ih, na.rm=TRUE)) / 
  (max(df$change_ih, na.rm=TRUE) - min(df$change_ih, na.rm=TRUE)) - 1
df$scaled_pre_ih <- 2 * (df$pre_IH - min(df$pre_IH, na.rm=TRUE)) / 
  (max(df$pre_IH, na.rm=TRUE) - min(df$pre_IH, na.rm=TRUE)) - 1

df$commentLength <- df$commentLength
df$comment <- df$comment
df$demonstratedIH <- df$demonstratedIH

table(df$environment_condtion, df$actually_treated)

df %>% 
  group_by(environment_condtion, social_cue_nudge) %>% 
  summarise(val = mean(commentLength, na.rm=TRUE))

# DEMONSTRATED IH MODELS

summary(df$demonstratedIH)
summary(df$post_ih)
summary(df$scaled_pre_ih)

df$post_IH

dem_base <- lm(demonstratedIH ~ social_cue_nudge + environment_condtion, data=df)
rep_base <- lm(change_ih ~ social_cue_nudge + environment_condtion, data=df)
eng_comment_base <- lm(comment ~ social_cue_nudge + environment_condtion, data=df)
summ(dem_base, confint = TRUE, digits = 2)
summ(rep_base, confint = TRUE, digits = 2)
summ(eng_comment_base, confint = TRUE, digits = 2)

dem_base_w_covariates <- lm(demonstratedIH ~ social_cue_nudge + environment_condtion + stance + scaled_pre_ih, data=df)
rep_base_w_covariates <- lm(change_ih ~ social_cue_nudge + environment_condtion + stance, data=df)
eng_comment_base_w_covariates <- lm(comment ~ social_cue_nudge + environment_condtion + stance + scaled_pre_ih, data=df)
summ(dem_base_w_covariates, confint = TRUE, digits = 2)
summ(rep_base_w_covariates, confint = TRUE, digits = 2)
summ(eng_comment_base_w_covariates, confint = TRUE, digits = 2)

dem_interaction <- lm(demonstratedIH ~ social_cue_nudge*environment_condtion, data=df)
rep_interaction <- lm(change_ih ~ social_cue_nudge*environment_condtion, data=df)
eng_comment_interaction<- lm(comment ~ social_cue_nudge*environment_condtion, data=df)
summ(dem_interaction, confint = TRUE, digits = 2)
summ(rep_interaction, confint = TRUE, digits = 2)
summ(eng_comment_interaction, confint=TRUE, digits = 2)

dem_interaction_covariates <- lm(demonstratedIH ~ social_cue_nudge*environment_condtion*(stance + scaled_pre_ih), data=df)
rep_interaction_covariates <- lm(change_ih ~ social_cue_nudge*environment_condtion*(stance + scaled_pre_ih), data=df)

# Function to generate a clean plot
make_plot <- function(base_model, title, ylab) {
  plot_coefs(base_model,
             coefs = c("Social Cue Nudge" ="social_cue_nudgeTrue", 
                       "IH Environment" = "environment_condtionIH", 
                       "IA Environment" = "environment_condtionIA"),
             show.p = TRUE,
             scale = TRUE) +
    labs(title = title, y = ylab) +      # Y-axis on right
    theme_minimal() +
    theme(legend.position = "none")
}

# Create the three plots
p1 <- make_plot(dem_base, "(A) Demonstrated IH", "Demonstrated IH") + ylab(NULL) + theme(axis.text.y = element_text(size = 14))
p2 <- make_plot(rep_base, "(B) Change in Self-Reported IH", "Change in Self-Reported IH") + ylab(NULL) + theme(axis.text.y = element_blank(),
                                                                                                               axis.ticks.y = element_blank())
p3 <- make_plot(eng_comment_base, "(C) Number of Comments", "Number of Comments") + ylab(NULL) + theme(axis.text.y = element_blank(),
                                                                                                       axis.ticks.y = element_blank())

# Combine plots with shared legend
plot_grid(p1, p2, p3,
          ncol = 3,
          rel_widths = c(1.5, 1, 1))    # align by left axis
plt_all


# Plot Marginal Effects
df$social_cue_nudge <- factor(df$social_cue_nudge, levels=c('True', 'False'))
df$environment_condtion <- factor(df$environment_condtion, levels=c('IA', 'NEUTRAL', 'IH'))
# Colors/shapes
p_int_colors <- pal_npg()(4)[c(3,4)]  # 2 social cue groups
p_int_shapes <- c(15, 19, 17)         # 3 environment groups

# Aggregate data if not done already
df_summary_dem <- df %>%
  group_by(social_cue_nudge, environment_condtion) %>%
  summarise(
    mean_dIH = mean(demonstratedIH, na.rm = TRUE),
    sd_dIH = sd(demonstratedIH, na.rm = TRUE),
    n = n(),
    .groups = "drop"
  ) %>%
  mutate(
    se = sd_dIH / sqrt(n),
    lb = mean_dIH - 1.96 * se,  # lower 95% CI
    ub = mean_dIH + 1.96 * se   # upper 95% CI
  )

# Aggregate data if not done already
df_summary_rep <- df %>%
  group_by(social_cue_nudge, environment_condtion) %>%
  summarise(
    mean_rIH = mean(scaled_change_ih, na.rm = TRUE),
    sd_rIH = sd(scaled_change_ih, na.rm = TRUE),
    n = n(),
    .groups = "drop"
  ) %>%
  mutate(
    se = sd_rIH / sqrt(n),
    lb = mean_rIH - 1.96 * se,  # lower 95% CI
    ub = mean_rIH + 1.96 * se   # upper 95% CI
  )

# Aggregate data if not done already
df_summary_comments <- df %>%
  group_by(social_cue_nudge, environment_condtion) %>%
  summarise(
    mean_Comments = mean(comment, na.rm = TRUE),
    sd_Comments = sd(comment, na.rm = TRUE),
    n = n(),
    .groups = "drop"
  ) %>%
  mutate(
    se = sd_Comments / sqrt(n),
    lb = mean_Comments - 1.96 * se,  # lower 95% CI
    ub = mean_Comments + 1.96 * se   # upper 95% CI
  )


# Plot with environment on x-axis
plt1 <- ggplot(df_summary_dem, aes(x = environment_condtion, y = mean_dIH, color = social_cue_nudge, group = social_cue_nudge)) +
  geom_line(size = 1) +
  geom_pointrange(aes(ymin = lb, ymax = ub, shape = social_cue_nudge),
                  position = position_dodge(width = 0),
                  size = 0.8) +
  scale_color_manual(values = p_int_colors) +
  scale_shape_manual(values = c(15, 19)) +  # match social cue levels
  scale_y_continuous(
    breaks = scales::pretty_breaks(n = 11),
    limits = c(-1, 1)
  ) +
  labs(
    title = "(A) Demonstrated IH",
    x = "Environment Condition",
    y = "Demonstrated IH",
    color = "Social Cue",
    shape = "Social Cue"
  ) +
  theme_minimal() +
  theme(
    legend.position = "right",
    axis.text.x = element_text(hjust = 0.5)
  )

plt2 <- ggplot(df_summary_rep, aes(x = environment_condtion, y = mean_rIH, color = social_cue_nudge, group = social_cue_nudge)) +
  geom_line(size = 1) +
  geom_pointrange(aes(ymin = lb, ymax = ub, shape = social_cue_nudge),
                  position = position_dodge(width = 0),
                  size = 0.8) +
  scale_color_manual(values = p_int_colors) +
  scale_shape_manual(values = c(15, 19)) +  # match social cue levels
  scale_y_continuous(
    breaks = scales::pretty_breaks(n = 11),
    limits = c(-1, 1)
  ) +
  labs(
    title = "(B) Self Reported IH",
    x = "Environment Condition",
    y = "Change in Self-Reported IH",
    color = "Social Cue",
    shape = "Social Cue"
  ) +
  theme_minimal() +
  theme(
    legend.position = "right",
    axis.text.x = element_text(hjust = 0.5)
  )

plt3 <- ggplot(df_summary_comments, aes(x = environment_condtion, y = mean_Comments, color = social_cue_nudge, group = social_cue_nudge)) +
  geom_line(size = 1) +
  geom_pointrange(aes(ymin = lb, ymax = ub, shape = social_cue_nudge),
                  position = position_dodge(width = 0),
                  size = 0.8) +
  scale_color_manual(values = p_int_colors) +
  scale_shape_manual(values = c(15, 19)) +  # match social cue levels
  scale_y_continuous(
    breaks = scales::pretty_breaks(n = 11),
    limits = c(0, 10)
  ) +
  labs(
    title = "(C) Engagement",
    x = "Environment Condition",
    y = "Number of Comments",
    color = "Social Cue",
    shape = "Social Cue"
  ) +
  theme_minimal() +
  theme(
    legend.position = "right",
    axis.text.x = element_text(hjust = 0.5)
  )

plt1

plt2

plt3

plt_all <- plot_grid(
  plt1, plt2, plt3,
  nrow = 1, 
  ncol = 3
)

plt_all


plot_all_covariates <- plot_summs(dem_base_w_covariates, dem_interaction_covariates,
                              show.p = TRUE)
plot_all_covariates

p<- plot_summs(demonstrated_model_0, demonstrated_model_1, demonstrated_model_2,
           show.p = TRUE,
           scale = TRUE,
           coefs = c("Social Cue Nudge" = "actually_treatedTrue", 
                     "IH Environment" = "environment_condtionIH", 
                     "IA Environment" = "environment_condtionIA", 
                     "Social Cue Nudge \nx IH Environment" = "social_cue_nudgeTrue:environment_condtionIH", 
                     "Social Cue Nudge\n x IA Environment" = "social_cue_nudgeTrue:environment_condtionIA", 
                     "Baseline IH" = "scaled_pre_ih"), 
           model.names = c("Base", "Interaction", 
                           "Base with \nParticipant \nCovariates"), 
           legend.title = "Models")
p + theme(axis.text.y = element_text(size = 16),
          axis.title.x = element_text(size = 12), 
          legend.text = element_text(size = 14), # Adjust size as needed
          legend.title = element_text(size = 14))


plot_summs(reported_model_0, reported_model_1, reported_model_2, show.p = TRUE, scale=TRUE,
           coefs.match = c("Social Cue Nudge" = "social_cue_nudgeTrue", "IA Environment" = "environment_condtionIA",
                           "IH Environment" = "environment_condtionIH"),
           omit.coefs = c("(Intercept)", "stanceAbortion and Reproductive Rights - FOR", "stanceAbortion and Reproductive Rights - AGAINST",
                          "stanceUndocumented Immigrant Rights and Immigration Policy - AGAINST", "stanceUndocumented Immigrant Rights and Immigration Policy - FOR", 
                          "social_cue_nudgeTrue:stanceAbortion and Reproductive Rights - FOR", "social_cue_nudgeTrue:stanceAbortion and Reproductive Rights - AGAINST",
                          "stanceAction for Climate Change - FOR","stanceAction for Climate Change - AGAINST",
                          "social_cue_nudgeTrue:stanceAction for Climate Change - FOR",  "social_cue_nudgeTrue:stanceAction for Climate Change - AGAINST",
                          "social_cue_nudgeTrue:stanceUndocumented Immigrant Rights and Immigration Policy - AGAINST", "social_cue_nudgeTrue:stanceUndocumented Immigrant Rights and Immigration Policy - FOR",
                          "environment_condtionIA:stanceAbortion and Reproductive Rights - FOR", "environment_condtionIH:stanceAbortion and Reproductive Rights - FOR",
                          "environment_condtionIA:stanceAction for Climate Change - AGAINST","environment_condtionIH:stanceAction for Climate Change - AGAINST",
                          "environment_condtionIA:stanceAction for Climate Change - FOR","environment_condtionIH:stanceAction for Climate Change - FOR",
                          "environment_condtionIA:stanceUndocumented Immigrant Rights and Immigration Policy - FOR","environment_condtionIH:stanceUndocumented Immigrant Rights and Immigration Policy - FOR",
                          "environment_condtionIA:stanceUndocumented Immigrant Rights and Immigration Policy - AGAINST","environment_condtionIH:stanceUndocumented Immigrant Rights and Immigration Policy - AGAINST"
           ))

reported_plot + ggtitle("Model Coefficients \nChange in Reported Intellectual Humility")


# REPORTED IH MODELS
reported_model_0 <- lm(change_ih ~ social_cue_nudge + environment_condtion, data=df)
summ(reported_model_0, confint = TRUE, digits = 2)

reported_model_1 <- lm(change_ih ~ social_cue_nudge*environment_condtion, data=df)
summ(reported_model_1, confint = TRUE, digits = 2)

reported_model_2 <- lm(change_ih ~ social_cue_nudge + environment_condtion + stance, data=df)
summ(reported_model_2, confint = TRUE, digits = 2)


demonstrated_model_3 <- lm(demonstratedIH ~ social_cue_nudge + environment_condtion + stance + scaled_pre_ih + Sex + ageBuckets + affiliation, data=df)
summ(demonstrated_model_3, confint = TRUE, digits = 2)

reported_model_3 <- lm(change_ih ~ social_cue_nudge + environment_condtion + stance + Sex + ageBuckets + affiliation, data=df)
summ(reported_model_3, confint = TRUE, digits = 2)


# ENGAGEMENT MODELS
engagement_model1 <- lm(commentLength ~ social_cue_nudge + environment_condtion, data=df)
summ(engagement_model1, confint = TRUE, digits = 2)

engagement_model2 <- lm(comment ~ social_cue_nudge + environment_condtion, data=df)
summ(engagement_model2, confint = TRUE, digits = 2)

# Extract and label
tidy1 <- tidy(engagement_model1) %>% mutate(model = "Model 1: Comment Length")
tidy2 <- tidy(engagement_model2) %>% mutate(model = "Model 2: Number of Comments")

# Combine
all_coefs <- bind_rows(tidy1, tidy2)
ggplot(all_coefs, aes(x = term, y = estimate, fill = model)) +
  geom_bar(stat = "identity", position = position_dodge(width = 0.7)) +
  geom_errorbar(aes(ymin = estimate - std.error, ymax = estimate + std.error),
                position = position_dodge(width = 0.7), width = 0.2) +
  labs(title = "Engagement Model Coefficients", y = "Estimate", x = "Term") +
  theme_minimal()

# PLOT TWO VIOLIN
ggplot(df, aes(x = social_cue_nudge, y = normalized_demonstrated_ih, fill = environment_condition)) +
  geom_violin() +
  labs(title = "Distribution of Normalized Demonstrated IH by Social Cue Nudge and Environment Condition",
       x = "Social Cue Nudge", y = "Normalized Demonstrated IH") +
  scale_fill_manual(values = c("NEUTRAL" = "lightblue", "IA" = "lightgreen", "IH" = "salmon")) +
  theme_minimal()

#PLOT THREE FACET
ggplot(df, aes(x = environment_condition, y = normalized_demonstrated_ih, fill = environment_condition)) +
  geom_boxplot() +
  facet_wrap(~ social_cue_nudge) +
  labs(title = "Boxplot of Normalized Demonstrated IH by Environment Condition",
       x = "Environment Condition", y = "Normalized Demonstrated IH") +
  scale_fill_manual(values = c("NEUTRAL" = "lightblue", "IA" = "lightgreen", "IH" = "salmon")) +
  theme_minimal()
