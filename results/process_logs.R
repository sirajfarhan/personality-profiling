library(dplyr)
library(data.table)

data_without_nmf <- fread("logs.csv")
data_with_nmf <- fread("logs_nmf.csv")

data_without_nmf <- data_without_nmf %>%
    group_by(metric, category, fold) %>%
    summarize(value = max(value)) %>%
    ungroup %>%
    group_by(metric, category) %>%
    summarize(value_without_nmf=mean(value)) %>%
    ungroup %>%
    mutate(id = paste(metric,category,sep="."))

data_with_nmf <- data_with_nmf %>%
    group_by(metric, category, fold) %>%
    summarize(value = max(value)) %>%
    ungroup %>%
    group_by(metric, category) %>%
    summarize(value_with_nmf=mean(value)) %>%
    ungroup %>%
    mutate(id = paste(metric,category,sep="."))

data <- merge(select(data_without_nmf, id, value_without_nmf), data_with_nmf, by="id") %>%
    mutate(improvement = value_with_nmf - value_without_nmf) %>%
    select(-id) %>%
    select(metric, category, value_with_nmf, value_without_nmf, improvement)

write.csv(data, "results.csv", row.names = F)




