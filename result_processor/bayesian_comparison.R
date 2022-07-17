if (!require("devtools")) {
  install.packages("devtools")
}

if (!require("scmamp")){
  devtools::install_github("b0rxa/scmamp")
}
if (!require("ggplot2")){
  install.packages("ggplot2", dependencies = TRUE)
  install.packages("geometry")
  install.packages("metRology")
  install.packages("MCMCpack")
}
if (!require("rstan")){
  install.packages("rstan")
}
if (!require("here")){
  install.packages("here")
}
if (!require("xtable")){
  install.packages("xtable")
}
if (!require("plyr")){
  install.packages("plyr")
}
library(reshape2)
library(stringr)

library("scmamp")
library("ggplot2")
library("here")
library(plyr)

if(dir.exists(file.path(getwd(), "result_processor/"))){
  setwd(file.path(getwd(), "result_processor/"))
}

DELETE_SEPSIS <- TRUE
DELETE_CAMARGO <- FALSE


data_per_fold <- read.csv("./data/paper_results/accuracy_raw_results_others.csv")
data_per_fold$X <- NULL
levels(data_per_fold$approach) <- c(levels(data_per_fold$approach), "ABASP")

# Read our results and append them to the previous results
files <- list.files("./data/paper_results/results/", recursive=TRUE, pattern="ggrnn_results.txt")
for (file in files) {
    line <- read.delim(file=paste("./data/paper_results/results/", file, sep=""), header=F, sep=":")
    fold <- gsub("train_fold(.).*", "\\1", file)
    log <- gsub("train_fold._variation._(.*).xes.gz/.*", "\\1", file)
    accuracy <- line$V2
    new_row <- list(approach="ABASP", accuracy=as.numeric(accuracy), fold=as.integer(fold), log=tolower(log))
    data_per_fold[nrow(data_per_fold) + 1, ] <- new_row
}
data_per_fold$accuracy <- as.numeric(data_per_fold$accuracy)
print(data_per_fold)
df_xd <- data_per_fold
df_xd["accuracy"] <- round(df_xd["accuracy"], 2)
data_per_fold_reshaped <- dcast(df_xd, log ~ approach + fold, value.var="accuracy")
write.csv(data_per_fold_reshaped, "./data/paper_results/full_results.csv")


# Group by approach and log and calculate the average accuracy
acc_data <- ddply(data_per_fold, .(approach, log), summarize, avg_accuracy=mean(accuracy) )


# Pivot table to have rows = log and columns = approach.
acc_data <- dcast(acc_data, log ~ approach, value.var="avg_accuracy")
rownames(acc_data) <- acc_data$log
colnames(acc_data)[colnames(acc_data) == "Theis_no_resource"] <- "Theis_no_attr"
colnames(acc_data)[colnames(acc_data) == "Theis_resource"] <- "Theis_attr"
colnames(acc_data)[colnames(acc_data) == "Pasquadibisceglie"] <- "Pasqua."
acc_data$log <- NULL
approaches <- colnames(acc_data)

if (!dir.exists(paste0("./data/paper_results/plots/"))){
  dir.create(paste0("./data/paper_results/plots/"), recursive = TRUE)
}
write.csv(acc_data, "./data/paper_results/plots/results_mean_acc_crossval.csv")

if (DELETE_SEPSIS) {
  acc_data <- acc_data[rownames(acc_data) != "nasa" & rownames(acc_data) != "sepsis",]
  subproblem <- "delete_sepsis"
}
if (DELETE_CAMARGO){
  #acc_data <- acc_data[, !(colnames(acc_data) %in% c("camargo"))]
  subproblem <- "delete_camargo"
}
if (!dir.exists(paste0("./data/paper_results/plots/", subproblem))){
  dir.create(paste0("./data/paper_results/plots/", subproblem), recursive = TRUE)
}
#acc_data <- t(acc_data)
print(acc_data)

data_matrix <- data.matrix(acc_data)
print(data_matrix)
results <- bPlackettLuceModel(data_matrix, min=FALSE, nsim=300000, nchains=10, parallel=TRUE, seed=42)
write.csv(results$expected.win.prob, file = paste0("./data/paper_results/plots/", subproblem, "/probabilities.csv"))
write.csv(results$expected.mode.rank, file = paste0("./data/paper_results/plots/", subproblem, "/rankings.csv"))
plackett_rownames <- names(results$expected.win.prob)
print(plackett_rownames)

# Plot posterior weights
probs <- data.frame(results$posterior.weights)
colnames(probs) <- plackett_rownames
print(colnames(probs))
stack_probs = stack(probs)
colnames(stack_probs)[colnames(stack_probs) == "ind"] <- "Approach"
colnames(stack_probs)[colnames(stack_probs) == "values"] <- "Probability"
stack_probs$Approach <- factor(stack_probs$Approach, levels=plackett_rownames)
print(unique(stack_probs$Approach))
dd_y95 <- ddply(stack_probs, .(Approach), function(x) quantile(x$Probability, 0.95))
dd_y05 <- ddply(stack_probs, .(Approach), function(x) quantile(x$Probability, 0.05))
dd_y50 <- ddply(stack_probs, .(Approach), function(x) median(x$Probability))
print(dd_y95)
df <- data.frame(
  x = plackett_rownames,
  y05 = dd_y05[2],
  y50 = dd_y50[2],
  y95 = dd_y95[2]
)
colnames(df) <- c("Approaches", "y05", "y50", "y95")
print(df)
png(paste0("./data/paper_results/plots/", subproblem, "/posterior_weights_", subproblem, ".png"), width=1000, height=400)
#ggplot(df, aes(x=Approaches)) + geom_boxplot(aes(ymin=y05, lower=y05, upper=y95, ymax=y95, middle=y50), stat="identity", width=0.5) + theme(text = element_text(size=20), axis.title.x=element_blank())
ggplot(df, aes(x=Approaches)) + geom_point(aes(y=y50), colour="blue", size=1.5) + geom_errorbar(aes(ymin=y05, ymax=y95), width=0.5) + theme(text = element_text(size=18), axis.title.x=element_blank())
dev.off()

index <- which(results$expected.mode.rank <= sort(results$expected.mode.rank, decreasing=FALSE)[3], arr.ind=TRUE)
weights <- results$posterior.weights[, index]
weights <- weights / rowSums(weights)
png(paste0("./data/paper_results/plots/", subproblem, "/baricentric_plackett.png"))
plotBarycentric(weights)
dev.off()


### Hierarchical tests
subset_1 <- data_per_fold[data_per_fold$approach == names(index[1]), ][c("log", "fold", "accuracy")]
reshaped_1 <- reshape(subset_1, direction="wide", idvar="log", timevar="fold")
rownames(reshaped_1) <- reshaped_1[,"log"]

subset_2 <- data_per_fold[data_per_fold$approach == names(index[2]), ][c("log", "fold", "accuracy")]
reshaped_2 <- reshape(subset_2, direction="wide", idvar="log", timevar="fold")
rownames(reshaped_2) <- reshaped_2[,"log"]

subset_3 <- data_per_fold[data_per_fold$approach == names(index[3]), ][c("log", "fold", "accuracy")]
reshaped_3 <- reshape(subset_3, direction="wide", idvar="log", timevar="fold")
rownames(reshaped_3) <- reshaped_3[,"log"]
reshaped_1$log <- NULL
reshaped_2$log <- NULL
reshaped_3$log <- NULL
matrix_1 <- data.matrix(reshaped_1)
matrix_2 <- data.matrix(reshaped_2)
matrix_3 <- data.matrix(reshaped_3)
ROPE <- c(-0.01, 0.02)

#if (DELETE_SEPSIS && (grepl("Camargo", names(index[1])) || grepl("Camargo" , names(index[2])))){
order_vector <- c("accuracy.0", "accuracy.1", "accuracy.2", "accuracy.3", "accuracy.4")
if (DELETE_SEPSIS){
  hier_matrix_1 <- matrix_1
  hier_matrix_1 <- hier_matrix_1[!rownames(hier_matrix_1) %in% c("sepsis", "nasa"), ]
  hier_matrix_1 <- hier_matrix_1[, order_vector]
  hier_matrix_1 <- hier_matrix_1[sort(rownames(hier_matrix_1)), ]
  hier_matrix_2 <- matrix_2
  hier_matrix_2 <- hier_matrix_2[!rownames(hier_matrix_2) %in% c("sepsis", "nasa"), ]
  hier_matrix_2 <- hier_matrix_2[, order_vector]
  hier_matrix_2 <- hier_matrix_2[sort(rownames(hier_matrix_2)), ]
} else {
  hier_matrix_1 <- matrix_1
  hier_matrix_2 <- matrix_2
}
print("matrix_1")
print(hier_matrix_1)
print("matrix_2")
print(hier_matrix_2)
results <- bHierarchicalTest(hier_matrix_1, hier_matrix_2, rho=0.2, rope=ROPE, nsim=75000, nchains=10, parallel=TRUE, seed=42)
filename_results_per_dataset <- c("./data/paper_results/plots/", subproblem, "/", "accuracy", "_hierarchical_test_", names(index)[1], "_vs_", names(index)[2], "_per_dataset.csv")
filename_left <- c("./data/paper_results/plots/", subproblem, "/", "accuracy", "_hierarchical_test_", names(index)[1], "_vs_", names(index)[2], "_matrix_left.csv")
filename_right <- c("./data/paper_results/plots/", subproblem, "/", "accuracy", "_hierarchical_test_", names(index)[1], "_vs_", names(index)[2], "_matrix_right.csv")
write.csv(hier_matrix_1, paste(filename_left, collapse=""))
write.csv(hier_matrix_2, paste(filename_right, collapse=""))
results_csv <- data.frame(results$additional$per.dataset)
results_csv["dataset"] <- rownames(hier_matrix_1)
write.csv(results_csv, paste(filename_results_per_dataset, collapse=""))
filename <- c("./data/paper_results/plots/", subproblem, "/", "accuracy", "_hierarchical_test_", names(index)[1], "_vs_", names(index)[2], ".png")
png(paste(filename, collapse=""))
plotSimplex(results, A=names(index[1]), B=names(index[2]), posterior.label=TRUE, alpha=0.5)
dev.off()
# P2
#if (DELETE_SEPSIS && (grepl("Camargo", names(index[3])) || grepl("Camargo" , names(index[2])))){
if (DELETE_SEPSIS){
  hier_matrix_3 <- matrix_3
  hier_matrix_3 <- hier_matrix_3[!rownames(hier_matrix_3) %in% c("sepsis", "nasa"), ]
  hier_matrix_3 <- hier_matrix_3[, order_vector]
  hier_matrix_3 <- hier_matrix_3[sort(rownames(hier_matrix_3)), ]
  hier_matrix_2 <- matrix_2
  hier_matrix_2 <- hier_matrix_2[!rownames(hier_matrix_2) %in% c("sepsis", "nasa"), ]
  hier_matrix_2 <- hier_matrix_2[, order_vector]
  hier_matrix_2 <- hier_matrix_2[sort(rownames(hier_matrix_2)), ]
} else{
  hier_matrix_3 <- matrix_3
  hier_matrix_2 <- matrix_2
}
results <- bHierarchicalTest(hier_matrix_2, hier_matrix_3, rho=0.2, rope=ROPE, nsim=75000, nchains=10, parallel=TRUE, seed=42)
filename_left <- c("./data/paper_results/plots/", subproblem, "/", "accuracy", "_hierarchical_test_", names(index)[2], "_vs_", names(index)[3], "_matrix_left.csv")
filename_right <- c("./data/paper_results/plots/", subproblem, "/", "accuracy", "_hierarchical_test_", names(index)[2], "_vs_", names(index)[3], "_matrix_right.csv")
filename_results_per_dataset <- c("./data/paper_results/plots/", subproblem, "/", "accuracy", "_hierarchical_test_", names(index)[2], "_vs_", names(index)[3], "_per_dataset.csv")
write.csv(hier_matrix_2, paste(filename_left, collapse=""))
write.csv(hier_matrix_3, paste(filename_right, collapse=""))
results_csv <- data.frame(results$additional$per.dataset)
results_csv["dataset"] <- rownames(hier_matrix_3)
write.csv(results_csv, paste(filename_results_per_dataset, collapse=""))
filename <- c("./data/paper_results/plots/", subproblem, "/", "accuracy", "_hierarchical_test_", names(index)[2], "_vs_", names(index)[3], ".png")
png(paste(filename, collapse=""))
plotSimplex(results, A=names(index[2]), B=names(index[3]), posterior.label=TRUE, alpha=0.5)
dev.off()
# P3
#if (DELETE_SEPSIS && (grepl("Camargo", names(index[3])) || grepl("Camargo" , names(index[1])))){
if (DELETE_SEPSIS){
  hier_matrix_3 <- matrix_3
  hier_matrix_3 <- hier_matrix_3[!rownames(hier_matrix_3) %in% c("sepsis", "nasa"), ]
  hier_matrix_3 <- hier_matrix_3[, order_vector]
  hier_matrix_3 <- hier_matrix_3[sort(rownames(hier_matrix_3)), ]
  hier_matrix_1 <- matrix_1
  hier_matrix_1 <- hier_matrix_1[!rownames(hier_matrix_1) %in% c("sepsis", "nasa"), ]
  hier_matrix_1 <- hier_matrix_1[, order_vector]
  hier_matrix_1 <- hier_matrix_1[sort(rownames(hier_matrix_1)), ]
} else {
  hier_matrix_3 <- matrix_3
  hier_matrix_1 <- matrix_1
}
results <- bHierarchicalTest(hier_matrix_1, hier_matrix_3, rho=0.2, rope=ROPE, nsim=75000, nchains=10, parallel=TRUE, seed=42)
filename_results_per_dataset <- c("./data/paper_results/plots/", subproblem, "/", "accuracy", "_hierarchical_test_", names(index)[1], "_vs_", names(index)[3], "_per_dataset.csv")
filename_left <- c("./data/paper_results/plots/", subproblem, "/", "accuracy", "_hierarchical_test_", names(index)[1], "_vs_", names(index)[3], "_matrix_left.csv")
filename_right <- c("./data/paper_results/plots/", subproblem, "/", "accuracy", "_hierarchical_test_", names(index)[1], "_vs_", names(index)[3], "_matrix_right.csv")
write.csv(hier_matrix_1, paste(filename_left, collapse=""))
write.csv(hier_matrix_3, paste(filename_right, collapse=""))
write.csv(results$additional$per.dataset, paste(filename_results_per_dataset, collapse=""))
filename <- c("./data/paper_results/plots/", subproblem, "/", "accuracy", "_hierarchical_test_", names(index)[1], "_vs_", names(index)[3], ".png")
png(paste(filename, collapse=""))
plotSimplex(results, A=names(index[1]), B=names(index[3]), posterior.label=TRUE, alpha=0.5)
dev.off()

# Save results per dataset
results_csv <- data.frame(results$additional$per.dataset)
results_csv["dataset"] <- rownames(hier_matrix_3)
filename_results_per_dataset <- c("./data/paper_results/plots/", subproblem, "/", "accuracy", "_hierarchical_test_", names(index)[1], "_vs_", names(index)[3], "_per_dataset.csv")
write.csv(results_csv, paste(filename_results_per_dataset, collapse=""))

# Perform the signed_rank_test pairwise
best_one <- acc_data[, names(index)[1]]
best_two <- acc_data[, names(index)[2]]
best_three <- acc_data[, names(index)[3]]
print("best one")
print(best_one)
print("best two")
print(best_two)
print("best three")
print(best_three)
# Pair 1
signed_test <- bSignedRankTest(best_one, best_two, rope=c(-0.1, 0.1), seed=42, nsim=10000)
filename <- c("./data/paper_results/plots/", subproblem, "/signed_rank_test_", names(index)[1], "_vs_", names(index)[2], "_", subproblem, ".png")
png(paste(filename, collapse=""))
plotSimplex(signed_test, A=names(index)[1], B=names(index)[2], plot.density=TRUE, alpha=0.5, posterior.label = TRUE)
dev.off()
# Pair 2
signed_test <- bSignedRankTest(best_two, best_three, rope=c(-0.1, 0.1), seed=42, nsim=10000)
filename <- c("./data/paper_results/plots/", subproblem, "/signed_rank_test_", names(index)[2], "_vs_", names(index)[3], "_", subproblem, ".png")
png(paste(filename, collapse=""))
plotSimplex(signed_test, A=names(index)[2], B=names(index)[3], plot.density=TRUE, alpha=0.5, posterior.label=TRUE)
dev.off()
# Pair 3
signed_test <- bSignedRankTest(best_one, best_three, rope=c(-0.1, 0.1), seed=42, nsim=10000)
filename <- c("./data/paper_results/plots/", subproblem, "/signed_rank_test_", names(index)[1], "_vs_", names(index)[3], "_", subproblem, ".png")
png(paste(filename, collapse=""))
plotSimplex(signed_test, A=names(index)[1], B=names(index)[3], plot.density=TRUE, alpha=0.5, posterior.label=TRUE)
dev.off()
