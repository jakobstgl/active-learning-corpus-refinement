source("preprocessing_functions.R")

library(tidyverse)
library(arrow)
library(stringr)
library(quanteda)
library(text2map)
library(tm)
library(tidyverse)
library(quanteda)
library(stopwords)

# We load all subcorpora including articles on various psychiatric disorders, and later filter to depression (psyC) and schizophrenia (psyB)
psyA <- read_parquet("../data/corpus_raw/snippets_psychA_0604.parquet")
psyB <- read_parquet("../data/corpus_raw/snippets_psychB_0604.parquet")
psyC <- read_parquet("../data/corpus_raw/snippets_psychC_0704.parquet")
psyD <- read_parquet("../data/corpus_raw/snippets_psychD_0604.parquet")
psyE <- read_parquet("../data/corpus_raw/snippets_psychE_0704.parquet")
psyF <- read_parquet("../data/corpus_raw/snippets_psychF_0704.parquet")
psyH <- read_parquet("../data/corpus_raw/snippets_psychH_0804.parquet")
psyI <- read_parquet("../data/corpus_raw/snippets_psychI_0804.parquet")
psyJ <- read_parquet("../data/corpus_raw/snippets_psychJ_0804.parquet")

corpora_raw_list <- list(psyA, psyB, psyC, psyD, psyE, psyF, psyH, psyI, psyJ)
names(corpora_raw_list) <- c("psyA", "psyB", "psyC", "psyD", "psyE", "psyF", "psyH", "psyI", "psyJ")
rm(psyA, psyB, psyC, psyD, psyE, psyF, psyH, psyI, psyJ)

# Basic cleaning, adding id variables, reducing to one dataframe
corpus_cleaned <- map(corpora_raw_list, basic_cleaning) |>
    imap(create_id_subcorpus) |> 
    reduce(rbind) |> 
    rowid_to_column(var = "id") |>
    mutate(
        id_subcorpus = as_factor(id_subcorpus),
        period = as_factor(period)
    ) 
    
# write_parquet(corpus_cleaned, "../data/corpus_cleaned/data_basiccleaned.parquet")
corpus_cleaned <- read_parquet("../data/corpus_cleaned/data_basiccleaned.parquet") |> 
    distinct(text, .keep_all = T)

# Save version with punctuation removed (for duplicate detection)
corpus_cleaned_nopunct <- corpus_cleaned |> 
    mutate(
        text = str_to_lower(text) |> 
            removePunctuation(
                preserve_intra_word_contractions = T, 
                preserve_intra_word_dashes = T)
    )

# Use only most frequent keywords as matched by regex, to exclude noise
most_frequent_keywords <- read_csv("../data/filtering_keywords.csv")
corpus_keywordcleaned_nopunct <- map(most_frequent_keywords[[1]], \(x) filtering_in_loop(corpus_cleaned_nopunct, x)) |> 
    reduce(rbind) |> 
    distinct(id, .keep_all = TRUE)


write_parquet(corpus_keywordcleaned_nopunct, "../data/corpus_cleaned/data_keywordcleaned_nopunct.parquet")
corpus_keywordcleaned_nopunct <- read_parquet("../data/corpus_cleaned/data_keywordcleaned_nopunct.parquet")


# Load distances from distance matching calculated in the respective python script
distances <- read_parquet("../data/cosine_distances/distances.parquet")
corpus_dplrm <- distance_matching(corpus_keywordcleaned_nopunct, distances, threshold = 0.85)

# Join on original corpus with punctuation and not lowercased for SentenceTransformer based Active Learning (al)
corpus_al <- left_join(corpus_dplrm, corpus_cleaned, by = c("id")) |> 
    select(-text.x) |> 
    rename(text = text.y)

write_parquet(corpus_al, "../data/data_activelearning.parquet")
corpus_al <- read_parquet("../data/data_activelearning.parquet")

# Keywords to replace
keywords <- most_frequent_keywords$keywords

# Named vector with replacements
replacements <- c(" alkohol_abhaengigkeit ", 
                  " drogen_abhaengigkeit ", 
                  " opioid_abhaengigkeit ", 
                  " heroin_abhaengigkeit ", 
                  " kokain_abhaengigkeit ", 
                  " medikamenten_abhaengigkeit ", 
                  " cannabis_abhaengigkeit ", 
                  " schizophrenie ", 
                  " psychose ", 
                  " wahnhafte_stoerung ", 
                  " depression ", 
                  " angst_stoerung ", 
                  " panik_stoerung ", 
                  " phobische_stoerung ", 
                  " ptbs ", 
                  " adhs ", 
                  " autismus ", 
                  " demenz ", 
                  " legasthenie ", 
                  " ess_stoerung ", 
                  " bulimia_nervosa ", 
                  " anorexia_nervosa ", 
                  " borderline_stoerung ", 
                  " persoenlichkeits_stoerung ", 
                  " abhaengigkeits_erkrankung ")
names(replacements) <- keywords

# Replace keywords, from here use text lowercased and without punctuation
corpus_replaced <- corpus_dplrm |> 
    mutate(
        text = str_to_lower(text) |>
            removePunctuation(
                preserve_intra_word_contractions = T, 
                preserve_intra_word_dashes = T) |> 
            str_replace_all(pattern = replacements),
        dup_id = id # duplicate the id field to enable computing stigma scores per snippet
    )

write_parquet(corpus_replaced, "../data/data_for_stigmascores.parquet")

