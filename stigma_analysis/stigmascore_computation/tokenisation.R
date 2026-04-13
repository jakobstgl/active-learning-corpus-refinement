library(tidyverse)
library(arrow)
library(quanteda)


# Load corpus -------------------------------------------------------------
corpus <- read_parquet("../data/preprocessing/data_for_stigmascores.parquet")

# tokenize, remove stopwords, numbers, symbols ----------------------------------------------------
prepare_corpus <- function(data){
    corpus <- quanteda::corpus(data, text_field = "text", docid_field = "id")
    
    toks <- quanteda::tokens(corpus, remove_symbols=T, remove_numbers=T, remove_separators=T)
    
    toks_nostop <- quanteda::tokens_select(toks, pattern = stopwords::stopwords("de"), selection = "remove", min_nchar=3)
    
    return(list(toks, toks_nostop))
}

tokens <- prepare_corpus(data = corpus)
tokens_nostopwords <- tokens[[2]]
tokens_stopwords <- tokens[[1]]

# further process tokenized corpus -----------------------------------------------
process_tokens <- function(tokens, min_termfreq = 10){
    
    if(typeof(tokens) == "character"){
        toks <- readr::read_rds(tokens)
    }
    
    else {
        toks <- tokens
    }
    
    feats_dfm <- quanteda::dfm(toks, tolower=T, verbose = FALSE) |>
        quanteda::dfm_trim(min_termfreq = 10)
    
    # Create tokens objekt  ---------------------------------------------------
    join_toks_feats <- function(x){
        feats <- quanteda::featnames(x)
        toks_feats <- quanteda::tokens_select(toks, feats, padding = T)
    }
    
    return(toks_feats <- join_toks_feats(feats_dfm))
}

tokens_nostopwords_processed <- process_tokens(tokens_nostopwords, min_termfreq = 10) #minimum term frequency set to 10
tokens_stopwords_processed <- process_tokens(tokens_stopwords, min_termfreq = 10) 

# save to rds -------------------------------------------------------------
write_rds(tokens_nostopwords_processed, "data/preprocessing/toks_nostop_processed.rds")
#write_rds(tokens_stopwords_processed, "data/preprocessing/toks_processed.rds")

